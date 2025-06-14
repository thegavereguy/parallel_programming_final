import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# FLOPS per punto della griglia per ogni timestep. DA ADATTARE AL TUO CODICE.
FLOPS_PER_POINT = 3  # Aggiornato a un valore più realistico per FTCS

# Performance di picco teorica della CPU in MFLOP/s
PEAK_PERFORMANCE = 6451.0
# Banda di memoria teorica in GB/s
MEMORY_BANDWIDTH = 140 * 4


def parse_filename(filename):
    """
    formato  {omp/mpi}_{thread/proc}_{ex/im}[_{variant}].csv
    """
    match = re.search(r"^(omp|mpi)_(\d+)_(ex|im)(?:_([a-zA-Z0-9_]+))?\.csv$", filename)
    if match:
        run_type, units_str, method_type, variant = match.groups()
        units = int(units_str)
        method_name = f"{run_type.upper()}"
        if variant:
            method_name += f" {variant.replace('_', ' ').capitalize()}"
        method_name += f" {method_type.capitalize()}"
        return {"method_base": method_name, "units": units}
    return {"method_base": "Unknown", "units": 0}


def load_and_process_data(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    all_files = [f for f in all_files if "cpu_info" not in os.path.basename(f)]

    if not all_files:
        print(
            f"Nessun file .csv trovato nella directory '{folder}'. Esegui prima i benchmark."
        )
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df["filename"] = os.path.basename(f)
            df_list.append(df)
        except Exception as e:
            print(f"Impossibile leggere o elaborare il file {f}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    parsed_info = full_df["filename"].apply(parse_filename).apply(pd.Series)
    full_df = pd.concat([full_df, parsed_info], axis=1)

    full_df["Implementation"] = full_df.apply(
        lambda row: (
            f"{row['method_base']} ({row['units']} u)"
            if row["units"] > 0
            else "Unknown"
        ),
        axis=1,
    )

    def get_test_case_label(row):
        name_col = row.get("NAME", "")
        nx_val = row["NX"]
        nt_val = row["NT"]
        if pd.notna(name_col) and isinstance(name_col, str):
            descriptive_name = name_col.strip().replace("_", " ")
            return f"{descriptive_name}\n(NX={nx_val} NT={nt_val})"
        else:
            return f"Problem\n(NX={nx_val})"

    full_df["TestCaseLabel"] = full_df.apply(get_test_case_label, axis=1)

    total_flops = full_df["NX"] * full_df["NT"] * FLOPS_PER_POINT
    full_df["GFLOPs/s"] = (total_flops / (full_df["MEAN"] * 1e-3)) / 1e9

    bytes_per_point = 2 * 8
    total_bytes = full_df["NX"] * full_df["NT"] * bytes_per_point
    full_df["Arithmetic Intensity"] = total_flops.divide(total_bytes).fillna(0)

    full_df = full_df[full_df["method_base"] != "Unknown"].copy()

    # 1 lettura e 1 scrittura di un double unto per passo temporale.
    bytes_per_point = 2 * 8
    total_bytes = full_df["NX"] * full_df["NT"] * bytes_per_point
    full_df["Arithmetic Intensity"] = total_flops.divide(total_bytes).fillna(0)

    execution_time_sec = full_df["MEAN"] * 1e-3
    full_df["Achieved Memory Bandwidth (GB/s)"] = (
        total_bytes / execution_time_sec
    ) / 1e9

    full_df = full_df[full_df["method_base"] != "Unknown"].copy()
    return full_df
    return full_df


def plot_performance_barchart(df, save_path, opt_level):

    df = df[~df["NAME"].str.contains("Weak", case=False, na=False)].copy()

    df_sorted = df.sort_values(by=["NX", "units"])
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(20, 12))
    unique_implementations = df_sorted["Implementation"].nunique()
    palette = sns.color_palette("tab20", n_colors=unique_implementations)
    ax = sns.barplot(
        data=df_sorted,
        x="TestCaseLabel",
        y="GFLOPs/s",
        hue="Implementation",
        palette=palette,
    )
    ax.set_title(
        f"Confronto Performance per Caso di Test (Opt: {opt_level})",
        fontsize=20,
        pad=20,
    )
    ax.set_xlabel("Caso di Test", fontsize=16)
    ax.set_ylabel("Performance (GFLOPs/s)", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=11)
    plt.yscale("log")
    plt.legend(title="Implementazione", fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grafico a barre salvato in: {save_path}")
    plt.close()


def plot_strong_scaling_speedup(df, save_path, opt_level):
    df = df[~df["NAME"].str.contains("Weak", case=False, na=False)].copy()
    baselines = df[df["units"] == 1].copy()
    if baselines.empty:
        print(
            f"[{opt_level}] Nessuna esecuzione di base (1 unità) trovata per lo Strong Scaling."
        )
        return

    baseline_map = baselines.set_index(["TestCaseLabel", "method_base"])[
        "MEAN"
    ].to_dict()
    df_speedup = df[df["units"] > 0].copy()
    df_speedup["BaselineTime"] = df_speedup.apply(
        lambda row: baseline_map.get((row["TestCaseLabel"], row["method_base"])), axis=1
    )
    df_speedup.dropna(subset=["BaselineTime"], inplace=True)

    if df_speedup.empty:
        print(f"[{opt_level}] Nessuna corrispondenza per lo Strong Scaling.")
        return

    df_speedup["Speedup"] = df_speedup["BaselineTime"] / df_speedup["MEAN"]
    test_cases_with_speedup = df_speedup[df_speedup["units"] > 1][
        "TestCaseLabel"
    ].unique()

    if len(test_cases_with_speedup) == 0:
        print(f"[{opt_level}] Nessun dato di Strong Scaling valido da plottare.")
        return

    df_to_plot = df_speedup[df_speedup["TestCaseLabel"].isin(test_cases_with_speedup)]
    df_to_plot = df_to_plot.sort_values(by="units").copy()

    g = sns.FacetGrid(
        df_to_plot,
        col="TestCaseLabel",
        hue="method_base",
        col_wrap=3,
        height=5,
        sharey=False,
        legend_out=True,
    )
    g.map(plt.plot, "units", "Speedup", marker="o", ms=8)

    for ax in g.axes.flat:
        title = ax.get_title()
        ax.plot([1, 32], [1, 32], "k--", zorder=1)

    g.add_legend(title="Metodo Base")
    g.set_axis_labels("Numero di Unità (Thread/Processi)", "Speedup")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"Strong Scaling Speedup (Opt: {opt_level})", y=1.03, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    print(f"Grafico Strong Scaling salvato in: {save_path}")
    plt.close()


def plot_roofline_model(df, peak_performance, memory_bandwidth, save_path, opt_level):
    df = df[~df["NAME"].str.contains("Weak", case=False, na=False)].copy()
    if df.empty:
        print(
            f"[{opt_level}] Nessun dato per il Roofline Model dopo aver escluso i test 'Weak'."
        )
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensità Aritmetica (FLOP/Byte)", fontsize=14)
    ax.set_ylabel("Performance (GFLOPs/s)", fontsize=14)
    ax.set_title(f"Roofline Model (Opt: {opt_level})", fontsize=18, pad=20)

    ai_range = np.logspace(-2, 4, 100)
    ai_knee = peak_performance / memory_bandwidth
    performance_roof = np.minimum(peak_performance, memory_bandwidth * ai_range)

    ax.plot(ai_range, performance_roof, color="black", lw=2, label="Tetto Teorico")
    ax.text(
        ai_range[-1],
        peak_performance,
        f"π = {peak_performance:.1f} GFLOP/s",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    ax.text(
        ai_range[0],
        memory_bandwidth * ai_range[0] * 1.2,
        f"β = {memory_bandwidth:.1f} GB/s",
        ha="left",
        va="bottom",
        fontsize=12,
        rotation=40,
        rotation_mode="anchor",
    )

    sns.scatterplot(
        data=df,
        x="Arithmetic Intensity",
        y="GFLOPs/s",
        hue="Implementation",
        ax=ax,
        style="TestCaseLabel",
        s=150,
        zorder=10,
    )

    ax.grid(True, which="both", ls="--")
    ax.legend(
        title="Implementazione / Caso di Test",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(save_path)
    print(f"Grafico Roofline Model salvato in: {save_path}")
    plt.close()


def plot_weak_scaling_efficiency(df, save_path, opt_level):
    df_weak = df[df["NAME"].str.contains("Weak", case=False, na=False)].copy()
    if df_weak.empty:
        print(f"[{opt_level}] Nessun test di Weak Scaling trovato.")
        return

    baselines = df_weak.loc[df_weak.groupby(["NAME", "method_base"])["units"].idxmin()]
    baseline_map = baselines.set_index(["NAME", "method_base"])["MEAN"].to_dict()

    df_weak["BaselineTime"] = df_weak.apply(
        lambda row: baseline_map.get((row["NAME"], row["method_base"])), axis=1
    )
    df_weak.dropna(subset=["BaselineTime"], inplace=True)

    df_weak["Efficiency"] = df_weak["BaselineTime"] / (
        df_weak["MEAN"] * df_weak["units"]
    )
    df_weak = df_weak.sort_values(by="units").copy()

    g = sns.FacetGrid(
        df_weak,
        col="NAME",
        hue="method_base",
        col_wrap=3,
        height=5,
        sharey=True,
        legend_out=True,
    )
    g.map(plt.plot, "units", "Efficiency", marker="o", ms=8)

    for ax in g.axes.flat:
        ax.axhline(1.0, ls="--", color="k", zorder=1)
        ax.set_ylim(bottom=0)

    g.add_legend(title="Metodo Base")
    g.set_axis_labels("Numero di Unità (Thread/Processi)", "Efficienza Parallela")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"Weak Scaling Efficiency (Opt: {opt_level})", y=1.03, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    print(f"Grafico Weak Scaling salvato in: {save_path}")
    plt.close()


def plot_achieved_bandwidth(df, theoretical_bandwidth, save_path, opt_level):

    df = df[~df["NAME"].str.contains("Weak", case=False, na=False)].copy()
    df_sorted = df.sort_values(by=["NX", "units"])
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(20, 12))
    unique_implementations = df_sorted["Implementation"].nunique()
    palette = sns.color_palette("viridis", n_colors=unique_implementations)

    ax = sns.barplot(
        data=df_sorted,
        x="TestCaseLabel",
        y="Achieved Memory Bandwidth (GB/s)",
        hue="Implementation",
        palette=palette,
    )

    # Aggiunge una linea per il picco teorico come riferimento
    ax.axhline(
        theoretical_bandwidth,
        ls="--",
        color="red",
        lw=2,
        label=f"Picco Teorico ({theoretical_bandwidth} GB/s)",
    )

    ax.set_title(
        f"Banda di Memoria Effettiva per Caso di Test (Opt: {opt_level})",
        fontsize=20,
        pad=20,
    )
    ax.set_xlabel("Caso di Test", fontsize=16)
    ax.set_ylabel("Banda di Memoria Effettiva (GB/s)", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=11)

    plt.legend(title="Implementazione", fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grafico banda di memoria salvato in: {save_path}")
    plt.close()


def main():
    """
    Scansiona le sottocartelle in RESULTS_DIR, ognuna rappresentante un livello
    di ottimizzazione, e genera un set di grafici per ciascuna.
    """
    if not os.path.exists(RESULTS_DIR):
        print(f"La cartella dei risultati '{RESULTS_DIR}' non esiste.")
        return

    # Trova tutte le sottocartelle in RESULTS_DIR (es. 'O2', 'O3')
    try:
        optimization_levels = [
            d
            for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))
        ]
    except FileNotFoundError:
        print(f"Errore nell'accesso a '{RESULTS_DIR}'.")
        return

    if not optimization_levels:
        print(f"Nessuna sottocartella di ottimizzazione trovata in '{RESULTS_DIR}'.")
        print(
            "Lo script si aspetta una struttura tipo 'results/O2/', 'results/O3/', ecc."
        )
        return

    # Itera su ogni livello di ottimizzazione trovato
    for opt_level in optimization_levels:
        print(f"\n--- Elaborazione livello di ottimizzazione: {opt_level} ---")
        current_results_dir = os.path.join(RESULTS_DIR, opt_level)
        current_plots_dir = os.path.join(PLOTS_DIR, opt_level)

        # Crea la cartella di output per i grafici se non esiste
        os.makedirs(current_plots_dir, exist_ok=True)

        # Carica e processa i dati per il livello di ottimizzazione corrente
        full_df = load_and_process_data(current_results_dir)

        if full_df.empty:
            print(
                f"Nessun dato da processare per il livello '{opt_level}'. Salto al prossimo."
            )
            continue

        print(f"Dati per '{opt_level}' caricati. Generazione grafici...")

        # Genera tutti i grafici per il set di dati corrente
        plot_performance_barchart(
            full_df,
            os.path.join(current_plots_dir, "performance_barchart.png"),
            opt_level,
        )
        plot_strong_scaling_speedup(
            full_df,
            os.path.join(current_plots_dir, "strong_scaling_speedup.png"),
            opt_level,
        )
        plot_roofline_model(
            full_df,
            PEAK_PERFORMANCE,
            MEMORY_BANDWIDTH,
            os.path.join(current_plots_dir, "roofline_model.png"),
            opt_level,
        )
        plot_weak_scaling_efficiency(
            full_df,
            os.path.join(current_plots_dir, "weak_scaling_efficiency.png"),
            opt_level,
        )
        plot_achieved_bandwidth(
            full_df,
            MEMORY_BANDWIDTH,
            os.path.join(current_plots_dir, "achieved_bandwidth.png"),
            opt_level,
        )

    print("\n--- Elaborazione completata. ---")


if __name__ == "__main__":
    main()
