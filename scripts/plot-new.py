import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
FLOPS_PER_POINT = 3


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

        # Cerca i nomi scrittivi dai benchmark CPU. Formato atteso: "Nome_Test,NX,NT"
        if pd.notna(name_col) and isinstance(name_col, str):
            descriptive_name = name_col.strip().replace("_", " ")
            return f"{descriptive_name}\n(NX={nx_val} NT={nt_val})"
        else:
            return f"Problem\n(NX={nx_val})"

    full_df["TestCaseLabel"] = full_df.apply(get_test_case_label, axis=1)

    full_df["GFLOPs/s"] = (
        (full_df["NX"] * full_df["NT"] * FLOPS_PER_POINT)
        / (full_df["MEAN"] * 1e-3)
        / 1e9
    )

    full_df = full_df[full_df["method_base"] != "Unknown"].copy()

    return full_df


def plot_performance_barchart(df, save_path):
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

    ax.set_title("Confronto Performance per Caso di Test", fontsize=20, pad=20)
    ax.set_xlabel("Caso di Test", fontsize=16)
    ax.set_ylabel("Performance (GFLOPs/s)", fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=11)

    plt.yscale("log")
    plt.legend(title="Implementazione", fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Grafico a barre delle performance salvato in: {save_path}")
    plt.close()


def plot_strong_scaling_speedup(df, save_path):
    """
    Genera un grafico dello speedup rispetto alla versione a 1 unità.
    """
    baselines = df[df["units"] == 1].copy()
    if baselines.empty:
        print(
            "Nessuna esecuzione di base (1 thread/processo) trovata. Impossibile calcolare lo speedup."
        )
        return

    baseline_map = baselines.set_index(["method_base", "NX"])["MEAN"].to_dict()
    df_speedup = df[df["units"] > 1].copy()
    df_speedup["BaselineTime"] = df_speedup.apply(
        lambda row: baseline_map.get((row["method_base"], row["NX"])), axis=1
    )
    df_speedup.dropna(subset=["BaselineTime"], inplace=True)

    if df_speedup.empty:
        print(
            "Nessuna corrispondenza trovata tra esecuzioni parallele e di base. Impossibile calcolare lo speedup."
        )
        return

    df_speedup["Speedup"] = df_speedup["BaselineTime"] / df_speedup["MEAN"]

    plt.style.use("seaborn-v0_8-whitegrid")
    g = sns.FacetGrid(
        df_speedup,
        col="NX",
        hue="method_base",
        col_wrap=4,
        height=5,
        sharey=False,
        legend_out=True,
    )
    g.map(plt.plot, "units", "Speedup", marker="o", ms=8)

    for ax in g.axes.flat:
        title_text = ax.get_title()
        try:
            nx_val = float(re.search(r"NX = ([\d.]+)", title_text).group(1))
            max_units_for_nx = df_speedup[df_speedup.NX == nx_val]["units"].max()
            ax.plot(
                [1, max_units_for_nx],
                [1, max_units_for_nx],
                "k--",
                label="Speedup Ideale",
            )
        except (AttributeError, ValueError):
            # se no riesce a parsare il titolo salta la linea ideale per quel subplot
            pass

    g.add_legend(title="Metodo Base")
    g.set_axis_labels("Numero di Unità (Thread/Processi)", "Speedup")
    g.set_titles("NX = {col_name}")
    g.fig.suptitle(
        "Strong Scaling Speedup per Dimensione del Problema", y=1.03, fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    print(f"Grafico dello speedup salvato in: {save_path}")
    plt.close()


def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    full_df = load_and_process_data(RESULTS_DIR)

    if full_df.empty:
        return

    print("Dati caricati ed elaborati con successo. Riepilogo:")
    print(
        full_df[
            ["TestCaseLabel", "Implementation", "units", "NX", "MEAN", "GFLOPs/s"]
        ].round(2)
    )

    plot_performance_barchart(
        full_df, os.path.join(PLOTS_DIR, "performance_barchart.png")
    )
    plot_strong_scaling_speedup(
        full_df, os.path.join(PLOTS_DIR, "strong_scaling_speedup.png")
    )


if __name__ == "__main__":
    main()
