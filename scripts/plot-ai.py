import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Load all CSV files from the "result" folder
def load_data(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    df_list = [pd.read_csv(f).assign(filename=os.path.basename(f)) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

# Compute arithmetic intensity and performance (GFLOPs/s)
def compute_metrics(df):
    df["FLOPs"] = df["NX"] * df["NT"] * 8  # Each iteration has 5 floating-point operations per grid point
    df["Memory Access"] = df["NX"] * df["NT"] * 3 * 4  # 3 floats per iteration, 4 bytes per float
    df["Arithmetic Intensity"] = df["FLOPs"] / df["Memory Access"]
    df["GFLOPs/s"] = df["FLOPs"] / (df["MEAN"] * 1e-3) / 1e9  # Convert to GFLOPs per second
    return df

# Plot Roofline model
def plot_roofline(df, max_bandwidth, max_throughput):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot empirical data
    sns.scatterplot(x=df["Arithmetic Intensity"], y=df["GFLOPs/s"], hue=df["filename"], s=100)
    
    # Plot roofline model
    x_vals = np.logspace(-2, 3, 100)
    roofline = np.minimum(max_bandwidth * x_vals, max_throughput)
    plt.plot(x_vals, roofline, linestyle="--", color="black", label="Roofline")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOPs per byte)")
    plt.ylabel("Performance (GFLOPs/s)")
    plt.title("1D Heat Equation Roofline Model")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    folder = "tmp"  # Folder containing CSV files
    df = load_data(folder)
    df = compute_metrics(df)
    
    max_bandwidth = float(input("Enter max memory bandwidth (GB/s): "))
    max_throughput = float(input("Enter max compute throughput (GFLOPs/s): "))
    
    plot_roofline(df, max_bandwidth, max_throughput)
