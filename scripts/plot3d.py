import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load all CSV files from the "result" folder
def load_data(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    df_list = [pd.read_csv(f).assign(filename=os.path.basename(f)) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

# Compute performance in FLOPs (Floating Point Operations per Second)
def compute_flops(df):
    df["FLOPs"] = df["NX"] * df["NT"] * 5  # Total floating-point operations
    df["GFLOPs/s"] = df["FLOPs"] / (df["MEAN"] * 1e-3) / 1e9  # Convert to GFLOPs per second
    return df

# Plot performance in 3D
def plot_performance_3d(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for filename in df["filename"].unique():
        subset = df[df["filename"] == filename]
        ax.scatter(subset["NX"], subset["NT"], subset["GFLOPs/s"], label=filename)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_zscale("log")
    ax.set_xlabel("Space Division (NX)")
    ax.set_ylabel("Time Division (NT)")
    ax.set_zlabel("Performance (GFLOPs/s)")
    ax.set_title("1D Heat Equation Performance")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    folder = "results"  # Folder containing CSV files
    df = load_data(folder)
    df = compute_flops(df)
    plot_performance_3d(df)

