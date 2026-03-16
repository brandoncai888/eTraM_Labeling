import argparse
import pandas as pd
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score
)

def get_cluster_labels(df, filename):
    """
    Extracts the cluster column from the DataFrame.
    Looks for 'cluster_id' first, then falls back to 'cluster'.
    """
    if 'cluster_id' in df.columns:
        return df['cluster_id']
    elif 'cluster' in df.columns:
        return df['cluster']
    else:
        raise ValueError(f"Neither 'cluster_id' nor 'cluster' column found in {filename}.")

def evaluate_clusters(file1, file2, method='all'):
    """
    Core function to evaluate clustering performance between two parquet files.
    """
    # Load the parquet files
    try:
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Validate that the files have the same number of events
    if len(df1) != len(df2):
        print(f"Error: Files have a different number of events. File 1: {len(df1)}, File 2: {len(df2)}")
        return

    # Extract the cluster columns, allowing for either name
    try:
        labels1 = get_cluster_labels(df1, file1)
        labels2 = get_cluster_labels(df2, file2)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Dictionary mapping the argument string to the evaluation function
    metrics_dict = {
        'rand': ('Rand Index (RI)', rand_score),
        'ari':  ('Adjusted Rand Index (ARI)', adjusted_rand_score),
        'nmi':  ('Normalized Mutual Information (NMI)', normalized_mutual_info_score),
        'ami':  ('Adjusted Mutual Information (AMI)', adjusted_mutual_info_score)
    }

    if method not in metrics_dict and method != 'all':
        print(f"Error: Invalid method '{method}'. Choose from: rand, ari, nmi, ami, all.")
        return

    # Determine which methods to execute
    methods_to_run = list(metrics_dict.keys()) if method == 'all' else [method]

    # Calculate and print the scores
    print(f"Comparing:\n  1) {file1}\n  2) {file2}\n")
    for m in methods_to_run:
        name, func = metrics_dict[m]
        score = func(labels1, labels2)
        print(f"{name}: {score:.5f}")

def main():
    # Set up argument parsing for command line usage
    parser = argparse.ArgumentParser(description="Evaluate clustering metrics between two identical-length parquet files.")
    parser.add_argument("file1", type=str, help="Path to the first .parquet file.")
    parser.add_argument("file2", type=str, help="Path to the second .parquet file.")
    parser.add_argument(
        "-m", "--method",
        type=str,
        required=True,
        choices=['rand', 'ari', 'nmi', 'ami', 'all'],
        help="Clustering evaluation method to use: 'rand', 'ari', 'nmi', 'ami', or 'all'."
    )

    args = parser.parse_args()
    
    # Pass command line arguments to the core function
    evaluate_clusters(args.file1, args.file2, args.method)

if __name__ == "__main__":
    # ---------------------------------------------------------
    # OPTION 1: Run via command line (CLI)
    # ---------------------------------------------------------
    # Comment this out if you only want to use Option 2
    # main()

    # ---------------------------------------------------------
    # OPTION 2: Run directly with hardcoded arguments
    # ---------------------------------------------------------
    # Uncomment the line below to run without command line arguments.
    # Replace the file paths and method with your actual values.
    #
    evaluate_clusters("data/E_patch_stdbscan.parquet", "data/E_patch_dstream_2x2.parquet", method="all")