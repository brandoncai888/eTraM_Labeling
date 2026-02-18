import h5py
import pandas as pd
import numpy as np
import sys
from pathlib import Path

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False


def load_h5(file_path, key=None):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if key and key in f:
            target = f[key]
        elif key is None and len(keys) > 0:
            target = f[keys[0]]
        else:
            raise ValueError(f"Key '{key}' not found. Available keys: {keys}")

        if isinstance(target, h5py.Group):
            data = {k: target[k][()] for k in target.keys()}
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(target[()])


def load_npy(file_path):
    data = np.load(file_path, allow_pickle=True)
    return pd.DataFrame(data)


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_parquet(file_path):
    return pd.read_parquet(file_path)


def load_feather(file_path):
    return pd.read_feather(file_path)


def save_h5(df, file_path, key='data'):
    with h5py.File(file_path, 'w') as f:
        group = f.create_group(key)
        for col in df.columns:
            group.create_dataset(col, data=df[col].values)
    print(f"Saved to {file_path}")


def save_npy(df, file_path):
    np.save(file_path, df.to_records(index=False))
    print(f"Saved to {file_path}")


def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")


def save_parquet(df, file_path):
    df.to_parquet(file_path, index=False)
    print(f"Saved to {file_path}")


def save_feather(df, file_path):
    df.to_feather(file_path)
    print(f"Saved to {file_path}")


LOAD_FUNCS = {
    '.h5':      load_h5,
    '.npy':     load_npy,
    '.csv':     load_csv,
    '.parquet': load_parquet,
    '.feather': load_feather,
}

SAVE_FUNCS = {
    '.h5':      save_h5,
    '.npy':     save_npy,
    '.csv':     save_csv,
    '.parquet': save_parquet,
    '.feather': save_feather,
}


def extract(input_path, h5_key=None, use_cudf=False):
    """
    Load any supported file and return a DataFrame.

    Args:
        input_path: Path to the input file (.h5, .npy, .csv, .parquet, .feather)
        h5_key:     For HDF5 files, which key/group to load (default: first key)
        use_cudf:   If True, return a cudf.DataFrame instead of pandas (requires cuDF/RAPIDS)

    Returns:
        pd.DataFrame or cudf.DataFrame
    """
    if use_cudf and not HAS_CUDF:
        raise ImportError("cudf is not installed. Install RAPIDS: https://rapids.ai/start.html")

    ext = Path(input_path).suffix.lower()
    if ext not in LOAD_FUNCS:
        supported = ", ".join(sorted(LOAD_FUNCS.keys()))
        raise ValueError(f"Unsupported format: {ext}\nSupported: {supported}")

    df = LOAD_FUNCS[ext](input_path, h5_key) if ext == '.h5' else LOAD_FUNCS[ext](input_path)

    if use_cudf:
        return cudf.DataFrame.from_pandas(df)
    return df


def convert(input_path, output_path, input_key=None):
    input_ext = Path(input_path).suffix.lower()
    output_ext = Path(output_path).suffix.lower()

    supported = ", ".join(sorted(LOAD_FUNCS.keys()))

    if input_ext not in LOAD_FUNCS:
        raise ValueError(f"Unsupported input format: {input_ext}\nSupported: {supported}")
    if output_ext not in SAVE_FUNCS:
        raise ValueError(f"Unsupported output format: {output_ext}\nSupported: {supported}")

    print(f"Loading {input_path}...")
    df = LOAD_FUNCS[input_ext](input_path, input_key) if input_ext == '.h5' else LOAD_FUNCS[input_ext](input_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    print(f"Saving to {output_path}...")
    SAVE_FUNCS[output_ext](df, output_path)
    print("Conversion complete.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_data.py <input_file> <output_file> [h5_key]")
        print("\nSupported formats: .feather, .parquet, .h5, .npy, .csv")
        print("\nExamples:")
        print("  python extract_data.py data.h5 data.csv")
        print("  python extract_data.py data.npy data.parquet")
        print("  python extract_data.py data.csv data.feather")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
