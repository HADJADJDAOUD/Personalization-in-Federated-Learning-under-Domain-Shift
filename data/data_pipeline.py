"""
data_pipeline.py — Federated Heart Disease Dataset Pipeline
Team 8-HANTA

Replaces ad-hoc preprocessing scattered across notebooks with a single,
reproducible pipeline that every experiment notebook can import.

Usage
-----
    from data_pipeline import load_client_data, DatasetReport

    # Default: impute → scale → split 80/20 stratified
    client_data = load_client_data()

    # Custom settings
    client_data = load_client_data(
        data_dir="data/heart_disease_dataset",
        test_size=0.25,
        impute_strategy="median",   # "median" | "mean" | "knn" | "drop"
        scale=True,
        random_state=42,
    )

    # Each value is a dict with keys: X_train, X_test, y_train, y_test
    X_train_cleveland = client_data["Cleveland"]["X_train"]

    # Print a data quality report
    DatasetReport(client_data).print()

Federated split note
--------------------
The four hospital files ARE the natural federated partition (one per client /
"center").  No artificial splitting is needed.  Each center's data is loaded,
cleaned, and split independently so no test-set information leaks across
clients.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Column schema ────────────────────────────────────────────────────────────

COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Which columns are continuous (will be scaled / imputed numerically)
CONTINUOUS_COLS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Which columns are categorical (imputed with mode, never scaled)
CATEGORICAL_COLS: list[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

# Source files → human-readable center names
CENTER_FILES: dict[str, str] = {
    "processed.cleveland.data":   "Cleveland",
    "processed.hungarian.data":   "Hungary",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data":          "LongBeach",
}

ImputeStrategy = Literal["median", "mean", "knn", "drop"]

# ── Per-center data container ────────────────────────────────────────────────

@dataclass
class CenterData:
    """Holds train/test arrays for one federated client (hospital center)."""
    center: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    n_train_original: int          # rows before any dropping
    n_missing_rows: int            # rows that had at least one '?' before imputation

    # Convenience properties
    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def positive_rate_train(self) -> float:
        return float(self.y_train.mean())

    def as_dict(self) -> dict:
        """Return the dict format expected by existing src/fedavg.py & src/pfedme.py."""
        return {
            "X_train": self.X_train,
            "X_test":  self.X_test,
            "y_train": self.y_train,
            "y_test":  self.y_test,
        }


# ── Core loading function ────────────────────────────────────────────────────

def load_center(
    filepath: str,
    center_name: str,
    impute_strategy: ImputeStrategy = "median",
    scale: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
) -> CenterData:
    """
    Load, clean, and split one center's .data file.

    Parameters
    ----------
    filepath        : path to processed.*.data file
    center_name     : human-readable label ("Cleveland", etc.)
    impute_strategy : how to handle '?' missing values
                      "median" — fill continuous with median, categorical with mode
                      "mean"   — fill continuous with mean,   categorical with mode
                      "knn"    — KNN imputation (k=5) across all features
                      "drop"   — drop any row with a missing value
    scale           : if True, StandardScaler fitted on train, applied to test
    test_size       : fraction held out for evaluation
    random_state    : reproducibility seed

    Returns
    -------
    CenterData
    """
    # ── 1. Read raw file ─────────────────────────────────────────────────────
    df = pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    n_original = len(df)
    n_missing_rows = int(df.isnull().any(axis=1).sum())

    # ── 2. Binarise target (0 = healthy, 1 = disease) ────────────────────────
    df["target"] = (df["target"] > 0).astype(int)

    # ── 3. Separate features / label ─────────────────────────────────────────
    feature_cols = COLUMNS[:-1]   # everything except "target"
    X = df[feature_cols].copy()
    y = df["target"].values

    # ── 4. Impute missing values ──────────────────────────────────────────────
    if impute_strategy == "drop":
        mask = ~df[feature_cols].isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        if len(y) == 0:
            raise ValueError(
                f"[{center_name}] All rows dropped — choose a different impute_strategy."
            )
    elif impute_strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    else:
        # Continuous columns: median or mean
        num_strategy = "median" if impute_strategy == "median" else "mean"
        for col in CONTINUOUS_COLS:
            if X[col].isnull().any():
                fill = X[col].median() if num_strategy == "median" else X[col].mean()
                X[col] = X[col].fillna(fill)
        # Categorical columns: mode
        for col in CATEGORICAL_COLS:
            if X[col].isnull().any():
                mode_val = X[col].mode()
                fill = mode_val.iloc[0] if not mode_val.empty else 0
                X[col] = X[col].fillna(fill)

    # ── 5. Train / test split (stratified) ───────────────────────────────────
    X_arr = X.values.astype(np.float32)
    y_arr = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_state,
    )

    # ── 6. Feature scaling (fit on train only) ────────────────────────────────
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    return CenterData(
        center=center_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        n_train_original=n_original,
        n_missing_rows=n_missing_rows,
    )


def load_client_data(
    data_dir: str = "data/heart_disease_dataset",
    impute_strategy: ImputeStrategy = "median",
    scale: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, dict]:
    """
    Load all four centers and return the dict format used by fedavg.py / pfedme.py.

    Returns
    -------
    {
      "Cleveland":    {"X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...},
      "Hungary":      {...},
      "Switzerland":  {...},
      "LongBeach":    {...},
    }
    """
    client_data: dict[str, dict] = {}
    missing: list[str] = []

    for filename, center_name in CENTER_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing.append(filepath)
            warnings.warn(f"[DataPipeline] File not found, skipping: {filepath}")
            continue

        center = load_center(
            filepath=filepath,
            center_name=center_name,
            impute_strategy=impute_strategy,
            scale=scale,
            test_size=test_size,
            random_state=random_state,
        )
        client_data[center_name] = center.as_dict()

    if not client_data:
        raise FileNotFoundError(
            f"No center files found in '{data_dir}'. "
            "Run data/downloader_script.py first, or check the path."
        )

    return client_data


def load_client_data_detailed(
    data_dir: str = "data/heart_disease_dataset",
    impute_strategy: ImputeStrategy = "median",
    scale: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, CenterData]:
    """
    Same as load_client_data() but returns CenterData objects instead of plain dicts.
    Useful for inspecting metadata (missing value counts, positive rates, etc.).
    """
    result: dict[str, CenterData] = {}

    for filename, center_name in CENTER_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            warnings.warn(f"[DataPipeline] File not found, skipping: {filepath}")
            continue

        result[center_name] = load_center(
            filepath=filepath,
            center_name=center_name,
            impute_strategy=impute_strategy,
            scale=scale,
            test_size=test_size,
            random_state=random_state,
        )

    return result


# ── Dataset quality report ────────────────────────────────────────────────────

@dataclass
class DatasetReport:
    """
    Print a concise data quality summary for all loaded centers.

    Example
    -------
        centers = load_client_data_detailed()
        DatasetReport(centers).print()
    """
    centers: dict[str, CenterData]

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for name, c in self.centers.items():
            rows.append({
                "Center":          name,
                "Total rows":      c.n_train_original,
                "Rows w/ missing": c.n_missing_rows,
                "Missing %":       f"{100 * c.n_missing_rows / max(c.n_train_original, 1):.1f}%",
                "Train N":         c.n_train,
                "Test N":          c.n_test,
                "Pos rate (train)": f"{c.positive_rate_train:.3f}",
            })
        return pd.DataFrame(rows).set_index("Center")

    def print(self) -> None:
        df = self.summary_df()
        print("\n" + "=" * 60)
        print("  DATASET QUALITY REPORT")
        print("=" * 60)
        print(df.to_string())
        print()

        # Domain-shift warning: flag centers where positive rate differs > 0.15
        rates = {n: c.positive_rate_train for n, c in self.centers.items()}
        mean_rate = np.mean(list(rates.values()))
        print(f"  Mean positive rate across centers: {mean_rate:.3f}")
        flagged = [n for n, r in rates.items() if abs(r - mean_rate) > 0.15]
        if flagged:
            print(f"  ⚠  High domain shift detected in: {', '.join(flagged)}")
            print("     (positive rate deviates > 15pp from mean)")
        else:
            print("  ✓  No severe class-distribution shift detected.")
        print("=" * 60 + "\n")


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/heart_disease_dataset"
    print(f"Loading data from: {data_dir}\n")

    centers = load_client_data_detailed(data_dir=data_dir)

    if not centers:
        print("No data loaded. Pass the correct path as the first argument.")
        sys.exit(1)

    DatasetReport(centers).print()

    # Confirm the dict format is compatible with fedavg.py
    client_data = {name: c.as_dict() for name, c in centers.items()}
    first = next(iter(client_data.values()))
    assert "X_train" in first and "y_train" in first, "Dict format mismatch"
    print("✓  Dict format compatible with fedavg.py / pfedme.py")
    print(f"   Feature count: {first['X_train'].shape[1]}")
    print(f"   Centers loaded: {list(client_data.keys())}")