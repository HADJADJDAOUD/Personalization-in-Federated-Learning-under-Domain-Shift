"""Preprocessing utilities for the Heart Disease FL dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES_BASE = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

ENGINEERED_FEATURES = [
    "chol_trestbps_ratio",
    "thalach_age_ratio",
    "oldpeak_sq",
    "trestbps_high",
    "chol_high",
    "age_bucket",
]


@dataclass
class PreprocessArtifacts:
    imputer: SimpleImputer
    scaler: StandardScaler
    clip_bounds: Dict[str, Tuple[float, float]]
    feature_names: List[str]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired features using already-imputed inputs."""
    df = df.copy()
    df["chol_trestbps_ratio"] = df["chol"] / (df["trestbps"] + 1.0)
    df["thalach_age_ratio"] = df["thalach"] / (df["age"] + 1.0)
    df["oldpeak_sq"] = df["oldpeak"] ** 2
    df["trestbps_high"] = (df["trestbps"] >= 140).astype(int)
    df["chol_high"] = (df["chol"] >= 240).astype(int)
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 120], labels=False).astype(int)
    return df


def _fit_clip_bounds(df: pd.DataFrame, features: List[str], q_low: float, q_high: float) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for feature in features:
        lower = df[feature].quantile(q_low)
        upper = df[feature].quantile(q_high)
        bounds[feature] = (lower, upper)
    return bounds


def _apply_clip_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for feature, (lower, upper) in bounds.items():
        df[feature] = df[feature].clip(lower=lower, upper=upper)
    return df


def _split_train_val_test(
    X: pd.DataFrame,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=y,
        random_state=seed,
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_client_datasets(
    df: pd.DataFrame,
    centers: List[str],
    base_features: List[str] | None = None,
    include_engineered: bool = True,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, PreprocessArtifacts]]:
    """Prepare per-center splits with imputation, feature engineering, clipping, and scaling."""
    if base_features is None:
        base_features = FEATURES_BASE

    client_data: Dict[str, Dict[str, np.ndarray]] = {}
    artifacts: Dict[str, PreprocessArtifacts] = {}

    for center in centers:
        df_center = df[df["center"] == center].copy()
        X_raw = df_center[base_features]
        y = df_center["target"].values

        X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(
            X_raw,
            y,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=base_features)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=base_features)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=base_features)

        if include_engineered:
            X_train_imp = add_engineered_features(X_train_imp)
            X_val_imp = add_engineered_features(X_val_imp)
            X_test_imp = add_engineered_features(X_test_imp)

        feature_names = list(X_train_imp.columns)

        bounds = _fit_clip_bounds(X_train_imp, feature_names, clip_quantiles[0], clip_quantiles[1])
        X_train_clip = _apply_clip_bounds(X_train_imp, bounds)
        X_val_clip = _apply_clip_bounds(X_val_imp, bounds)
        X_test_clip = _apply_clip_bounds(X_test_imp, bounds)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clip)
        X_val_scaled = scaler.transform(X_val_clip)
        X_test_scaled = scaler.transform(X_test_clip)

        client_data[center] = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
        artifacts[center] = PreprocessArtifacts(
            imputer=imputer,
            scaler=scaler,
            clip_bounds=bounds,
            feature_names=feature_names,
        )

    return client_data, artifacts
