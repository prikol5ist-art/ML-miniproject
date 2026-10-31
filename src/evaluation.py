"""Rigorous 5-fold cross-validation and holdout evaluation engine.

Measures Macro-averaged F1, Accuracy, training duration, and inference latency.
"""

from typing import Any, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from src.data import CLASS_NAMES
from src.models import get_linear_svc_pipeline


def run_cross_validation(
    models_dict: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute stratified k-fold cross-validation across all models.

    Args:
        models_dict: Mapping from model name to Estimator or Pipeline.
        X: Training features array.
        y: Training labels array.
        cv_splits: Number of stratified folds.
        random_state: Seed for fold shuffling.

    Returns:
        DataFrame with Macro-F1 mean/std, Accuracy mean/std, and fit/score runtimes.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_macro"]

    records = []
    for name, model in models_dict.items():
        t0 = time.time()
        cv_res = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        wall_time = time.time() - t0

        records.append({
            "Model": name,
            "Macro-F1 (Mean)": float(np.mean(cv_res["test_f1_macro"])),
            "Macro-F1 (Std)": float(np.std(cv_res["test_f1_macro"])),
            "Accuracy (Mean)": float(np.mean(cv_res["test_accuracy"])),
            "Accuracy (Std)": float(np.std(cv_res["test_accuracy"])),
            "Fit Time (s)": float(np.sum(cv_res["fit_time"])),
            "Score Time (s)": float(np.sum(cv_res["score_time"])),
            "Wall Time (s)": float(wall_time),
        })

    return pd.DataFrame(records)


def run_pca_sweep(
    X: np.ndarray,
    y: np.ndarray,
    k_values: List[int] = [10, 20, 50, 100, 200],
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sweep PCA component count k for LinearSVC and compute cumulative explained variance.

    Args:
        X: Training features array.
        y: Training labels array.
        k_values: List of k components to evaluate.
        cv_splits: Number of stratified folds.
        random_state: Seed for reproducibility.

    Returns:
        (results_df, cumulative_variance_array)
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    records = []

    for k in k_values:
        pipeline = get_linear_svc_pipeline(n_components=k, random_state=random_state)
        cv_res = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
        )
        records.append({
            "k": k,
            "Macro-F1 (Mean)": float(np.mean(cv_res["test_f1_macro"])),
            "Macro-F1 (Std)": float(np.std(cv_res["test_f1_macro"])),
            "Accuracy (Mean)": float(np.mean(cv_res["test_accuracy"])),
            "Fit Time (s)": float(np.mean(cv_res["fit_time"])),
        })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA(n_components=max(k_values), random_state=random_state)
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    return pd.DataFrame(records), cumulative_variance


def evaluate_holdout(
    pipeline: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Train pipeline on training partition and perform a locked evaluation on holdout test set.

    Args:
        pipeline: Estimator or Pipeline.
        X_train: Training features.
        y_train: Training labels.
        X_test: Locked test features.
        y_test: Locked test labels.

    Returns:
        (metrics_dict, y_pred, confusion_mat, decision_margins)
    """
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - t0

    t1 = time.time()
    y_pred = pipeline.predict(X_test)
    inference_duration = time.time() - t1

    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    decision_margins = np.zeros(len(y_test), dtype=np.float32)
    if hasattr(pipeline, "decision_function"):
        df_vals = pipeline.decision_function(X_test)
        if df_vals.ndim == 2:
            pred_scores = df_vals[np.arange(len(y_test)), y_pred]
            true_scores = df_vals[np.arange(len(y_test)), y_test]
            decision_margins = pred_scores - true_scores
    elif hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)
        pred_probs = probs[np.arange(len(y_test)), y_pred]
        true_probs = probs[np.arange(len(y_test)), y_test]
        decision_margins = pred_probs - true_probs

    metrics = {
        "Accuracy": test_acc,
        "Macro-F1": test_f1,
        "Train Duration (s)": train_duration,
        "Inference Duration (s)": inference_duration,
        "Classification Report": report,
    }

    return metrics, y_pred, cm, decision_margins
