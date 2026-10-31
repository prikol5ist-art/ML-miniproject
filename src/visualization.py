"""High-resolution visualization generation for empirical diagnostics.

Includes scree curves, normalized confusion heatmaps, PCA inverse reconstruction grids,
high-confidence error galleries, and Pareto frontier scatter plots.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data import CLASS_NAMES


def set_academic_style() -> None:
    """Configure matplotlib/seaborn parameters for publication-quality figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })


def plot_scree_and_validation(
    cumulative_variance: np.ndarray,
    k_results: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative explained variance alongside cross-validation Macro-F1 across PCA components."""
    set_academic_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    components_range = np.arange(1, len(cumulative_variance) + 1)
    ax1.plot(
        components_range,
        cumulative_variance * 100.0,
        color="#2b5c8f",
        linewidth=2.2,
        label="Cumulative Explained Variance (%)",
    )
    ax1.set_xlabel("Number of Principal Components ($k$)")
    ax1.set_ylabel("Explained Variance (%)", color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.set_ylim(40, 100)

    ax1.axhline(80, color="#2b5c8f", linestyle=":", alpha=0.6, label="80% Variance Threshold")
    ax1.axvline(50, color="#d95f02", linestyle="--", alpha=0.7, label="$k = 50$ (Elbow / Benchmark)")

    ax2 = ax1.twinx()
    ax2.errorbar(
        k_results["k"],
        k_results["Macro-F1 (Mean)"],
        yerr=k_results["Macro-F1 (Std)"],
        color="#d95f02",
        marker="o",
        linewidth=2.0,
        capsize=4,
        label="LinearSVC Macro-F1 (CV)",
    )
    ax2.set_ylabel("Validation Macro-F1 Score", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax2.set_ylim(0.70, 0.90)

    plt.title("Information Preservation: Scree Curve & LinearSVC Validation Macro-F1", pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
    title: str = "Normalized Confusion Matrix (RBF-SVC, PCA $k=50$)",
) -> plt.Figure:
    """Plot row-normalized confusion matrix highlighting upper-wear confusion clusters."""
    set_academic_style()
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pca_reconstruction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    k_values: List[int] = [10, 50, 200],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize original vs PCA reconstructed images to illustrate spatial information loss."""
    set_academic_style()
    target_classes = [0, 2, 4, 6, 8, 9]
    selected_indices = []
    for cls in target_classes:
        idx = np.where(y_train == cls)[0][0]
        selected_indices.append(idx)

    X_sample = X_train[selected_indices]
    y_sample = y_train[selected_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)

    reconstructions = {}
    for k in k_values:
        pca = PCA(n_components=k, random_state=42)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_sample_scaled)
        X_rec_scaled = pca.inverse_transform(X_pca)
        X_rec = scaler.inverse_transform(X_rec_scaled)
        reconstructions[k] = np.clip(X_rec, 0, 255)

    n_rows = len(target_classes)
    n_cols = 1 + len(k_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))

    col_titles = ["Original (784 px)"] + [f"PCA $k={k}$" for k in k_values]

    for row_idx, cls in enumerate(target_classes):
        label = class_names[cls]
        axes[row_idx, 0].imshow(X_sample[row_idx].reshape(28, 28), cmap="gray_r")
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        for col_offset, k in enumerate(k_values):
            col_idx = col_offset + 1
            axes[row_idx, col_idx].imshow(reconstructions[k][row_idx].reshape(28, 28), cmap="gray_r")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, pad=10)

    fig.suptitle("PCA Inverse Reconstruction $\\tilde{\\mathbf{X}} = \\mathbf{X}_{\\mathrm{pca}}\\mathbf{W}^T + \\boldsymbol{\\mu}$ Across Classes", y=1.01, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_high_confidence_errors(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    decision_margins: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    top_n: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Display visual gallery of misclassifications with highest decision margin violations."""
    set_academic_style()
    error_mask = (y_test != y_pred)
    error_indices = np.where(error_mask)[0]

    sorted_error_idx = error_indices[np.argsort(-decision_margins[error_indices])]
    top_errors = sorted_error_idx[:top_n]

    rows = 2
    cols = int(np.ceil(top_n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(top_errors):
        ax = axes[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        true_name = class_names[y_test[idx]]
        pred_name = class_names[y_pred[idx]]
        margin = decision_margins[idx]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}\nMargin: +{margin:.2f}", fontsize=10, color="#a50026")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("High-Confidence Decision Boundary Violations (False Positives)", y=1.02, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Macro-F1 vs Training Time to evaluate computational efficiency trade-offs."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    colors = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#e6ab02", "#386cb0"]

    for i, row in results_df.iterrows():
        name = row["Model"]
        f1_mean = row["Macro-F1 (Mean)"]
        f1_std = row["Macro-F1 (Std)"]
        time_sec = row["Fit Time (s)"]
        color = colors[i % len(colors)]

        ax.errorbar(
            time_sec,
            f1_mean,
            yerr=f1_std,
            fmt="o",
            color=color,
            markersize=9,
            capsize=4,
            label=name,
        )

        offset_y = 0.012 if "Hist" not in name else -0.02
        offset_x = 1.05
        ax.annotate(
            name.split(". ")[-1],
            (time_sec * offset_x, f1_mean + offset_y),
            fontsize=9.5,
            fontweight="semibold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total 5-Fold Training Time (seconds, log scale)")
    ax.set_ylabel("Macro-Averaged F1-Score (5-Fold CV)")
    ax.set_title("Pareto Efficiency: Macro-F1 vs Training Time on CPU", pad=12)
    ax.set_ylim(0.05, 0.95)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
