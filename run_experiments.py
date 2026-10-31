#!/usr/bin/env python3
"""Main CLI entrypoint for Fashion-MNIST empirical benchmarking.

Executes stratified cross-validation, PCA component sweep, holdout test evaluation,
and saves all 5 diagnostic figures to the figures/ directory.
Completes in <90 seconds on standard CPU hardware under default quick mode.
"""

import argparse
import os
import sys
import time
import pandas as pd

from src.data import CLASS_NAMES, get_stratified_split
from src.evaluation import evaluate_holdout, run_cross_validation, run_pca_sweep
from src.models import build_benchmark_models, get_rbf_svc_pipeline
from src.visualization import (
    plot_confusion_matrix,
    plot_high_confidence_errors,
    plot_pareto_frontier,
    plot_pca_reconstruction,
    plot_scree_and_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST Dimensionality Reduction & Classifier Benchmark"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick",
        dest="quick",
        action="store_true",
        default=True,
        help="Run on stratified 12,000-sample subsample (<90s runtime, default).",
    )
    group.add_argument(
        "--full",
        dest="quick",
        action="store_false",
        help="Run on full 60,000 training set (requires ~15-25 minutes).",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Directory to save generated visual figures.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 80)
    print("NaUKMA Machine Learning Mini-Project: Fashion-MNIST PCA & Classifier Benchmark")
    print(f"Mode: {'Quick (12,000 train samples)' if args.quick else 'Full (60,000 train samples)'}")
    print(f"Seed: {args.random_state} | Figures Directory: {args.figures_dir}/")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/5] Loading Fashion-MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = get_stratified_split(
        quick=args.quick,
        random_state=args.random_state,
    )
    print(f"--> Training partition: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"--> Holdout test set:   {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # 2. Benchmark Suite Cross-Validation
    print("\n[Step 2/5] Running Stratified 5-Fold Cross-Validation on Benchmark Suite...")
    models = build_benchmark_models(random_state=args.random_state)
    benchmark_df = run_cross_validation(
        models,
        X_train,
        y_train,
        cv_splits=5,
        random_state=args.random_state,
    )

    print("\n" + "-" * 88)
    print("BENCHMARK CROSS-VALIDATION SUMMARY (5-Fold Stratified CV):")
    print("-" * 88)
    print(benchmark_df.to_string(index=False))
    print("-" * 88)

    # 3. PCA Component Sweep
    print("\n[Step 3/5] Running PCA Dimensionality Sweep (k in [10, 20, 50, 100, 200])...")
    k_results, cumulative_variance = run_pca_sweep(
        X_train,
        y_train,
        k_values=[10, 20, 50, 100, 200],
        cv_splits=5,
        random_state=args.random_state,
    )
    print("\nPCA SWEEP SUMMARY:")
    print(k_results.to_string(index=False))

    # 4. Final Holdout Evaluation (Locked Test Set)
    print("\n[Step 4/5] Evaluating Winning Model (RBF-SVC, PCA 50) on Locked 10,000 Test Set...")
    winning_model = get_rbf_svc_pipeline(
        n_components=50,
        C=5.0,
        gamma="scale",
        random_state=args.random_state,
    )
    metrics, y_pred, cm, decision_margins = evaluate_holdout(
        winning_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print(f"--> Test Macro-F1: {metrics['Macro-F1']:.4f}")
    print(f"--> Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"--> Fit Duration:  {metrics['Train Duration (s)']:.2f}s")
    print(f"--> Inference:     {metrics['Inference Duration (s)']:.2f}s")

    # 5. Generate and Save Figures
    print("\n[Step 5/5] Generating Diagnostic Visualizations...")
    fig1_path = os.path.join(args.figures_dir, "scree_and_validation_curve.png")
    plot_scree_and_validation(cumulative_variance, k_results, save_path=fig1_path)
    print(f"--> Saved: {fig1_path}")

    fig2_path = os.path.join(args.figures_dir, "confusion_matrix_heatmap.png")
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path=fig2_path)
    print(f"--> Saved: {fig2_path}")

    fig3_path = os.path.join(args.figures_dir, "pca_inverse_reconstruction.png")
    plot_pca_reconstruction(X_train, y_train, class_names=CLASS_NAMES, save_path=fig3_path)
    print(f"--> Saved: {fig3_path}")

    fig4_path = os.path.join(args.figures_dir, "high_confidence_errors.png")
    plot_high_confidence_errors(X_test, y_test, y_pred, decision_margins, class_names=CLASS_NAMES, save_path=fig4_path)
    print(f"--> Saved: {fig4_path}")

    fig5_path = os.path.join(args.figures_dir, "pareto_frontier.png")
    plot_pareto_frontier(benchmark_df, save_path=fig5_path)
    print(f"--> Saved: {fig5_path}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE: All tasks finished successfully in {total_elapsed:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
