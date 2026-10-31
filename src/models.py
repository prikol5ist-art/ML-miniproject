"""Scikit-learn Pipeline definitions ensuring zero data leakage.

Every model encapsulates feature scaling and PCA decomposition inside
Pipeline objects so transformations are fitted exclusively on training folds.
"""

from typing import Dict, Optional
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC


def get_dummy_baseline(random_state: int = 42) -> DummyClassifier:
    """Stratified empirical chance baseline (expected Macro-F1 = 0.10)."""
    return DummyClassifier(strategy="stratified", random_state=random_state)


def get_logistic_regression_pipeline(
    max_iter: int = 500,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """L2-regularized multinomial logistic regression on standardized 784 pixels."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def get_linear_svc_pipeline(
    n_components: Optional[int] = None,
    C: float = 1.0,
    max_iter: int = 1500,
    tol: float = 1e-3,
    random_state: int = 42,
) -> Pipeline:
    """Linear Support Vector Classifier with optional PCA dimensionality reduction."""
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    steps.append(("clf", LinearSVC(
        dual="auto",
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )))
    return Pipeline(steps)


def get_rbf_svc_pipeline(
    n_components: int = 50,
    C: float = 5.0,
    gamma: str = "scale",
    random_state: int = 42,
) -> Pipeline:
    """Non-linear Gaussian Kernel Support Vector Classifier on PCA components."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("clf", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            random_state=random_state,
        )),
    ])


def get_hist_gradient_boosting_pipeline(
    n_components: Optional[int] = None,
    max_iter: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Pipeline:
    """Histogram-based Gradient Boosted Trees operating on binned pixel features."""
    steps = []
    if n_components is not None:
        steps.append(("scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    steps.append(("clf", HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_state,
    )))
    return Pipeline(steps)


def build_benchmark_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Instantiate the primary benchmark suite comparing baselines, linear, and non-linear models."""
    return {
        "1. Dummy Baseline": get_dummy_baseline(random_state=random_state),
        "2. Logistic Regression (Raw 784)": get_logistic_regression_pipeline(random_state=random_state),
        "3. LinearSVC (Raw 784)": get_linear_svc_pipeline(n_components=None, random_state=random_state),
        "4. LinearSVC (PCA 50)": get_linear_svc_pipeline(n_components=50, random_state=random_state),
        "5. RBF-SVC (PCA 50)": get_rbf_svc_pipeline(n_components=50, C=5.0, random_state=random_state),
        "6. HistGradientBoosting (Raw 784)": get_hist_gradient_boosting_pipeline(n_components=None, random_state=random_state),
    }
