import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)

from sklearn.calibration import calibration_curve

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit



def evaluate_model(y_true, y_proba):
    return {
        "roc_auc": roc_auc_score(
            y_true,
            y_proba
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_proba
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_proba
        )
    }


def threshold_analysis(
    y_true,
    y_proba,
    thresholds
):

    results = []

    for threshold in thresholds:

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "predicted_positive_rate": y_pred.mean()
        })

    return pd.DataFrame(results)


def ranking_analysis(
    y_true,
    y_proba,
    k_values
):

    ranking = pd.DataFrame({
        "y": np.asarray(y_true),
        "proba": y_proba
    }).sort_values(
        "proba",
        ascending=False
    )

    total_positives = ranking["y"].sum()
    baseline_rate = ranking["y"].mean()

    results = []

    for k in k_values:

        n = max(
            1,
            int(
                np.ceil(
                    len(ranking) * k
                )
            )
        )

        selected = ranking.head(n)

        true_positives = selected["y"].sum()

        precision_at_k = (
            true_positives / n
        )

        recall_at_k = (
            true_positives / total_positives
            if total_positives > 0
            else 0
        )

        lift_at_k = (
            precision_at_k / baseline_rate
        )

        results.append({
            "top_k": k,
            "n_users": n,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "lift_at_k": lift_at_k
        })

    return pd.DataFrame(results)


def plot_roc_curves(
    y_true,
    predictions,
    title="ROC Curve"
):

    plt.figure(figsize=(10, 6))

    for name, y_proba in predictions.items():

        fpr, tpr, _ = roc_curve(
            y_true,
            y_proba
        )

        auc = roc_auc_score(
            y_true,
            y_proba
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_precision_recall_curves(
    y_true,
    predictions,
    title="Precision-Recall Curve"
):

    baseline = y_true.mean()

    plt.figure(figsize=(10, 6))

    for name, y_proba in predictions.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                y_proba
            )
        )

        ap = average_precision_score(
            y_true,
            y_proba
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.4f})"
        )

    plt.axhline(
        baseline,
        linestyle="--",
        label=f"Baseline ({baseline:.4f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_calibration(
    y_true,
    probabilities,
    title="Calibration Curve",
    n_bins=10
):
    """
    Plot calibration curve(s) for one or multiple probability predictions.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    probabilities : array-like or dict
        Predicted probabilities.

        If array-like:
            A single calibration curve is plotted.

        If dict:
            Keys are curve labels and values are predicted probabilities.

    title : str
        Plot title.

    n_bins : int
        Number of calibration bins.
    """

    if isinstance(probabilities, dict):
        probability_sets = probabilities
    else:
        probability_sets = {"Model": probabilities}

    plt.figure(figsize=(8, 6))

    # Perfect calibration reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    for label, y_proba in probability_sets.items():

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_proba,
            n_bins=n_bins,
            strategy="quantile"
        )

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=label
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def generate_oof_predictions(
    model,
    X,
    y,
    cv
):
    oof_predictions = []
    oof_indices = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):

        fold_model = clone(model)

        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]

        X_fold_val = X.iloc[val_idx]

        fold_model.fit(
            X_fold_train,
            y_fold_train
        )

        fold_proba = (
            fold_model
            .predict_proba(X_fold_val)[:, 1]
        )

        oof_predictions.extend(fold_proba)
        oof_indices.extend(val_idx)

        print(
            f"Fold {fold}: "
            f"train={len(train_idx):,}, "
            f"validation={len(val_idx):,}"
        )

    return (
        np.array(oof_indices),
        np.array(oof_predictions)
    )


def logit(p):
    p = np.clip(
        p,
        1e-6,
        1 - 1e-6
    )

    return np.log(
        p / (1 - p)
    )


def calibrate_probability(
    probabilities,
    calibrator
):
    probabilities_logit = logit(
        probabilities
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        probabilities_logit
    )[:, 1]