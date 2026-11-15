# """Helpers to evaluate churn models with ROC and Precision-Recall curves.

# Typical use in a notebook:

#     from evaluation import get_oof_proba, plot_roc_pr, threshold_for_recall

#     probas = {
#         'logreg': get_oof_proba(models['logreg'], X_train, y_train, seed=SEED),
#         'rf':     get_oof_proba(models['rf'],     X_train, y_train, seed=SEED),
#     }
#     plot_roc_pr(y_train, probas, target_recall=0.80, min_precision=0.65)
#     threshold_for_recall(y_train, probas['logreg'], target_recall=0.80)
# """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from data_visualizer import plot_line_chart, create_fig, fig_suptitle_format

def get_oof_proba(
        model : Pipeline,
        X,
        y,
        seed : int,
        n_splits : int = 5
    ) -> np.ndarray:
    """Out-of-fold P(churn=1) for every row, so no row is scored by a model that saw it.

    The whole pipeline (preprocessing + model) is refit on each fold.
    Use the TRAIN set here; keep the test set for the final evaluation.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        random_state=seed,
        shuffle=True
    )

    return cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method='predict_proba'
        )[:, 1]

def get_threshold_for_recall(
        y_true,
        y_proba,
        target_recall : float    
    ) -> dict | None:
    """Highest threshold that still reaches `target_recall`, with the precision you get there."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # drop the last entry in precision and recall
    # the last recall index that meets target_recall is the highest threshold that does
    # (recall never increases as threshold rises, only stays the same or goes down)

    meets_criteria = np.where(recall[:-1] >= target_recall)[0]
    if len(meets_criteria) == 0:
        return None
    
    best_idx = meets_criteria[-1]
    return {
            'threshold': float(thresholds[best_idx]),
            'recall':float(recall[best_idx]),
            'precision':float(precision[best_idx])
        }

def plot_roc(
        y_true,
        y_proba,
        ax,
        plot_line_args:dict | None = None
    ):

    false_positive_rates, true_positive_rates, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(false_positive_rates, true_positive_rates)
    
    if plot_line_args is None:
            plot_line_args = {}
    
    args : dict = {
        'title':f'Receiver Operating Characteristic (AUC = {roc_auc:.4f})',
        'xlabel':'False Positive Rates',
        'ylabel':'True Positive Rates',
        'color':'#004f8f',
        **plot_line_args
    }

    plot_line_chart(
        ax=ax,
        x=false_positive_rates,
        y=true_positive_rates,
        **args
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])


def plot_pr(
        y_true,
        y_proba,
        ax,
        plot_line_args:dict | None = None
    ):

    precision_rates, recall_rates, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_rates, precision_rates)

    if plot_line_args is None:
        plot_line_args = {}
    args : dict = {
        'title':f'Precision-Recall (AUC = {pr_auc:.4f})',
        'xlabel':'Recall',
        'ylabel':'Precision',
        'color':'#004f8f',
        **plot_line_args
    }
    plot_line_chart(
        ax=ax,
        x=recall_rates,
        y=precision_rates,
        **args
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

def plot_roc_pr(
        y_true,
        y_proba,
        plot_line_args: dict | None = None,
        suptitle_args: dict | None = None
):
    fig, axs = create_fig(
        nplots=2,
        ncols=2,
        figsize=(10, 4)
    )

    axs_flattened = axs.flatten()

    plot_roc(
        y_true,
        y_proba,
        ax=axs_flattened[0],
        plot_line_args=plot_line_args
    )

    plot_pr(
        y_true,
        y_proba,
        ax=axs_flattened[1],
        plot_line_args=plot_line_args
    )

    fig_suptitle_format(
        fig,
        title='ROC and PR curves',
        **suptitle_args
    )

def plot_roc_pr_old(
    y_true,
    probas: dict,
    target_recall: float | None = None,
    min_precision: float | None = None,
    figsize=(12, 5),
):
    """Plot ROC and Precision-Recall curves side by side for one or more models.

    probas: {model_name: array of P(churn=1)}
    target_recall / min_precision: draw your business goal on the PR plot.
    """
    y_true = np.asarray(y_true)
    prevalence = y_true.mean()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)

    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC-ROC = {roc_auc_score(y_true, p):.3f})")

        prec, rec, _ = precision_recall_curve(y_true, p)
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (AUC-PR = {average_precision_score(y_true, p):.3f})")

    # ROC: random-guess diagonal
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC-ROC = 0.5)")
    ax_roc.set(
        xlabel="False Positive Rate (FPR)",
        ylabel="True Positive Rate (TPR = recall)",
        title="ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    # PR: random-guess baseline is the churn rate
    ax_pr.axhline(prevalence, color="k", ls="--", lw=1, label=f"Random (AUC-PR = {prevalence:.2f})")
    if target_recall is not None:
        ax_pr.axvline(target_recall, color="gray", ls=":", lw=1.5)
    if min_precision is not None:
        ax_pr.axhline(min_precision, color="gray", ls=":", lw=1.5)
    if target_recall is not None and min_precision is not None:
        # The goal is the top-right region beyond both dotted lines.
        ax_pr.fill_between(
            [target_recall, 1], min_precision, 1, color="green", alpha=0.12, label="Goal region"
        )
    ax_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    fig.tight_layout()
    return fig, (ax_roc, ax_pr)