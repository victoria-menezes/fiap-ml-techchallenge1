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
    roc_curve
)
from sklearn.model_selection import (StratifiedKFold,
                                     cross_val_predict,
                                     GridSearchCV)
from sklearn.pipeline import Pipeline

from data_visualizer import plot_line_chart, create_fig, fig_suptitle_format

default_line_color : str = '#004f8f'

def _validate_binary(y_true, y_proba=None):
    """Return y_true (and y_proba) as arrays after checking they're usable for binary metrics."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 labels")
    if np.unique(y_true).size < 2:
        raise ValueError("y_true contains a single class; ROC/PR are undefined")

    if y_proba is None:
        return y_true

    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.shape != y_true.shape:
        raise ValueError(f"y_proba shape {y_proba.shape} != y_true shape {y_true.shape}")
    if not np.isfinite(y_proba).all():
        raise ValueError("y_proba contains NaN or inf")
    return y_true, y_proba

def grid_search_model(model, X_train, y_train, **grid_kwargs):
    args : dict = {
        'cv':5,
        'scoring':'average_precision',
        'n_jobs':-1,
        **(grid_kwargs or {})
    }
    grid_search = GridSearchCV(estimator=model, **args)

    return grid_search.fit(X_train, y_train)

def grid_search_models(
        models : dict,
        models_grid_params : dict,
        X_train, y_train, X_test=None
    ):
    grid_results : dict = {}
    for model_name in models:
        grid_kwargs : dict = models_grid_params.get(model_name, {})
        grid_search = grid_search_model(models[model_name], X_train, y_train, **grid_kwargs)
        grid_results[model_name] = {
            'grid_search':grid_search,
            'y_proba': grid_search.predict_proba(X_test)[:, 1] if X_test is not None else None
        }

    return grid_results

def display_grid_results(
        grid_search_models_result : dict
    ):
    for model in grid_search_models_result:
        grid_search = grid_search_models_result[model]['grid_search']
        print('='*30)
        print(f'RESULTS FOR "{model}"')
        
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Val AUC-PR:", round(grid_search.best_score_, 4))


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
    y = _validate_binary(y)
    if not hasattr(model, 'predict_proba'):
        raise TypeError(f'{type(model).__name__} has no predict_proba')
    
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
    """Highest threshold that still reaches `target_recall`, with the precision you get there.
    
    Returns None if no threshold reaches the target"""
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

## plotters

def plot_roc(
        y_true,
        y_proba,
        ax,
        label:str,
        plot_line_args:dict | None = None
    ):
    y_true, y_proba = _validate_binary(y_true, y_proba)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    args : dict = {
        'title':f'Receiver Operating Characteristic (AUC = {roc_auc:.4f})',
        'xlabel':'False Positive Rates',
        'ylabel':'True Positive Rates',
        'color':'#004f8f',
        **(plot_line_args or {})
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
        label:str,
        plot_line_args:dict | None = None
    ):
    y_true, y_proba = _validate_binary(y_true, y_proba)
    
    precision_rates, recall_rates, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    args : dict = {
        'title':f'Precision-Recall (Average Precision = {ap:.4f})',
        'xlabel':'Recall',
        'ylabel':'Precision',
        'color':'#004f8f',
        **(plot_line_args or {})
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
        label='',
        plot_line_args=plot_line_args
    )

    plot_pr(
        y_true,
        y_proba,
        ax=axs_flattened[1],
        label='',
        plot_line_args=plot_line_args
    )

    fig_suptitle_format(
        fig,
        title='ROC and PR curves',
        **(suptitle_args or {})
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