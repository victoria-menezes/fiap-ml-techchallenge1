"""Helpers to evaluate binary churn classifiers with ROC and Precision-Recall curves.
 
Conventions
-----------
Labels are 0/1 and 1 means churn. Anything else is rejected up front, because
``predict_proba(...)[:, 1]`` is only P(churn) when the classes are ``[0, 1]``.

Choose thresholds on out-of-fold predictions from the TRAIN set, then apply
that fixed threshold to the test set. Never re-tune it on test.
 
Typical use in a notebook:: 
    from evaluation import get_oof_proba, plot_roc_pr, get_threshold_for_recall
 
    probas = {
        'logreg': get_oof_proba(models['logreg'], X_train, y_train, seed=SEED),
        'rf':     get_oof_proba(models['rf'],     X_train, y_train, seed=SEED),
    }
    plot_roc_pr(y_train, probas, target_recall=0.80, min_precision=0.65)
    get_threshold_for_recall(y_train, probas['logreg'], target_recall=0.80)
"""

from collections.abc import Mapping
from itertools import cycle

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV
)
from sklearn.pipeline import Pipeline

from data_visualizer import (
    plot_line_chart,
    create_fig,
    fig_suptitle_format,
    make_color_zone,
    draw_line
)


_line_color_cycle : list[str] = ['#004f8f']
default_line_color : str = _line_color_cycle[0]

### Validation helpers

def _as_binary_labels(y) -> np.ndarray:
    """Converts y to a 1-D array and checks it holds 0/1 labels of both classes.
 
    Parameters
    ---
        y : array-like
            True labels
 
    Returns
    ---
        np.ndarray
 
    Raises
    ---
        ValueError
            If y is not 1-D, has values other than 0 and 1 (NaN included),
            or contains only one class
    """
    y_new = np.asanyarray(y)
    if y_new.ndim != 1:
        raise ValueError(f'y_true must be 1-D, got shape {y.shape}')
    if not np.isin(y_new, (0, 1)).all():
        raise ValueError('y_true must contain only 0/1 labels')
    if np.unique(y_new).size < 2:
        raise ValueError('y_true must contain both classes; ROC/PR are undefined otherwise')
    return y_new


def _validate_binary(
        y_true,
        y_proba=None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Validates a (labels, scores) pair and returns both as arrays.
 
    Parameters
    ---
        y_true : array-like
            True 0/1 labels
        y_proba : array-like
            Predicted P(churn=1), one value per row of y_true
 
    Returns
    ---
        tuple[np.ndarray, np.ndarray]
            (y_true, y_proba as float)
 
    Raises
    ---
        ValueError
            If y_true is invalid (see _as_binary_labels), y_proba does not have
            the same shape as y_true, or y_proba has NaN/inf
    """
    y_true = _as_binary_labels(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.shape != y_true.shape:
        hint = ' (pass a single column, e.g. predict_proba(X)[:, 1])' if y_proba.ndim == 2 else ''
        raise ValueError(
            f'y_proba shape {y_proba.shape} != y_true shape {y_true.shape}{hint}'
        )
    if not np.isfinite(y_proba).all():
        raise ValueError('y_proba contains NaN or inf')

    return y_true, y_proba

def _check_unit_interval(name: str, value: float, low_open: bool = False) -> None:
    """Raises ValueError unless value is in [0, 1] (or (0, 1] if low_open)."""
    ok = (0 < value if low_open else 0 <= value) and value <= 1
    if not ok:
        interval = '(0, 1]' if low_open else '[0, 1]'
        raise ValueError(f'{name} must be in {interval}, got {value}')

def _make_cv(seed: int | None, n_splits: int = 5) -> StratifiedKFold:
    """Builds the stratified splitter shared by grid search and OOF predictions.
 
    Parameters
    ---
        seed : int | None
            If given, folds are shuffled with this random state.
            If None, folds are not shuffled (deterministic, like cv=5)
        n_splits : int, default 5
 
    Returns
    ---
        StratifiedKFold
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=seed is not None,
        random_state=seed,
    )

### Model selection
 
def grid_search_model(
        model,
        X_train,
        y_train,
        seed : int,
        **grid_kwargs
    ) -> GridSearchCV:
    """Fits a GridSearchCV tuned for imbalanced binary classification.
 
    Defaults: stratified 5-fold CV, ``scoring='average_precision'`` and
    ``n_jobs=-1``. Any ``GridSearchCV`` argument in grid_kwargs overrides them.
 
    Parameters
    ---
        model : estimator
            Usually a Pipeline from build_model_pipeline
        X_train : array-like
        y_train : array-like
            0/1 labels
        seed : int | None, default None
            Random state for the default CV splitter (see _make_cv).
            Ignored if grid_kwargs contains its own ``cv``
        **grid_kwargs
            Passed to GridSearchCV. Must include ``param_grid``
 
    Returns
    ---
        GridSearchCV
            The fitted search
 
    Raises
    ---
        ValueError
            If param_grid is missing or y_train is not valid 0/1 labels
    """
    if 'param_grid' not in grid_kwargs:
        raise ValueError('grid_kwargs must include "param_grid"')
    _as_binary_labels(y_train)

    args : dict = {
        'cv':_make_cv(seed),
        'scoring':'average_precision',
        'n_jobs':-1,
        **(grid_kwargs or {})
    }
    grid_search = GridSearchCV(estimator=model, **args)

    return grid_search.fit(X_train, y_train)

def grid_search_models(
        models : dict,
        models_grid_params : dict,
        seed : int,
        X_train,
        y_train,
        X_test=None
    ) -> dict:
    """Runs grid_search_model for every model.
 
    Parameters
    ---
        models : dict
            name -> estimator
        models_grid_params : dict
            name -> dict of GridSearchCV kwargs (must include ``param_grid``).
            Must have an entry for every model and no entries for unknown names
        X_train : array-like
        y_train : array-like
        X_test : array-like | None, default None
            If given, the refit best estimator scores it and P(churn=1)
            is returned under 'y_proba'
        seed : int | None, default None
            Forwarded to grid_search_model
 
    Returns
    ---
        dict
            name -> {'grid_search': GridSearchCV, 'y_proba': np.ndarray | None}
 
    Raises
    ---
        ValueError
            If models and models_grid_params names do not match, or X_test is
            given but refit was disabled so there is no best estimator
    """
    unknown = set(models_grid_params) - set(models)
    missing = set(models) - set(models_grid_params)
    if unknown or missing:
        raise ValueError(
            f'models and models_grid_params differ. '
            f'Grid params without model: {sorted(unknown)}; '
            f'models without grid params: {sorted(missing)}'
        )

    results : dict = {}
    for name, model in models.items():
        grid_search = grid_search_model(
            model,
            X_train,
            y_train,
            seed=seed,
            **models_grid_params[name]
        )

        y_proba = None
        if X_test is not None:
            if not grid_search.refit:
                raise ValueError(f'"{name}": refit is disabled, cannot score X_test')

            y_proba = grid_search.predict_proba(X_test)[:, 1]
        results[name] = {
            'grid_search': grid_search,
            'y_proba': y_proba
        }
    return results

def display_grid_results(
        grid_search_models_result : dict
    ) -> None:
    """Prints the best parameters and best CV score of each model.
 
    Parameters
    ---
        grid_search_models_result : dict
            Output of grid_search_models
    """
    # for model in grid_search_models_result:
    for name, result in grid_search_models_result.items():
        grid_search = result['grid_search']
        
        scoring = grid_search.scoring if isinstance(grid_search.scoring, str) else 'score'

        print('='*30)
        print(f'RESULTS FOR "{name}"')
        
        print("Best Parameters:", grid_search.best_params_)
        print(f"Best CV {scoring}: {grid_search.best_score_:.4f}")

### Predictions and thresholds

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
 
    Parameters
    ---
        model : Pipeline
            Any estimator with predict_proba. It is cloned, not fitted in place
        X : array-like | pd.DataFrame
        y : array-like
            0/1 labels
        seed : int
            Random state for the shuffled stratified folds
        n_splits : int, default 5
 
    Returns
    ---
        np.ndarray
            Shape (len(y),), P(churn=1) for each row
 
    Raises
    ---
        TypeError
            If model has no predict_proba
        ValueError
            If y is not valid 0/1 labels
    """
    y = _as_binary_labels(y) # guarantees values are 0 or 1

    if not hasattr(model, 'predict_proba'):
        raise TypeError(f'{type(model).__name__} has no predict_proba')

    return cross_val_predict(
            model,
            X,
            y,
            cv=_make_cv(seed, n_splits),
            method='predict_proba'
        )[:, 1]

def get_threshold_for_recall(
        y_true,
        y_proba,
        target_recall : float    
    ) -> dict | None:
    """Highest threshold that still reaches `target_recall`, with the precision you get there.
 
    Parameters
    ---
        y_true : array-like
            True 0/1 labels
        y_proba : array-like
            Predicted P(churn=1)
        target_recall : float
            Minimum recall to keep. Must be in (0, 1]
 
    Returns
    ---
        dict | None
            {'threshold', 'recall', 'precision'} (all floats), or None if no
            threshold reaches target_recall
 
    Raises
    ---
        ValueError
            If target_recall is outside (0, 1] or the inputs are invalid
    """
    _check_unit_interval('target_recall', target_recall, low_open=True)
    y_true, y_proba = _validate_binary(y_true, y_proba)

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

### Plotters

def set_color_cycle(colors : list[str]):
    _line_color_cycle[:] = colors
    global default_line_color
    default_line_color = _line_color_cycle[0]

def _color_cycle() -> cycle:
    return cycle(_line_color_cycle,)

def plot_roc(
        y_true,
        y_proba,
        ax,
        label:str,
        plot_line_kwargs:dict | None = None
    ) -> float:    
    """Draws one ROC curve on ax. The AUC is added to the legend entry.
 
    Parameters
    ---
        y_true : array-like
            True 0/1 labels
        y_proba : array-like
            Predicted P(churn=1)
        ax : matplotlib.axes.Axes
        label : str
            Legend name of the curve; the AUC is appended to it
        plot_line_kwargs : dict | None, default None
            Overrides for plot_line_chart (color, title, ...)
 
    Returns
    ---
        float
            ROC AUC
    """
    y_true, y_proba = _validate_binary(y_true, y_proba)
    false_positive_rates, true_positive_rates, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    args : dict = {
        'title':f'Receiver Operating Characteristic',
        'xlabel':'False Positive Rates',
        'ylabel':'True Positive Rates',
        'color':default_line_color,
        'label':f'{label} (AUC = {roc_auc:.4f})',
        **(plot_line_kwargs or {})
    }

    plot_line_chart(
        ax=ax,
        x=false_positive_rates,
        y=true_positive_rates,
        **args
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    return float(roc_auc)

def plot_pr(
        y_true,
        y_proba,
        ax,
        label:str,
        color_zone_kwargs:dict | None = None,
        plot_line_kwargs:dict | None = None
    ):
    """Draws one Precision-Recall curve on ax. Average precision is added to the legend entry.
 
    Average precision (a step-wise sum) is used instead of trapezoidal AUC,
    which is optimistic on PR curves.
 
    Parameters
    ---
        y_true : array-like
            True 0/1 labels
        y_proba : array-like
            Predicted P(churn=1)
        ax : matplotlib.axes.Axes
        label : str
            Legend name of the curve; the AP is appended to it
        plot_line_kwargs : dict | None, default None
            Overrides for plot_line_chart (color, title, ...)
 
    Returns
    ---
        float
            Average precision
    """
    y_true, y_proba = _validate_binary(y_true, y_proba)
    
    precision_rates, recall_rates, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    kwargs : dict = {
        'title':f'Precision-Recall (Average Precision = {ap:.4f})',
        'xlabel':'Recall',
        'ylabel':'Precision',
        'color':default_line_color,
        'label':f'{label} (AP = {ap:.4f})',
        **(plot_line_kwargs or {})
    }

    plot_line_chart(
        ax=ax,
        x=recall_rates,
        y=precision_rates,
        **kwargs
    )

    if color_zone_kwargs:
        color_zone_kwargs_all : dict = {
            'ax':ax,
            **(color_zone_kwargs or {})
        }

        make_color_zone(
            **color_zone_kwargs_all
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    return ap

def plot_roc_pr(
        ax_roc : Axes,
        ax_pr : Axes,
        y_true,
        y_proba,
        label:str,
        target_recall : float | None = None,
        min_precision : float = 0.5,
        plot_line_kwargs: dict | None = None
):
    plot_roc(
        y_true,
        y_proba,
        ax=ax_roc,
        label=label,
        plot_line_kwargs=plot_line_kwargs
    )

    # ROC: random-guess baseline is 0.5
    prevalence = y_true.mean()
    ax_roc.plot([0, 1], [0, 1], "-", lw=0.5, color='red', label="Random (AUC-ROC = 0.5)")


    color_zone_kwargs_all : dict = {}
    if target_recall:
        color_zone_kwargs_all = {
            'ax':ax_pr,
            'x':[target_recall, 1],
            'y1':min_precision,
            'y2':1,
            'color':'#14c50e',
            'fill_between_kwargs':{'alpha': 0.2}
        }
    plot_pr(
        y_true,
        y_proba,
        ax=ax_pr,
        label=label,
        plot_line_kwargs=plot_line_kwargs,
        color_zone_kwargs=color_zone_kwargs_all
    )

    # PR: random-guess baseline is the churn rate
    prevalence = y_true.mean()
    draw_line(
        ax=ax_pr,
        coord=prevalence,
        color='red',
        label=f'Random (AUC-PR = {prevalence:.2f}'
    )

def plot_multiple_roc_pr(
        y_true,
        results: Mapping[str, np.ndarray] | np.ndarray,
        target_recall: float | None = None,
        min_precision: float = 0.5,
        plot_line_kwargs: dict | None = None,
        suptitle_kwargs: dict | None = None,
        figsize: tuple = (10, 4)
    ) -> tuple[Figure, np.ndarray]:
    """Plots ROC and Precision-Recall curves side by side, one curve per model.
 
    Both panels include the random-guess baseline (ROC diagonal; PR line at the
    churn rate). If target_recall is given, the PR panel shades the region
    where recall >= target_recall and precision >= min_precision.
 
    Parameters
    ---
        y_true : array-like
            True 0/1 labels
        y_proba : Mapping[str, array-like] | array-like
            name -> P(churn=1) for each model, e.g. from get_oof_proba.
            A bare array is plotted as a single model called 'model'
        target_recall : float | None, default None
            Left edge of the shaded target zone. Must be in (0, 1]
        min_precision : float, default 0.5
            Bottom edge of the target zone. Must be in [0, 1].
            Only used if target_recall is given
        plot_line_kwargs : dict | None, default None
            Overrides for every curve (a 'color' here applies to all models)
        suptitle_kwargs : dict | None, default None
            Overrides for fig_suptitle_format
        figsize : tuple, default (10, 4)
 
    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (1, 2): [[roc, pr]]
 
    Raises
    ---
        ValueError
            If probas is empty, target_recall or min_precision are out of
            range, or any input is invalid (see _validate_binary)
    """
    if target_recall is not None:
        _check_unit_interval('target_recall', target_recall, low_open=True)
        _check_unit_interval('min_precision', min_precision)
    y_true = _as_binary_labels(y_true)
    if not isinstance(results, Mapping):
        results = {'model': results}
    if not results:
        raise ValueError('probas is empty')
 
    fig, axs = create_fig(nplots=2, ncols=2, figsize=figsize)
    ax_roc, ax_pr = axs.ravel()
 
    # baselines and target zone first, so they are part of the final legends.
    prevalence = float(y_true.mean())
    ax_roc.plot([0, 1], [0, 1], lw=0.5, color='red', label='Random (AUC = 0.5)')
    draw_line(ax_pr, coord=prevalence, color='red', label=f'Random (AP = {prevalence:.2f})')
 
    colors = _color_cycle()
    for name, results in results.items():
        y_proba = results['y_proba']
        kwargs = {'color': next(colors), **(plot_line_kwargs or {})}
        plot_roc(
            y_true=y_true,
            y_proba=y_proba,
            ax=ax_roc,
            label=name,
            plot_line_kwargs=kwargs
        )
        color_zone_kwargs : dict = {}
        if target_recall is not None:
            color_zone_kwargs={
                'x':[target_recall or 1, 1],
                'y1':min_precision,
                'y2':1,
                'color':'#14c50e',
                'fill_between_kwargs':{
                    'alpha': 0.2,
                    'label': f'Target (recall \u2265 {target_recall:.2f}, precision \u2265 {min_precision:.2f})',
                    }
                }
        plot_pr(
            y_true=y_true,
            y_proba=y_proba,
            ax=ax_pr,
            label=name,
            color_zone_kwargs=color_zone_kwargs,
            plot_line_kwargs=kwargs
        )
 
    # plot_line_chart rebuilds the legend on every call; rebuild once at the end
    # so the baselines and the target zone are included.
    for ax in (ax_roc, ax_pr):
        ax.legend(loc='lower right')
 
    fig_suptitle_format(fig, title='ROC and PR curves', **(suptitle_kwargs or {}))
    return fig, axs