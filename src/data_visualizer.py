from collections.abc import Hashable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np

### Private methods

# Validators

def _validate_ncols(ncols : int) -> None:
    """Raises ValueError if ncols is below the minimum (1)."""
    MIN_NCOLS = 1
    if ncols < MIN_NCOLS:
        raise ValueError(f'ncols must be at least {MIN_NCOLS}')

def _validate_max_categories(max_categories : int) -> None:
    """Raises ValueError if max_categories is below the minimum (2)."""
    MIN_CATEGORIES = 2
    if max_categories < MIN_CATEGORIES:
        raise ValueError(f'max_categories must be {MIN_CATEGORIES} or higher')

def _validation_batch_plotters(
        data : pd.DataFrame | None = None,
        max_categories : int | None = None,
        ncols : int | None = None
    ) -> None:
    """Validates arguments shared by the plotting functions before
    any figure is created.

    Only the arguments that are not None are checked, so callers can
    validate a subset of them.

    Parameters
    ---
        data : pd.DataFrame | None, default None
            Must have at least one row and one column
        max_categories : int | None, default None
            Must be 2 or higher
        ncols : int | None, default None
            Must be at least 1

    Raises
    ---
        ValueError
            If data is empty, max_categories is less than 2 or ncols is less than 1
    """
    if data is not None and data.empty:
        raise ValueError('DataFrame is empty')
    if max_categories is not None:
        _validate_max_categories(max_categories)
    if ncols is not None:
        _validate_ncols(ncols)

def _validation_batch_groupers(
        data : pd.DataFrame,
        target_col : Hashable
    ) -> None:
    """Validates the arguments specific to the grouped-sum plotting functions.

    Parameters
    ---
        data : pd.DataFrame
            Must have unique column names and at least one column
            besides target_col
        target_col : Hashable
            Must be a column of data with a numeric dtype

    Raises
    ---
        ValueError
            If column names are duplicated, target_col is not in data,
            target_col is not numeric, or there is no column to group by
    """
    if data.columns.has_duplicates:
        raise ValueError('data must not have duplicate column names')
    if target_col not in data.columns:
        raise ValueError(f'target_col {target_col!r} not found in data')
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(f'target_col {target_col!r} must be numeric')
    if len(data.columns) < 2:
        raise ValueError('data needs at least one column besides target_col')

# Util

def _get_number_of_rows(
        amount_of_plots : int,
        ncols : int
    ) -> int:
    """Calculates how many rows are needed to fit amount_of_plots across ncols columns.

    Parameters
    ---
        amount_of_plots : int
            Total number of subplots to draw
        ncols : int
            Number of subplot columns
            Must be at least 1

    Returns
    ---
        int

    Raises
    ---
        ValueError
            If ncols is less than 1
    """
    _validate_ncols(ncols)
    return int(np.ceil(amount_of_plots / ncols))



def _sort_labels_alphabetically(
        value_labels : pd.Index | list | tuple,
        value_data : pd.Series | list | tuple
    ) -> tuple[tuple, tuple]:
    """Sorts labels alphabetically, keeping each value paired with its label.

    Labels are compared by their case-insensitive string representation, so
    labels of mixed types (e.g. strings and ints) can be sorted together.

    Parameters
    ---
        value_labels : pd.Index | list | tuple
        value_data : pd.Series | list | tuple
            Values, in the same order as value_labels

    Returns
    ---
        tuple[tuple, tuple]
            (sorted labels, values reordered to match)
            Two empty tuples if the input is empty
    """
    sorted_pairs : list = sorted(
        zip(value_labels, value_data),
        key = lambda p : str(p[0]).casefold()
    )

    if not sorted_pairs:
        return (), ()

    ordered_value_labels, ordered_value_counts_data = zip(*sorted_pairs)
    return ordered_value_labels, ordered_value_counts_data

def _group_overflow(
        value_counts : pd.Series,
        max_categories : int,
        other_label : str = 'Outros'
    ) -> pd.Series:
    """Keeps the largest categories and sums the rest into a single one.

    Expects value_counts sorted in descending order. If there are more than
    max_categories categories, the top (max_categories - 1) are kept and the
    remainder is summed into other_label, so the result has exactly
    max_categories entries. Works for integer counts and float sums alike.

    Parameters
    ---
        value_counts : pd.Series
            Values indexed by category, sorted in descending order
        max_categories : int
            Maximum number of entries in the result, including other_label
            Must be at least 2
        other_label : str, default 'Outros'
            Label for the aggregated entry. Should not match an existing category

    Returns
    ---
        pd.Series
            value_counts unchanged if len(value_counts) <= max_categories,
            otherwise the grouped values
    
    Raises
    ---
        ValueError
            If max_categories is less than 2 (checked even when no grouping
            is needed)
    """
    _validate_max_categories(max_categories)
    if len(value_counts) <= max_categories:
        return value_counts

    head : pd.Series = value_counts.iloc[:max_categories - 1]
    other_total : float | int = value_counts.iloc[max_categories - 1:].sum()
    return pd.concat([head, pd.Series({other_label: other_total})])

def _group_overflow_and_sort(
        values : pd.Series,
        sort_alphabetically : bool,
        max_categories : int,
        other_label : str
    ) -> tuple[tuple, tuple]:
    """Groups the overflowing categories into other_label and, optionally,
    sorts the result alphabetically.

    Parameters
    ---
        values : pd.Series
            Counts or sums indexed by label, sorted in descending order
        sort_alphabetically : bool
            If True, labels are sorted alphabetically.
            The aggregated entry is always placed last
        max_categories : int
            Maximum number of entries, including the aggregated one
            Must be at least 2
        other_label : str
            Label for the aggregated entry

    Returns
    ---
        tuple[tuple, tuple]
            (labels, values), aligned by position.
            Two empty tuples if values is empty
    Raises
    ---
        ValueError
            If max_categories is less than 2
    """
    values = _group_overflow(values, max_categories, other_label)

    if not sort_alphabetically:
        return tuple(values.index), tuple(values)

    other_value : float | int | None = values.get(other_label)
    visible_categories : pd.Series = values.drop(other_label, errors='ignore')

    labels, amounts = _sort_labels_alphabetically(
        visible_categories.index,
        visible_categories
    )

    # aggregated entry goes last
    if other_value is not None:
        labels = (*labels, other_label)
        amounts = (*amounts, other_value)
    return labels, amounts

def _sort_colors(
        color_dict : dict | None,
        value_labels : list | pd.Index | tuple,
        generic_colors : list
    ) -> list:
    """Returns one color per label, using color_dict where possible.

    Labels found in color_dict get their mapped color. The remaining labels
    take colors from generic_colors in order, wrapping around if there are
    more of them than colors. If color_dict is None, generic_colors is
    returned as is.

    Parameters
    ---
        color_dict : dict | None
            label:color mapping
        value_labels : list | pd.Index | tuple
            Labels to assign colors to, in plotting order
        generic_colors : list
            Fallback color cycle

    Returns
    ---
        list
            Colors aligned with value_labels (or generic_colors if color_dict is None)
    """
    
    if color_dict is None:
        return generic_colors

    ordered_colors : list = []
    generic_color_count : int = 0

    for label in value_labels:
        if label in color_dict:
            ordered_colors.append(color_dict[label])
        else:
            ordered_colors.append(generic_colors[generic_color_count % len(generic_colors)])
            generic_color_count += 1
    return ordered_colors

def _create_fig(
        nplots : int,
        ncols : int,
        figsize : tuple,
        layout : str = 'constrained'
    ) -> tuple[Figure, np.ndarray]:
    """Creates a figure with a grid of subplots large enough for nplots.

    Parameters
    ---
        nplots : int
            Number of subplots that will be drawn
        ncols : int
            Number of subplot columns.
            Must be at least 1
        figsize : tuple
        layout : str, default 'constrained'
            Matplotlib layout engine name

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols), always 2D

    Raises
    ---
        ValueError
            If ncols is less than 1
    """
    nrows : int = _get_number_of_rows(nplots, ncols)

    return plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        squeeze=False,
        layout=layout
    )

def _hide_unused_axes(
        axs : np.ndarray,
        nplots : int
    ) -> None:
    """Hides the subplots after the first nplots (in row-major order).

    Parameters
    ---
        axs : np.ndarray
            2D array of Axes
        nplots : int
            Number of subplots that were actually drawn
    """
    for ax in axs.flatten()[nplots:]:
        ax.set_visible(False)

# Data

def _get_value_counts_and_sort_labels(
        series : pd.Series,
        sort_alphabetically : bool = False,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado',
    ) -> tuple[tuple, tuple]:
    """Computes the labels and counts to plot for a single column.

    Counts the values, drops categories with zero occurrences,
    replaces missing values with na_label, groups the overflowing
    categories into other_label and optionally sorts the
    result alphabetically.

    Parameters
    ---
        series : pd.Series
            Column to count
        sort_alphabetically : bool, default False
            If True, labels are ordered alphabetically instead of by
            frequency. The aggregated entry is always placed last
        max_categories : int, default 5
            Maximum number of entries, including the aggregated one.
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the aggregated entry.
            Should not match an existing category
        dropna : bool, default False
            If False, missing values are kept as their own entry (see na_label)
        na_label : str, default 'Sem dado'
            Label for missing values when dropna is False.
            Should not match an existing category

    Returns
    ---
        tuple[tuple, tuple]
            (labels, counts), aligned by position.
            Two empty tuples if there is nothing to count

    Raises
    ---
        ValueError
            If max_categories is less than 2
    """
    _validate_max_categories(max_categories)

    value_counts_series : pd.Series = series.value_counts(dropna=dropna)
    value_counts_series = value_counts_series[value_counts_series > 0] # ignores unobserved categories

    value_counts_series.index = value_counts_series.index.astype(object).fillna(na_label)

    return _group_overflow_and_sort(
        value_counts_series,
        sort_alphabetically,
        max_categories,
        other_label
    )

def _get_group_by_and_sort_labels(
        data : pd.DataFrame,
        group_by : Hashable,
        target_col : Hashable,
        sort_alphabetically : bool = False,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado',
    ) -> tuple[tuple, tuple]:
    """Sums target_col per category of group_by, then groups overflow and sorts.

    Categories are ordered by descending sum before grouping, so the
    categories merged into other_label are the ones with the smallest sums.
    Ties keep a deterministic order. Unobserved categorical levels are
    ignored. Missing values in target_col are skipped by the sum.

    Parameters
    ---
        data : pd.DataFrame
            Must contain group_by and target_col, each exactly once
        group_by : Hashable
            Name of the column whose categories become the labels
        target_col : Hashable
            Name of the numeric column that is summed per category
        sort_alphabetically : bool, default False
            If True, labels are ordered alphabetically instead of by sum.
            The aggregated entry is always placed last
        max_categories : int, default 5
            Maximum number of entries, including the aggregated one.
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the aggregated entry.
            Should not match an existing category
        dropna : bool, default False
            If False, rows with a missing category are grouped into their
            own entry (see na_label)
        na_label : str, default 'Sem dado'
            Label for the missing category when dropna is False.
            Should not match an existing category

    Returns
    ---
        tuple[tuple, tuple]
            (labels, sums), aligned by position.
            Two empty tuples if there is nothing to sum

    Raises
    ---
        ValueError
            If max_categories is less than 2
    """
    _validate_max_categories(max_categories)

    values_series : pd.Series = data[target_col].groupby(
        data[group_by],
        dropna=dropna,
        observed=True
    ).sum()

    values_series.index = values_series.index.astype(object).fillna(na_label) # replaces NA label with na_label
    values_series = values_series.sort_values(ascending=False, kind='stable')

    return _group_overflow_and_sort(
        values_series,
        sort_alphabetically,
        max_categories,
        other_label
    )

# Draw

def _draw_pie(
        ax : Axes,
        x : tuple | list,
        labels : tuple | list,
        autopct : str,
        colors : list,
        title : str
    ) -> None:
    """Draws a pie chart with a title on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : tuple | list
            Slice values. Must be non-negative
        labels : tuple | list
            Slice labels, aligned with x
        autopct : str
            Format string for the percentage shown on each slice
        colors : list
            Slice colors. Cycled if shorter than x
        title : str
    """
    ax.pie(
        x=x,
        labels=labels,
        autopct=autopct,
        colors=colors
    )
    ax.set_title(title)

def _draw_barh(
        ax : Axes,
        labels : tuple | list,
        width : tuple | list,
        color : list,
        title : str,
        xlabel : str
    ) -> None:
    """Draws a horizontal bar chart on ax, with the first label on top.

    Bars are drawn at integer positions and labeled through the tick
    labels, so labels of any type work and labels that look the same as
    strings (e.g. 1 and '1') stay separate bars.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        labels : tuple | list
            Bar labels, in top-to-bottom order. Converted to strings for display
        width : tuple | list
            Bar lengths, aligned with labels
        color : list
            Bar colors. Cycled if shorter than width
        title : str
        xlabel : str
            Label for the length axis
    """
    positions : range = range(len(labels))
    ax.barh(
        y=positions,  # draws at position first
        width=width,
        color=color
    )
    ax.set_yticks(
        positions,  # labels later
        labels=[str(label) for label in labels],
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)

### Public methods

### VALUE COUNTS

def plot_value_counts_pie(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Distribuição de clientes por ',
        color_dict : dict | None = None,
        generic_colors : list | None = None,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
) -> tuple[Figure, np.ndarray]:
    """Plots each column's value counts as a pie chart, one subplot per column.

    Parameters
    ---
        data : pd.DataFrame
            One pie chart is drawn per column in data. Must not be empty
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Unused subplots are hidden,
            row amount is calculated based on ncols
            Must be at least 1
        sort_alphabetically : bool, default False
            If True, slices are ordered alphabetically by label instead of by
            frequency. The aggregated slice is always placed last
        title_prefix : str, default 'Distribuição de clientes por '
            Text concatenated before the column name in each subplot title
        color_dict : dict | None, default None
            label:color dictionary to color slices according to their label.
            Unmapped labels use generic_colors cycle
        generic_colors : list | none, default None
            If None, the default matplotlib color cycle will be used
        max_categories : int, default 5
            Maximum number of slices per chart, including the aggregated slice.
            Less frequent categories are summed into other_label
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the slice that aggregates the least frequent categories.
            Should not match an existing category
        dropna : bool, default False
            If False, missing values are shown as their own slice (see na_label)
        na_label : str, default 'Sem dado'
            Label for the missing values slice when dropna is False.
            Should not match an existing category

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If data is empty, max_categories is less than 2 or ncols is less than 1
    """
    _validation_batch_plotters(
        data=data,
        max_categories=max_categories,
        ncols=ncols
    )

    if generic_colors is None:
        generic_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs = _create_fig(
        nplots=len(data.columns),
        ncols=ncols,
        figsize=figsize
    )

    # items() yields one Series per column, even if column names are duplicated
    for (col, series), ax in zip(data.items(), axs.flatten()):
        value_labels, value_counts = _get_value_counts_and_sort_labels(
            series,
            sort_alphabetically=sort_alphabetically,
            max_categories=max_categories,
            other_label=other_label,
            dropna=dropna,
            na_label=na_label
        )

        ordered_colors : list = _sort_colors(color_dict, value_labels, generic_colors)

        _draw_pie(
            ax=ax,
            x=value_counts,
            labels=value_labels,
            autopct='%1.1f%%',
            colors=ordered_colors,
            title=f'{title_prefix}{col}'
        )

    _hide_unused_axes(axs, len(data.columns))

    fig.suptitle(fig_title)
    return fig, axs

def plot_value_counts(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Clientes por ',
        xlabel : str = 'Qtd',
        color_dict : dict | None = None,
        generic_color : str = 'blue',
        max_categories : int = 10,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
    ) -> tuple[Figure, np.ndarray]:
    """Plots each column's value counts as a horizontal bar chart, one subplot per column.

    Bars are listed top to bottom: by descending frequency, or alphabetically
    if sort_alphabetically is True. The aggregated bar is always last.

    Parameters
    ---
        data : pd.DataFrame
            One bar chart is drawn per column in data
            Must not be empty
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Unused subplots are hidden,
            row amount is calculated based on ncols
            Must be at least 1
        sort_alphabetically : bool, default False
            If True, bars are ordered alphabetically by label instead of by
            frequency. The aggregated bar is always placed last
        title_prefix : str, default 'Clientes por '
            Text concatenated before the column name in each subplot title
        xlabel : str, default 'Qtd'
            Label for the count axis of every subplot
        color_dict : dict | None, default None
            label:color dictionary to color bars according to their label.
            Unmapped labels use generic_color
        generic_color : str, default 'blue'
            Color for every bar whose label is not in color_dict
        max_categories : int, default 10
            Maximum number of bars per chart, including the aggregated bar.
            Less frequent categories are summed into other_label.
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the bar that aggregates the least frequent categories.
            Should not match an existing category
        dropna : bool, default False
            If False, missing values are shown as their own bar (see na_label)
        na_label : str, default 'Sem dado'
            Label for the missing values bar when dropna is False.
            Should not match an existing category

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If data is empty, max_categories is less than 2 or ncols is less than 1
    """
    _validation_batch_plotters(
        data=data,
        max_categories=max_categories,
        ncols=ncols
    )

    fig, axs = _create_fig(
        nplots=len(data.columns),
        ncols=ncols,
        figsize=figsize
    )

    for (col, series), ax in zip(data.items(), axs.flatten()):
        value_labels, value_counts = _get_value_counts_and_sort_labels(
            series,
            sort_alphabetically=sort_alphabetically,
            max_categories=max_categories,
            other_label=other_label,
            dropna=dropna,
            na_label=na_label
        )

        ordered_colors : list = _sort_colors(color_dict, value_labels, [generic_color])

        _draw_barh(
            ax=ax,
            labels=value_labels,
            width=value_counts,
            color=ordered_colors,
            title=f'{title_prefix}{col}',
            xlabel=xlabel
        )

    _hide_unused_axes(axs, len(data.columns))

    fig.suptitle(fig_title)
    return fig, axs

### GROUPED SUMS
def plot_group_variable_sum_into_pie(
        data : pd.DataFrame,
        target_col : Hashable,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Agrupamento por ',
        color_dict : dict | None = None,
        generic_colors : list | None = None,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
    ) -> tuple[Figure, np.ndarray]:
    """Plots, for each column of data, a pie chart of target_col's sum per category.

    Every column except target_col is used as a grouping variable and gets
    its own subplot. Slice sizes are the sums of target_col within each
    category.

    Parameters
    ---
        data : pd.DataFrame
            Must contain target_col, have unique column names and at least
            one other column. One pie chart is drawn per column other than
            target_col
        target_col : Hashable
            Name of the numeric column that is summed per category.
            Missing values in this column are ignored by the sum.
            Pie slices cannot be negative, so a category whose sum is
            negative makes matplotlib raise a ValueError
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Must be at least 1.
            Unused subplots are hidden, row amount is calculated based on ncols
        sort_alphabetically : bool, default False
            If True, slices are ordered alphabetically by label instead of by
            summed value. The aggregated slice is always placed last
        title_prefix : str, default 'Agrupamento por '
            Text concatenated before the column name in each subplot title
        color_dict : dict | None, default None
            label:color dictionary to color slices according to their label.
            Unmapped labels use generic_colors cycle
        generic_colors : list | none, default None
            If None, the default matplotlib color cycle will be used
        max_categories : int, default 5
            Maximum number of slices per chart, including the aggregated slice.
            Categories with the smallest sums are merged into other_label.
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the slice that aggregates the smallest categories.
            Should not match an existing category
        dropna : bool, default False
            If False, rows with a missing category are grouped into their
            own slice (see na_label)
        na_label : str, default 'Sem dado'
            Label for the missing category slice when dropna is False.
            Should not match an existing category

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If data is empty, max_categories is less than 2, ncols is less
            than 1, column names are duplicated, or target_col is missing,
            non-numeric or the only column
    """
    _validation_batch_plotters(
        data=data,
        max_categories=max_categories,
        ncols=ncols
    )
    _validation_batch_groupers(
        data=data,
        target_col=target_col
    )

    group_cols : list = [col for col in data.columns if col != target_col]

    if generic_colors is None:
            generic_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs = _create_fig(
        nplots=len(group_cols),
        ncols=ncols,
        figsize=figsize
    )

    for col, ax in zip(group_cols, axs.flatten()):
        value_labels, value_sums = _get_group_by_and_sort_labels(
            data[[col, target_col]],
            group_by=col,
            target_col=target_col,
            sort_alphabetically=sort_alphabetically,
            max_categories=max_categories,
            other_label=other_label,
            dropna=dropna,
            na_label=na_label
        )

        ordered_colors : list = _sort_colors(color_dict, value_labels, generic_colors)

        _draw_pie(
            ax=ax,
            x=value_sums,
            labels=value_labels,
            autopct='%1.1f%%',
            colors=ordered_colors,
            title=f'{title_prefix}{col}'
        )

    _hide_unused_axes(axs, len(group_cols))

    fig.suptitle(fig_title)
    return fig, axs

def plot_group_variable_sum_into_barh(
        data : pd.DataFrame,
        target_col : Hashable,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Agrupamento por ',
        xlabel : str = 'Qtd',
        color_dict : dict | None = None,
        generic_color : str = 'blue',
        max_categories : int = 10,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
    ) -> tuple[Figure, np.ndarray]:
    """Plots, for each column of data, a horizontal bar chart of target_col's sum per category.

    Every column except target_col is used as a grouping variable and gets
    its own subplot. Bar lengths are the sums of target_col within each
    category. Bars are listed top to bottom: by descending sum, or
    alphabetically if sort_alphabetically is True. The aggregated bar is
    always last.

    Parameters
    ---
        data : pd.DataFrame
            Must contain target_col, have unique column names and at least
            one other column. One bar chart is drawn per column other than
            target_col
        target_col : Hashable
            Name of the numeric column that is summed per category.
            Missing values in this column are ignored by the sum
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Must be at least 1.
            Unused subplots are hidden, row amount is calculated based on ncols
        sort_alphabetically : bool, default False
            If True, bars are ordered alphabetically by label instead of by
            summed value. The aggregated bar is always placed last
        title_prefix : str, default 'Agrupamento por '
            Text concatenated before the column name in each subplot title
        xlabel : str, default 'Qtd'
            Label for the value axis of every subplot
        color_dict : dict | None, default None
            label:color dictionary to color bars according to their label.
            Unmapped labels use generic_color
        generic_color : str, default 'blue'
            Color for every bar whose label is not in color_dict
        max_categories : int, default 10
            Maximum number of bars per chart, including the aggregated bar.
            Categories with the smallest sums are merged into other_label.
            Must be 2 or higher
        other_label : str, default 'Outros'
            Label for the bar that aggregates the smallest categories.
            Should not match an existing category
        dropna : bool, default False
            If False, rows with a missing category are grouped into their
            own bar (see na_label)
        na_label : str, default 'Sem dado'
            Label for the missing category bar when dropna is False.
            Should not match an existing category

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If data is empty, max_categories is less than 2, ncols is less
            than 1, column names are duplicated, or target_col is missing,
            non-numeric or the only column
    """
    _validation_batch_plotters(
        data=data,
        max_categories=max_categories,
        ncols=ncols
    )
    _validation_batch_groupers(
        data=data,
        target_col=target_col
    )

    group_cols : list = [col for col in data.columns if col != target_col]

    fig, axs = _create_fig(
        nplots=len(group_cols),
        ncols=ncols,
        figsize=figsize
    )

    for col, ax in zip(group_cols, axs.flatten()):
        value_labels, value_sums = _get_group_by_and_sort_labels(
            data[[col, target_col]],
            group_by=col,
            target_col=target_col,
            sort_alphabetically=sort_alphabetically,
            max_categories=max_categories,
            other_label=other_label,
            dropna=dropna,
            na_label=na_label
        )

        ordered_colors : list = _sort_colors(color_dict, value_labels, [generic_color])

        _draw_barh(
            ax=ax,
            labels=value_labels,
            width=value_sums,
            color=ordered_colors,
            title=f'{title_prefix}{col}',
            xlabel=xlabel
        )

    _hide_unused_axes(axs, len(group_cols))

    fig.suptitle(fig_title)
    return fig, axs