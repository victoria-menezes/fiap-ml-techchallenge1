import inspect
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np

defaults_ax_title : dict = {
    'fontsize' : 10,
    'fontweight' : 'bold',
    'fontstyle' : 'normal',
    'fontfamily' : 'sans-serif',
    'color' : '#000000'
}

defaults_fig_title : dict = {
    'fontsize' : 14,
    'fontweight' : 'bold',
    'fontstyle' : 'normal',
    'fontfamily' : 'sans-serif',
    'color' : '#000000'
}

# matplotlib 3.9 renamed Axes.boxplot's 'labels' argument to 'tick_labels'
_BOXPLOT_LABEL_KEY : str = (
    'tick_labels'
    if 'tick_labels' in inspect.signature(Axes.boxplot).parameters
    else 'labels'
)

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

def _validation_batch_comparers(
        data1 : pd.DataFrame,
        data2 : pd.DataFrame,
        max_categories : int | None = None,
        ncols : int | None = None
    ) -> list:
    """Validates the arguments of the two-sample plotting functions and
    finds the columns they will compare.

    Parameters
    ---
        data1 : pd.DataFrame
            Must not be empty. Column names must be unique
        data2 : pd.DataFrame
            Must not be empty. Column names must be unique
        max_categories : int | None, default None
            Checked like in _validation_batch_plotters (2 or higher)
        ncols : int | None, default None
            Checked like in _validation_batch_plotters (at least 1)

    Returns
    ---
        list
            Column names present in both DataFrames, in data1's order.
            Never empty

    Raises
    ---
        ValueError
            If data1 or data2 is empty, max_categories is less than 2,
            ncols is less than 1, either DataFrame has duplicate column
            names, or they share no column
    """
    _validation_batch_plotters(
        data=data1,
        max_categories=max_categories,
        ncols=ncols
    )
    _validation_batch_plotters(
        data=data2
    )
    return _get_common_columns(data1, data2)

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
        other_label : str,
        na_label : str
    ) -> tuple[tuple, tuple]:
    """Prepares per-category values for plotting.

    Replaces a missing category label with na_label, orders the categories
    by descending value, groups the overflowing ones into other_label and,
    optionally, sorts the result alphabetically. Ties keep their original
    order.

    Parameters
    ---
        values : pd.Series
            Counts or sums indexed by label, in any order
        sort_alphabetically : bool
            If True, labels are sorted alphabetically.
            The aggregated entry is always placed last
        max_categories : int
            Maximum number of entries, including the aggregated one
            Must be at least 2
        other_label : str
            Label for the aggregated entry
        na_label : str
            Label that replaces a missing category

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
    values = values.set_axis(values.index.astype(object).fillna(na_label))
    values = values.sort_values(ascending=False, kind='stable')
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

def create_fig(
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
            Figure size in inches, as (width, height)
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

def _get_common_columns(
        data1 : pd.DataFrame,
        data2 : pd.DataFrame
    ) -> list:
    """Returns the column names present in both DataFrames, in data1's order.

    Used by the functions that compare two samples, so that each column of
    data1 is always matched with the column of data2 with the same name.

    Parameters
    ---
        data1 : pd.DataFrame
            Must have unique column names
        data2 : pd.DataFrame
            Must have unique column names

    Returns
    ---
        list
            Shared column names, ordered as in data1. Never empty

    Raises
    ---
        ValueError
            If either DataFrame has duplicate column names, or if they
            have no column in common
    """
    if data1.columns.has_duplicates or data2.columns.has_duplicates:
        raise ValueError('data1 and data2 must not have duplicate column names')

    common_cols : list = [col for col in data1.columns if col in data2.columns]
    if not common_cols:
        raise ValueError('data1 and data2 have no columns in common')
    return common_cols

def _titled_series(
        data : pd.DataFrame,
        title_prefix : str
    ) -> list[tuple[pd.Series, str]]:
    """Pairs each column of data with the title of its subplot.

    Parameters
    ---
        data : pd.DataFrame
        title_prefix : str
            Text concatenated before the column name in each title

    Returns
    ---
        list[tuple[pd.Series, str]]
            One (column, title) pair per column of data, in order.
            Columns are taken by position, so duplicated column names
            still yield one Series each
    """
    return [(series, f'{title_prefix}{col}') for col, series in data.items()]

def _paired_series(
        common_cols : list,
        data1 : pd.DataFrame,
        data2 : pd.DataFrame,
        data1_label : str,
        data2_label : str,
        title_prefix : str
    ) -> list[tuple[pd.Series, str]]:
    """Alternates the two samples for each shared column, with subplot titles.

    Parameters
    ---
        common_cols : list
            Columns present in both DataFrames (see _get_common_columns)
        data1 : pd.DataFrame
        data2 : pd.DataFrame
        data1_label : str
            Name of the first sample, appended to its titles
        data2_label : str
            Name of the second sample, appended to its titles
        title_prefix : str
            Text concatenated before the column name in each title

    Returns
    ---
        list[tuple[pd.Series, str]]
            (column, title) pairs: for each column of common_cols, the
            data1 column followed by the data2 column. Titles follow the
            format '{title_prefix}{column} - {label}'
    """
    return [
        (data[col], f'{title_prefix}{col} - {label}')
        for col in common_cols
        for data, label in ((data1, data1_label), (data2, data2_label))
    ]

def _value_counts_panels(
        titled_series : list[tuple[pd.Series, str]],
        sort_alphabetically : bool = False,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
    ) -> list[tuple[tuple, tuple, str]]:
    """Computes the (labels, counts, title) panel of each series.

    Parameters
    ---
        titled_series : list[tuple[pd.Series, str]]
            (series, title) pairs, as built by _titled_series
        sort_alphabetically, max_categories, other_label, dropna, na_label
            See _get_value_counts_and_sort_labels

    Returns
    ---
        list[tuple[tuple, tuple, str]]
            One (labels, counts, title) panel per series, in order

    Raises
    ---
        ValueError
            If max_categories is less than 2
    """
    return [
        (
            *_get_value_counts_and_sort_labels(
                series,
                sort_alphabetically=sort_alphabetically,
                max_categories=max_categories,
                other_label=other_label,
                dropna=dropna,
                na_label=na_label
            ),
            title
        )
        for series, title in titled_series
    ]

def _group_sum_panels(
        data : pd.DataFrame,
        target_col : Hashable,
        title_prefix : str,
        sort_alphabetically : bool = False,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
    ) -> list[tuple[tuple, tuple, str]]:
    """Computes the (labels, sums, title) panel of every column except target_col.

    Parameters
    ---
        data : pd.DataFrame
            Must contain target_col (see _validation_batch_groupers)
        target_col : Hashable
            Numeric column that is summed per category of each other column
        title_prefix : str
            Text concatenated before the column name in each title
        sort_alphabetically, max_categories, other_label, dropna, na_label
            See _get_group_by_and_sort_labels

    Returns
    ---
        list[tuple[tuple, tuple, str]]
            One (labels, sums, title) panel per grouping column, in order

    Raises
    ---
        ValueError
            If max_categories is less than 2
    """
    group_cols : list = [col for col in data.columns if col != target_col]

    return [
        (
            *_get_group_by_and_sort_labels(
                data[[col, target_col]],
                group_by=col,
                target_col=target_col,
                sort_alphabetically=sort_alphabetically,
                max_categories=max_categories,
                other_label=other_label,
                dropna=dropna,
                na_label=na_label
            ),
            f'{title_prefix}{col}'
        )
        for col in group_cols
    ]

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
    value_counts_series : pd.Series = series.value_counts(dropna=dropna)
    value_counts_series = value_counts_series[value_counts_series > 0] # ignores unobserved categories

    return _group_overflow_and_sort(
        value_counts_series,
        sort_alphabetically,
        max_categories,
        other_label,
        na_label
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
    values_series : pd.Series = data[target_col].groupby(
        data[group_by],
        dropna=dropna,
        observed=True
    ).sum()

    return _group_overflow_and_sort(
        values_series,
        sort_alphabetically,
        max_categories,
        other_label,
        na_label
    )

# Draw

def _title_format(
    ax : Axes,
    title : str,
    title_kwargs : dict | None = None
) -> None:
    """Sets the title of ax, styled with defaults_ax_title.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        title : str
            Title text
        title_kwargs : dict | None, default None
            Style overrides (fontsize, color, ...) merged over
            defaults_ax_title and forwarded to Axes.set_title
    """

    final_kwargs : dict = {
        'label':title,
        **defaults_ax_title,
        **(title_kwargs or {})
    }

    ax.set_title(**final_kwargs)

def fig_suptitle_format(
    fig : Figure,
    title : str,
    suptitle_kwargs : dict | None = None
) -> None:
    """Sets the figure's suptitle, styled with defaults_fig_title.

    Parameters
    ---
        fig : matplotlib.figure.Figure
        title : str
            Title text
        suptitle_kwargs : dict | None, default None
            Style overrides (fontsize, color, ...) merged over
            defaults_fig_title and forwarded to Figure.suptitle
    """

    final_kwargs : dict = {
        't':title,
        **defaults_fig_title,
        **(suptitle_kwargs or {})
    }

    fig.suptitle(**final_kwargs)

def _visual_setup(
    ax : Axes,
    kwargs_major : dict | None = None,
    kwargs_minor : dict | None = None,
    facecolor="#F0F0F0"
) -> None:
    """Applies the shared visual style to ax.

    Sets the background color, draws the grid behind the data (a stronger
    major grid and a fainter minor one, with minor ticks enabled) and hides
    the top and right spines. Called by every _draw_* function before it
    draws anything.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        kwargs_major : dict | None, default None
            Overrides for the major grid line style. Defaults are
            color '#000000', alpha 0.20 and linestyle '-'
            Forwarded to Axes.grid
        kwargs_minor : dict | None, default None
            Overrides for the minor grid line style. Defaults are
            color '#000000', alpha 0.05 and linestyle '-'
            Forwarded to Axes.grid
        facecolor : str, default '#F0F0F0'
            Background color of the plotting area
    """
    grid_major_kwargs : dict = {
        'color':'#000000',
        'alpha':0.20,
        'linestyle':'-',
        **(kwargs_major or {})
    }

    grid_minor_kwargs : dict = {
        'color':'#000000',
        'alpha':0.05,
        'linestyle':'-',
        **(kwargs_minor or {})
    }
    
    ax.set_facecolor(facecolor)
    
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(
        True,
        which='major',
        **grid_major_kwargs
    )
    ax.grid(
        which='minor',
        **grid_minor_kwargs
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def _draw_pie(
        ax : Axes,
        x : tuple | list,
        labels : tuple | list,
        title : str,
        colors : list[str],
        pie_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
) -> None:
    """Draws a pie chart with a title on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : tuple | list
            Slice values. Must be non-negative
        labels : tuple | list
            Slice labels, aligned with x
        title : str
        colors : list[str]
            Slice colors. Cycled if shorter than x
        pie_kwargs : dict | None, default None
            Overrides for the Axes.pie arguments. The defaults are
            x=x, labels=labels, colors=colors and autopct='%1.1f%%'
            (percentage shown on each slice)
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
    """
    _visual_setup(ax, **(setup_kwargs or {}))

    pie_kwargs_all : dict = {
        'x':x,
        'labels':labels,
        'colors':colors,
        'autopct':'%1.1f%%',
        **(pie_kwargs or {})
    }

    ax.pie(**pie_kwargs_all)

    _title_format(ax, title, title_kwargs)


def _draw_barh(
        ax : Axes,
        labels : tuple | list,
        width : tuple | list,
        title : str,
        xlabel : str,
        colors : list[str],
        barh_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
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
        title : str
        xlabel : str
            Label for the length axis
        colors : list[str]
            Bar colors. Cycled if shorter than bar amount (len(width))
        barh_kwargs : dict | None, default None
            Overrides for the Axes.barh arguments. The defaults are
            y=range(len(labels)), width=width and color=colors.
            Overriding y desyncs the bars from the tick labels
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
    """
    _visual_setup(ax, **(setup_kwargs or {}))

    positions : range = range(len(labels))
    barh_kwargs_all : dict = {
        'y':positions,  # draws at position first
        'width':width,
        'color':colors,
        **(barh_kwargs or {})
    }

    ax.barh(**barh_kwargs_all)
    ax.set_yticks(
        positions,  # labels later
        labels=[str(label) for label in labels],
    )
    ax.invert_yaxis()

    _title_format(ax, title, title_kwargs)

    ax.set_xlabel(xlabel)

def _draw_hist(
        ax : Axes,
        x : pd.Series | pd.DataFrame,
        color : str,
        title : str,
        hist_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
) -> None:
    """Draws a histogram with a title on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : pd.Series | pd.DataFrame
            Numeric data to bin
        color : str
            Bar color
        title : str
        hist_kwargs : dict | None, default None
            Overrides for the Axes.hist arguments (e.g. bins, alpha).
            The defaults are x=x and color=color
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
    """
    _visual_setup(ax, **(setup_kwargs or {}))
    hist_kwargs_all : dict = {
        'x': x,
        'color': color,
        **(hist_kwargs or {})
    }

    ax.hist(**hist_kwargs_all)
    _title_format(ax, title, title_kwargs)

def _draw_boxplot(
        ax : Axes,
        x : pd.Series,
        label : str,
        position : float,
        title : str,
        boxplot_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
) -> None:
    """Draws a single box on ax at the given position, with a title.

    Calling it more than once on the same ax puts several boxes side by
    side (use different positions). The title is replaced on every call,
    so the last call's title is the one that remains.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : pd.Series
            Numeric data summarized by the box
        label : str
            Tick label shown under the box
        position : float
            Horizontal position of the box
        title : str
        boxplot_kwargs : dict | None, default None
            Overrides for the Axes.boxplot arguments. The defaults are
            x=x, positions=[position] and the tick label set to [label]
            (passed as tick_labels, or as labels on matplotlib older than 3.9)
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
    """
    _visual_setup(ax, **(setup_kwargs or {}))
    boxplot_kwargs_all : dict = {
        'x':x,
        _BOXPLOT_LABEL_KEY:[label],
        'positions':[position],
        **(boxplot_kwargs or {})
    }
    ax.boxplot(**boxplot_kwargs_all)

    _title_format(ax, title, title_kwargs)

def _draw_legend(
        ax : Axes,
        legend_kwargs : dict | None = None
    ) -> None:
    """Draws a legend on ax, in the lower right corner by default.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments. The default is
            loc='lower right'
    """
    legend_kwargs_all : dict = {
        'loc':'lower right',
        **(legend_kwargs or {})
    }
    ax.legend(**legend_kwargs_all)

def _annotate_axes(
        ax : Axes,
        title : str,
        xlabel : str,
        ylabel : str,
        label : str | None = None,
        title_kwargs : dict | None = None,
        legend_kwargs : dict | None = None
    ) -> None:
    """Sets the title, legend and axis labels of an x-y plot on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        title : str
        xlabel : str
        ylabel : str
        label : str | None, default None
            The legend is only drawn if this is not None
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments (see _draw_legend)
    """
    _title_format(ax, title, title_kwargs)

    if label is not None:
        _draw_legend(ax, legend_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def _draw_lineplot(
        ax : Axes,
        x : pd.Series | np.ndarray,
        y : pd.Series | np.ndarray,
        xlabel : str,
        ylabel : str,
        title : str,
        color : str,
        label : str | None = None,
        line_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None,
        legend_kwargs : dict | None = None
    ) -> None:
    """Draws a line with a title, axis labels and an optional legend on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : pd.Series | np.ndarray
            Horizontal coordinates
        y : pd.Series | np.ndarray
            Vertical coordinates, aligned with x
        xlabel : str
        ylabel : str
        title : str
        color : str
            Line color
        label : str | None, default None
            Legend entry for the line. The legend is only drawn if this
            is not None
        line_kwargs : dict | None, default None
            Overrides for the Axes.plot arguments. The defaults are
            color=color, label=label, lw=0.75 and alpha=1
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments. The default is
            loc='lower right'. Only used if label is not None
    """
    _visual_setup(ax, **(setup_kwargs or {}))
    line_kwargs_all : dict = {
        'color':color,
        'label':label,
        'lw': 0.75,
        'alpha':1,
        **(line_kwargs or {})
    }

    ax.plot(
        x,
        y,
        **line_kwargs_all
    )

    _annotate_axes(ax, title, xlabel, ylabel, label, title_kwargs, legend_kwargs)

def draw_scatter(
        ax : Axes,
        x,
        y,
        xlabel : str,
        ylabel : str,
        title : str,
        color : str,
        label : str | None = None,
        scatter_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        setup_kwargs : dict | None = None,
        legend_kwargs : dict | None = None
    ) -> None:
    """Draws a scatter plot with a title, axis labels and an optional legend on ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : array-like
            Horizontal coordinates
        y : array-like
            Vertical coordinates, aligned with x
        xlabel : str
        ylabel : str
        title : str
        color : str
            Marker color. To color the markers by value instead, pass
            scatter_kwargs={'c': values, 'color': None}, since matplotlib
            rejects c and color together
        label : str | None, default None
            Legend entry for the markers. The legend is only drawn if this
            is not None
        scatter_kwargs : dict | None, default None
            Overrides for the Axes.scatter arguments (e.g. s, alpha, cmap).
            The defaults are x=x, y=y, color=color and label=label
        title_kwargs : dict | None, default None
            Style overrides for the title (see _title_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments. The default is
            loc='lower right'. Only used if label is not None
    """
    _visual_setup(ax, **(setup_kwargs or {}))
    scatter_kwargs_all : dict = {
        'x':x,
        'y':y,
        'color':color,
        'label':label,
        **(scatter_kwargs or {})
    }
    ax.scatter(**scatter_kwargs_all)

    _annotate_axes(ax, title, xlabel, ylabel, label, title_kwargs, legend_kwargs)

def draw_line(
        ax : Axes,
        coord : float,
        linetype : str = 'h',
        color = '#3232',
        ls : str = '-',
        lw : float = 0.5,
        label : str | None = None,
        show_label : bool = False,
        axline_kwargs : dict | None = None,
        legend_kwargs : dict | None = None
    ):
    """Draws a horizontal or vertical reference line across ax.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        coord : float
            Position of the line: the y value for a horizontal line or
            the x value for a vertical one
        linetype : str, default 'h'
            'h' for a horizontal line, 'v' for a vertical one.
            Any other value draws nothing
        color : str, default '#3232'
            Line color. The default is a semi-transparent color in
            matplotlib's 4-digit RGBA hex format
        ls : str, default '-'
            Line style
        lw : float, default 0.5
            Line width
        label : str | None, default None
            Legend entry for the line. The line is created with an empty
            label if None
        show_label : bool, default False
            If True and label is not None, a legend is drawn on ax.
            The legend includes every labeled artist on ax, not only this line
        axline_kwargs : dict | None, default None
            Overrides for the Axes.axhline / Axes.axvline arguments
            (y or x, color, ls, lw and label by default)
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments. The default is
            loc='lower right'. Only used if a legend is drawn
    """

    axline_kwargs_all : dict = {
        'color':color,
        'ls':ls,
        'lw':lw,
        'label':(label or ''),
        **(axline_kwargs or {})
    }

    match linetype:
        case 'h':
            ax.axhline(**{'y':coord, **axline_kwargs_all})
        case 'v':
            ax.axvline(**{'x':coord, **axline_kwargs_all})

    if show_label and label is not None:
        _draw_legend(ax, legend_kwargs)

# Grid

def _plot_grid(
        items : Iterable,
        draw : Callable[[Axes, Any], None],
        ncols : int,
        figsize : tuple,
        fig_title : str = '',
        suptitle_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Creates a grid with one subplot per item and draws each item on its own Axes.

    Shared skeleton of the plotting functions: it creates the figure, calls
    draw once per item (in order, filling the grid row by row), hides the
    leftover subplots and sets the figure title. Callers validate their
    arguments before calling it, so no figure is created for invalid input.

    Parameters
    ---
        items : Iterable
            One entry per subplot. Consumed once, and must not be empty
        draw : Callable[[Axes, Any], None]
            Called as draw(ax, item) to draw one item on its subplot.
            It usually is a closure over the caller's arguments
        ncols : int
            Number of subplot columns.
            Must be at least 1
        figsize : tuple
            Figure size in inches, as (width, height)
        fig_title : str, default ''
            Title for the whole figure
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If ncols is less than 1
    """
    items = list(items)

    fig, axs = create_fig(
        nplots=len(items),
        ncols=ncols,
        figsize=figsize
    )

    for ax, item in zip(axs.flatten(), items):
        draw(ax, item)

    _hide_unused_axes(axs, len(items))

    fig_suptitle_format(fig, fig_title, suptitle_kwargs)
    return fig, axs

def _plot_category_grid(
        panels : list[tuple[tuple, tuple, str]],
        chart : str,
        ncols : int,
        figsize : tuple,
        fig_title : str = '',
        color_dict : dict | None = None,
        generic_colors : list | None = None,
        xlabel : str = '',
        chart_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots one category chart (pie or horizontal bars) per panel, on a grid.

    Shared by every plotting function that draws labeled values, so pie
    and bar variants differ only in the chart argument.

    Parameters
    ---
        panels : list[tuple[tuple, tuple, str]]
            One (labels, values, title) entry per subplot, as built by
            _value_counts_panels or _group_sum_panels. Must not be empty
        chart : str
            'pie' or 'barh'
        ncols : int
            Number of subplot columns.
            Must be at least 1
        figsize : tuple
            Figure size in inches, as (width, height)
        fig_title : str, default ''
            Title for the whole figure
        color_dict : dict | None, default None
            label:color dictionary. Unmapped labels use generic_colors cycle
        generic_colors : list | None, default None
            Fallback color cycle. If None, the default matplotlib color
            cycle will be used. Pass [color] for a single color
        xlabel : str, default ''
            Label for the length axis. Only used by 'barh'
        chart_kwargs : dict | None, default None
            Overrides for the Axes.pie or Axes.barh arguments, depending on chart
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols).
            Unused subplots are hidden

    Raises
    ---
        ValueError
            If chart is not 'pie' or 'barh', or ncols is less than 1
    """
    if chart not in ('pie', 'barh'):
        raise ValueError(f"chart must be 'pie' or 'barh', got {chart!r}")

    if generic_colors is None:
        generic_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def draw(ax : Axes, panel : tuple) -> None:
        labels, values, title = panel
        colors : list = _sort_colors(color_dict, labels, generic_colors)

        if chart == 'pie':
            _draw_pie(
                ax=ax,
                x=values,
                labels=labels,
                colors=colors,
                title=title,
                pie_kwargs=chart_kwargs,
                title_kwargs=title_kwargs,
                setup_kwargs=setup_kwargs
            )
        else:
            _draw_barh(
                ax=ax,
                labels=labels,
                width=values,
                colors=colors,
                title=title,
                xlabel=xlabel,
                barh_kwargs=chart_kwargs,
                title_kwargs=title_kwargs,
                setup_kwargs=setup_kwargs
            )

    return _plot_grid(
        items=panels,
        draw=draw,
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        suptitle_kwargs=suptitle_kwargs
    )

def _plot_hist_grid(
        panels : list[tuple[pd.Series, str]],
        color : str,
        ncols : int,
        figsize : tuple,
        fig_title : str = '',
        hist_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots one histogram per panel, on a grid.

    Parameters
    ---
        panels : list[tuple[pd.Series, str]]
            One (series, title) entry per subplot, as built by
            _titled_series or _paired_series. Must not be empty
        color : str
            Bar color of every histogram
        ncols : int
            Number of subplot columns.
            Must be at least 1
        figsize : tuple
            Figure size in inches, as (width, height)
        fig_title : str, default ''
            Title for the whole figure
        hist_kwargs : dict | None, default None
            Overrides for the Axes.hist arguments of every histogram (e.g.
            bins, alpha)
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols).
            Unused subplots are hidden

    Raises
    ---
        ValueError
            If ncols is less than 1
    """
    def draw(ax : Axes, panel : tuple) -> None:
        series, title = panel
        _draw_hist(
            ax=ax,
            x=series,
            color=color,
            title=title,
            hist_kwargs=hist_kwargs,
            title_kwargs=title_kwargs,
            setup_kwargs=setup_kwargs
        )

    return _plot_grid(
        items=panels,
        draw=draw,
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        suptitle_kwargs=suptitle_kwargs
    )

### Public methods

def make_color_zone(
        ax : Axes,
        x,
        y1,
        y2,
        color : str | None = None,
        fill_between_kwargs : dict | None = None
    ):
    """Shades the area between two curves (or two constant values) on ax.

    Useful for highlighting ranges, such as a tolerance band or a
    threshold zone. Thin wrapper around Axes.fill_between.

    Parameters
    ---
        ax : matplotlib.axes.Axes
        x : array-like
            Horizontal coordinates that define the zone
        y1 : array-like | float
            First boundary of the zone
        y2 : array-like | float
            Second boundary of the zone
        color : str | None, default None
            Fill color. If None, matplotlib picks one from its color cycle
        fill_between_kwargs : dict | None, default None
            Overrides for the Axes.fill_between arguments (e.g. alpha,
            where, label). Take precedence over x, y1, y2 and color
    """
    fill_between_kwargs_all : dict = {
        'x':x,
        'y1':y1,
        'y2':y2,
        'color':color,
        **(fill_between_kwargs or {})
    }
    ax.fill_between(**fill_between_kwargs_all)

def plot_line_chart(
        x : pd.Series | np.ndarray,
        y : pd.Series | np.ndarray,
        xlabel : str,
        ylabel : str,
        label : str,
        color : str,
        ax : Axes | None = None,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        title : str = '',
        line_kwargs : dict | None = None,
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        legend_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> Axes | tuple[Figure, np.ndarray]:
    """Plots a line chart, either on an existing Axes or on a new figure.

    Parameters
    ---
        x : pd.Series | np.ndarray
            Horizontal coordinates
        y : pd.Series | np.ndarray
            Vertical coordinates, aligned with x
        xlabel : str
        ylabel : str
        label : str
            Legend entry for the line
        color : str
            Line color
        ax : matplotlib.axes.Axes | None, default None
            If given, the line is drawn on it and no figure is created.
            If None, a new figure with a single subplot is created
        fig_title : str, default ''
            Title for the whole figure. Only used when ax is None
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height).
            Only used when ax is None
        title : str, default ''
            Title of the plot. If empty, '{ylabel} por {xlabel}' is used
        line_kwargs : dict | None, default None
            Overrides for the Axes.plot arguments (see _draw_lineplot)
        title_kwargs : dict | None, default None
            Style overrides for the plot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format).
            Only used when ax is None
        legend_kwargs : dict | None, default None
            Overrides for the Axes.legend arguments (see _draw_lineplot)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        matplotlib.axes.Axes
            The given ax, if ax was provided
        tuple[matplotlib.figure.Figure, np.ndarray]
            (fig, axs), if ax was None. axs is a 2D array holding a single Axes
    """
    if title == '':
        title = f'{ylabel} por {xlabel}'

    def draw(ax : Axes, _ : None) -> None:
        _draw_lineplot(
            ax=ax,
            x=x,
            y=y,
            xlabel=xlabel,
            ylabel=ylabel,
            label=label,
            title=title,
            color=color,
            line_kwargs=line_kwargs,
            title_kwargs=title_kwargs,
            legend_kwargs=legend_kwargs,
            setup_kwargs=setup_kwargs
        )

    if ax is not None:
        draw(ax, None)
        return ax

    # a figure with a single subplot, built like the other plot_* functions
    return _plot_grid(
        items=[None],
        draw=draw,
        ncols=1,
        figsize=figsize,
        fig_title=fig_title,
        suptitle_kwargs=suptitle_kwargs
    )

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
        na_label : str = 'Sem dado',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        pie_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
) -> tuple[Figure, np.ndarray]:
    """Plots each column's value counts as a pie chart, one subplot per column.

    Parameters
    ---
        data : pd.DataFrame
            One pie chart is drawn per column in data. Must not be empty
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height)
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
        generic_colors : list | None, default None
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
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        pie_kwargs : dict | None, default None
            Overrides for the Axes.pie arguments of every chart (e.g.
            startangle). The default is autopct='%1.1f%%'
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

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

    panels : list = _value_counts_panels(
        _titled_series(data, title_prefix),
        sort_alphabetically=sort_alphabetically,
        max_categories=max_categories,
        other_label=other_label,
        dropna=dropna,
        na_label=na_label
    )

    return _plot_category_grid(
        panels=panels,
        chart='pie',
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        color_dict=color_dict,
        generic_colors=generic_colors,
        chart_kwargs=pie_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


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
        na_label : str = 'Sem dado',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        barh_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
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
            Figure size in inches, as (width, height)
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
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        barh_kwargs : dict | None, default None
            Overrides for the Axes.barh arguments of every chart (e.g.
            height, edgecolor)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

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

    panels : list = _value_counts_panels(
        _titled_series(data, title_prefix),
        sort_alphabetically=sort_alphabetically,
        max_categories=max_categories,
        other_label=other_label,
        dropna=dropna,
        na_label=na_label
    )

    return _plot_category_grid(
        panels=panels,
        chart='barh',
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        color_dict=color_dict,
        generic_colors=[generic_color],
        xlabel=xlabel,
        chart_kwargs=barh_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


### HISTOGRAM

def plot_hist(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        title_prefix : str = 'Distribuição de clientes por ',
        color : str = 'blue',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        hist_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots a histogram of each column of data, one subplot per column.

    Parameters
    ---
        data : pd.DataFrame
            One histogram is drawn per column in data. Must not be empty
            and should contain only numeric columns
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height)
        ncols : int, default 3
            Number of subplot columns. Unused subplots are hidden,
            row amount is calculated based on ncols
            Must be at least 1
        title_prefix : str, default 'Distribuição de clientes por '
            Text concatenated before the column name in each subplot title
        color : str, default 'blue'
            Bar color of every histogram
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        hist_kwargs : dict | None, default None
            Overrides for the Axes.hist arguments of every histogram (e.g.
            bins, alpha)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols)

    Raises
    ---
        ValueError
            If data is empty or ncols is less than 1
    """
    _validation_batch_plotters(
        data=data,
        ncols=ncols
    )

    panels : list = _titled_series(data, title_prefix)

    return _plot_hist_grid(
        panels=panels,
        color=color,
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        hist_kwargs=hist_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


def plot_alternating_hist(
        data1 : pd.DataFrame,
        data2 : pd.DataFrame,
        data1_label : str = 'Sample 1',
        data2_label : str = 'Sample 2',
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 2,
        title_prefix : str = 'Distribuição de clientes por ',
        color : str = 'blue',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        hist_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots histograms of two samples in alternation, for easy comparison.

    Columns are matched by name, in the order they appear in data1. For
    each column present in both DataFrames, the data1 histogram is drawn on
    one subplot and the data2 histogram on the next one, so the grid reads
    (col 1 of data1, col 1 of data2, col 2 of data1, ...). Columns that
    exist in only one of the DataFrames are skipped.

    Parameters
    ---
        data1 : pd.DataFrame
            First sample. Must not be empty and should contain only
            numeric columns. Column names must be unique
        data2 : pd.DataFrame
            Second sample. Must not be empty and should contain only
            numeric columns. Column names must be unique
        data1_label : str, default 'Sample 1'
            Name of the first sample, appended to its subplot titles
        data2_label : str, default 'Sample 2'
            Name of the second sample, appended to its subplot titles
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height)
        ncols : int, default 2
            Number of subplot columns. Must be at least 1.
            Two subplots are used per column, so an even value
            keeps each pair on the same row
        title_prefix : str, default 'Distribuição de clientes por '
            Text concatenated before the column name in each subplot title
            Titles follow the format '{title_prefix}{column} - {label}'
        color : str, default 'blue'
            Bar color of every histogram
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        hist_kwargs : dict | None, default None
            Overrides for the Axes.hist arguments of every histogram (e.g.
            bins, alpha)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols), where nrows
            fits 2 * (number of shared columns) subplots.
            Unused subplots are hidden

    Raises
    ---
        ValueError
            If data1 or data2 is empty, ncols is less than 1, either
            DataFrame has duplicate column names, or they share no column
    """
    common_cols : list = _validation_batch_comparers(
        data1,
        data2,
        ncols=ncols
    )

    panels : list = _paired_series(
        common_cols,
        data1,
        data2,
        data1_label,
        data2_label,
        title_prefix
    )

    return _plot_hist_grid(
        panels=panels,
        color=color,
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        hist_kwargs=hist_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


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
        na_label : str = 'Sem dado',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        pie_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
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
            Figure size in inches, as (width, height)
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
        generic_colors : list | None, default None
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
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        pie_kwargs : dict | None, default None
            Overrides for the Axes.pie arguments of every chart (e.g.
            startangle). The default is autopct='%1.1f%%'
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

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

    panels : list = _group_sum_panels(
        data,
        target_col,
        title_prefix,
        sort_alphabetically=sort_alphabetically,
        max_categories=max_categories,
        other_label=other_label,
        dropna=dropna,
        na_label=na_label
    )

    return _plot_category_grid(
        panels=panels,
        chart='pie',
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        color_dict=color_dict,
        generic_colors=generic_colors,
        chart_kwargs=pie_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


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
        na_label : str = 'Sem dado',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        barh_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
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
            Figure size in inches, as (width, height)
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
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        barh_kwargs : dict | None, default None
            Overrides for the Axes.barh arguments of every chart (e.g.
            height, edgecolor)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

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

    panels : list = _group_sum_panels(
        data,
        target_col,
        title_prefix,
        sort_alphabetically=sort_alphabetically,
        max_categories=max_categories,
        other_label=other_label,
        dropna=dropna,
        na_label=na_label
    )

    return _plot_category_grid(
        panels=panels,
        chart='barh',
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        color_dict=color_dict,
        generic_colors=[generic_color],
        xlabel=xlabel,
        chart_kwargs=barh_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


def plot_alternating_value_counts_barh(
        data1 : pd.DataFrame,
        data2 : pd.DataFrame,
        data1_label : str = 'Sample 1',
        data2_label : str = 'Sample 2',
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 2,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Clientes por ',
        xlabel : str = 'Qtd',
        color_dict : dict | None = None,
        generic_color : str = 'blue',
        max_categories : int = 10,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado',
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        barh_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots the value counts of two samples as horizontal bar charts, in alternation.
    Good for comparisons.

    Columns are matched by name, in the order they appear in data1. For
    each column present in both DataFrames, the data1 chart is drawn on
    one subplot and the data2 chart on the next one, so the grid reads
    (col 1 of data1, col 1 of data2, col 2 of data1, ...). Columns that
    exist in only one of the DataFrames are skipped. Bars are listed top
    to bottom: by descending frequency, or alphabetically if
    sort_alphabetically is True. The aggregated bar is always last.

    Parameters
    ---
        data1 : pd.DataFrame
            First sample. Must not be empty. Column names must be unique
        data2 : pd.DataFrame
            Second sample. Must not be empty. Column names must be unique
        data1_label : str, default 'Sample 1'
            Name of the first sample, appended to its subplot titles
        data2_label : str, default 'Sample 2'
            Name of the second sample, appended to its subplot titles
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height)
        ncols : int, default 2
            Number of subplot columns. Must be at least 1.
            Row amount is calculated based on ncols.
            Two subplots are used per column, so an even value
            is recommended to keep each pair on the same row
        sort_alphabetically : bool, default False
            If True, bars are ordered alphabetically by label instead of by
            frequency. The aggregated bar is always placed last
        title_prefix : str, default 'Clientes por '
            Text concatenated before the column name in each subplot title.
            Titles follow the format '{title_prefix}{column} - {label}'
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
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        barh_kwargs : dict | None, default None
            Overrides for the Axes.barh arguments of every chart (e.g.
            height, edgecolor)
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols), where nrows
            fits 2 * (number of shared columns) subplots.
            Unused subplots are hidden

    Raises
    ---
        ValueError
            If data1 or data2 is empty, max_categories is less than 2,
            ncols is less than 1, either DataFrame has duplicate column
            names, or they share no column
    """
    common_cols : list = _validation_batch_comparers(
        data1,
        data2,
        max_categories=max_categories,
        ncols=ncols
    )

    panels : list = _value_counts_panels(
        _paired_series(
            common_cols,
            data1,
            data2,
            data1_label,
            data2_label,
            title_prefix
        ),
        sort_alphabetically=sort_alphabetically,
        max_categories=max_categories,
        other_label=other_label,
        dropna=dropna,
        na_label=na_label
    )

    return _plot_category_grid(
        panels=panels,
        chart='barh',
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        color_dict=color_dict,
        generic_colors=[generic_color],
        xlabel=xlabel,
        chart_kwargs=barh_kwargs,
        title_kwargs=title_kwargs,
        suptitle_kwargs=suptitle_kwargs,
        setup_kwargs=setup_kwargs
    )


def plot_comparing_box(
        data1 : pd.DataFrame,
        data2 : pd.DataFrame,
        data1_label : str = 'Sample 1',
        data2_label : str = 'Sample 2',
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 2,
        title_prefix : str = 'Clientes por ',
        max_categories : int = 10,
        title_kwargs : dict | None = None,
        suptitle_kwargs : dict | None = None,
        boxplot_kwargs : dict | None = None,
        setup_kwargs : dict | None = None
    ) -> tuple[Figure, np.ndarray]:
    """Plots, for each column, a box plot of data1 next to one of data2 on the same subplot.

    Columns are matched by name, in the order they appear in data1. Each
    column present in both DataFrames gets one subplot, with the data1 box
    at position 0 and the data2 box at position 1, labeled by data1_label
    and data2_label. Columns that exist in only one of the DataFrames are
    skipped.

    Parameters
    ---
        data1 : pd.DataFrame
            First sample. Must not be empty and should contain only
            numeric columns. Column names must be unique
        data2 : pd.DataFrame
            Second sample. Must not be empty and should contain only
            numeric columns. Column names must be unique
        data1_label : str, default 'Sample 1'
            Tick label of the first sample's box
        data2_label : str, default 'Sample 2'
            Tick label of the second sample's box
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
            Figure size in inches, as (width, height)
        ncols : int, default 2
            Number of subplot columns. Must be at least 1.
            Row amount is calculated based on ncols
        title_prefix : str, default 'Clientes por '
            Text concatenated before the column name in each subplot title
        max_categories : int, default 10
            Not used. Accepted only so existing calls keep working
        title_kwargs : dict | None, default None
            Style overrides for each subplot title (see _title_format)
        suptitle_kwargs : dict | None, default None
            Style overrides for the figure title (see fig_suptitle_format)
        boxplot_kwargs : dict | None, default None
            Overrides for the Axes.boxplot arguments of both boxes (e.g.
            showfliers, notch). Overriding positions or tick labels
            would affect both boxes, so leave those alone
        setup_kwargs : dict | None, default None
            Keyword arguments forwarded to _visual_setup for every subplot
            (kwargs_major, kwargs_minor, facecolor)

    Returns
    ---
        fig : matplotlib.figure.Figure
        axs : np.ndarray
            2D array of Axes with shape (nrows, ncols), where nrows
            fits one subplot per shared column. Unused subplots are hidden

    Raises
    ---
        ValueError
            If data1 or data2 is empty, ncols is less than 1, either
            DataFrame has duplicate column names, or they share no column
    """
    common_cols : list = _validation_batch_comparers(
        data1,
        data2,
        ncols=ncols
    )

    def draw(ax : Axes, col : Hashable) -> None:
        for series, label, position in (
            (data1[col], data1_label, 0),
            (data2[col], data2_label, 1)
        ):
            _draw_boxplot(
                ax=ax,
                x=series,
                label=label,
                position=position,
                title=f'{title_prefix}{col}',
                boxplot_kwargs=boxplot_kwargs,
                title_kwargs=title_kwargs,
                setup_kwargs=setup_kwargs
            )

    return _plot_grid(
        items=common_cols,
        draw=draw,
        ncols=ncols,
        figsize=figsize,
        fig_title=fig_title,
        suptitle_kwargs=suptitle_kwargs
    )