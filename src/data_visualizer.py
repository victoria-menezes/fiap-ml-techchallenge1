import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    Returns
    ---
        int
    """
    return int(np.ceil(amount_of_plots / ncols))

def _sort_labels_alphabetically(
            value_labels : pd.Index | list | tuple,
            value_counts_data : pd.Series | list | tuple
    ) -> tuple[tuple, tuple]: 
    """Sorts labels alphabetically, keeping each count paired with its label.

    Labels are compared by their string representation, so labels of mixed
    types (e.g. strings and ints) can be sorted together.

    Parameters
    ---
        value_labels : pd.Index | list | tuple
        value_counts_data : pd.Series | list | tuple
            Counts, in the same order as value_labels

    Returns
    ---
        tuple[tuple, tuple]
            (sorted labels, counts reordered to match)
            Two empty tuples if the input is empty
    """
    sorted_pairs : list = sorted(
        zip(value_labels, value_counts_data),
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
    """Keeps the most frequent categories and sums the rest into a single one.

    Expects value_counts sorted in descending order (as returned by
    Series.value_counts()). If there are more than max_categories categories,
    the top (max_categories - 1) are kept and the remainder is summed into
    other_label, so the result has exactly max_categories entries.

    Parameters
    ---
        value_counts : pd.Series
            Counts indexed by category, sorted in descending order
        max_categories : int
            Maximum number of entries in the result, including other_label
        other_label : str, default 'Outros'
            Label for the aggregated entry. Should not match an existing category

    Returns
    ---
        pd.Series
            value_counts unchanged if len(value_counts) <= max_categories,
            otherwise the grouped counts
    """
    if len(value_counts) <= max_categories:
        return value_counts

    head : pd.Series = value_counts.iloc[:max_categories - 1]
    other_total : int = int(value_counts.iloc[max_categories - 1:].sum())
    return pd.concat([head, pd.Series({other_label: other_total})])


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

def _get_value_labels_and_counts(
        series : pd.Series,
        sort_alphabetically : bool = False,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado',
    ):
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
    """
    value_counts_series : pd.Series = series.value_counts(dropna=dropna)
    value_counts_series = value_counts_series[value_counts_series > 0] # ignores values with 0 incidence

    value_counts_series.index = value_counts_series.index.astype(object).fillna(na_label) # replaces NA label with na_label
    value_counts_series = _group_overflow(value_counts_series, max_categories, other_label)

    value_labels : tuple
    value_counts : tuple
    if sort_alphabetically:
        other_count : int | None = value_counts_series.get(other_label)
        visible_categories : pd.Series = value_counts_series.drop(other_label, errors='ignore') # runs even if there is no other_label column

        value_labels, value_counts = _sort_labels_alphabetically(visible_categories.index, visible_categories)

        # others at the end
        if other_count is not None:
            value_labels, value_counts = (*value_labels, other_label), (*value_counts, other_count)
    else:
        value_labels = tuple(value_counts_series.index)
        value_counts = tuple(value_counts_series)
    
    return value_labels, value_counts

def _plot_arg_validation(
        data : pd.DataFrame | None = None,
        max_categories : int | None = None,
        ncols : int | None = None
):
    """Validates arguments shared by the plotting functions.

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
    if max_categories is not None and max_categories<2:
        raise ValueError('max_categories must be 2 or higher')
    if ncols is not None and ncols<1:
        raise ValueError('ncols must be at least 1')

### Public methods

def plot_value_counts_pie(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_prefix : str = 'Distribuição de clientes por ',
        color_dict : dict | None = None,
        max_categories : int = 5,
        other_label : str = 'Outros',
        dropna : bool = False,
        na_label : str = 'Sem dado'
):
    """Plots each column's value counts as a pie chart, one subplot per column.

    Parameters
    ---
        data : pd.DataFrame
            One pie chart is drawn per column in data
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Unused subplots are hidden,
            row amount is calculated based on ncols
        sort_alphabetically : bool, default False
            If True, slices are ordered alphabetically by label instead of by
            frequency. The aggregated slice is always placed last
        title_prefix : str, default 'Distribuição de clientes por '
            Text concatenated before the column name in each subplot title
        color_dict : dict | None, default None
            label:color dictionary to color slices according to their label.
            Unmapped labels use the default matplotlib color cycle        
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
    """
    _plot_arg_validation(
            data=data,
            max_categories=max_categories,
            ncols=ncols
        )

    generic_colors : list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    nrows : int = _get_number_of_rows(len(data.columns), ncols)
    fig, axs = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        squeeze = False,
        layout='constrained'
        )

    for (col, series), ax in zip(data.items(), axs.flatten()):
        value_labels, value_counts = _get_value_labels_and_counts(
            series,
            sort_alphabetically = sort_alphabetically,
            max_categories = max_categories,
            other_label = other_label,
            dropna=dropna,
            na_label=na_label
            )

        ordered_colors : list = _sort_colors(color_dict, value_labels, generic_colors)
        
        ax.pie(
            x = value_counts,
            labels = value_labels,
            autopct='%1.1f%%',
            colors = ordered_colors
        )
        
        ax.set_title(f'{title_prefix}{col}')

    # hiding unused axs
    for ax in axs.flatten()[len(data.columns):]:
        ax.set_visible(False)
    
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
):
    """Plots each column's value counts as a horizontal bar chart, one subplot per column.

    Bars are listed top to bottom: by descending frequency, or alphabetically
    if sort_alphabetically is True.
    The aggregated bar is always last.

    Parameters
    ---
        data : pd.DataFrame
            One bar chart is drawn per column in data
        fig_title : str, default ''
            Title for the whole figure
        figsize : tuple, default (16, 6)
        ncols : int, default 3
            Number of subplot columns. Unused subplots are hidden,
            row amount is calculated based on ncols
        sort_alphabetically : bool, default False
            If True, bars are ordered alphabetically by label instead of by
            frequency.
            The aggregated bar is always placed last
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
    """
    
    _plot_arg_validation(
        data=data,
        max_categories=max_categories,
        ncols=ncols
    )

    nrows : int = _get_number_of_rows(len(data.columns), ncols)
    fig, axs = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        squeeze = False,
        layout='constrained'
        )

    for (col, series), ax in zip(data.items(), axs.flatten()):
        value_labels, value_counts = _get_value_labels_and_counts(
            series,
            sort_alphabetically = sort_alphabetically,
            max_categories = max_categories,
            other_label = other_label,
            dropna=dropna,
            na_label=na_label
            )

        ordered_colors : list = _sort_colors(color_dict, value_labels, [generic_color])
        
        # barh needs strings as labels
        value_labels_strings : list[str] = [str(label) for label in value_labels]
        ax.barh(
            y = value_labels_strings,
            width = value_counts,
            color = ordered_colors
        )

        ax.invert_yaxis()
        
        ax.set_title(f'{title_prefix}{col}')
        ax.set_xlabel(xlabel)
        
    # hiding unused axs
    for ax in axs.flatten()[len(data.columns):]:
        ax.set_visible(False)
    
    fig.suptitle(fig_title)
    return fig, axs