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
        key = lambda p : str(p[0].casefold())
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

### Public methods

def plot_value_counts_pie(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        sort_alphabetically : bool = False,
        title_suffix : str = 'Distribuição de clientes por ',
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
        title_suffix : str, default 'Distribuição de clientes por '
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
    if max_categories<2:
        raise ValueError('max_categories must be 2 or higher') 

    generic_colors : list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    nrows : int = _get_number_of_rows(len(data.columns), ncols)
    cols : list = list(data.columns)
    fig, axs = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        squeeze = False,
        layout='constrained'
        )

    for col, ax in zip(cols, axs.flatten()):
        value_counts_series : pd.Series = data[col].value_counts(dropna=dropna)
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

        ordered_colors : list = _sort_colors(color_dict, value_labels, generic_colors)
        
        ax.pie(
            x = value_counts,
            labels = value_labels,
            autopct='%1.1f%%',
            colors = ordered_colors
        )
        
        ax.set_title(f'{title_suffix}{col}')

    # hiding unused axs
    for ax in axs.flatten()[len(cols):]:
        ax.set_visible(False)
    
    fig.suptitle(fig_title)
    return fig, axs

def plot_value_counts(
        data : pd.DataFrame,
        fig_title : str = '',
        figsize : tuple = (16, 6),
        ncols : int = 3,
        max_categories : int = 10,
        sort_alphabetically : bool = False,
        title_suffix : str = 'Clientes por ',
        xlabel : str = 'Qtd'
):
    return
    """Plots columns' value counts in horizontal bar charts
    
    Parameters
    ---
        data : pd.Dataframe
        figsize : tuple, default (16, 6)
        ncols : int
        max_categories : int
        sort_alphabetically : bool
            If True, the category axis will be sorter alphabetically
        title_suffix : str, default 'Clientes por '
            Text that will be concatenated before the column name in the graph's title
        xlabel : str, default 'Qtd'
    
    Returns
    ---
        fig, axs
    """
    nrows : int = _get_number_of_rows(len(data.columns), ncols)
    cols = [str(col) for col in data.columns]
    fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)

    if nrows == 1 and ncols == 1:
        value_counts_data : pd.Series = data[cols[0]].value_counts()
        value_labels : list | pd.Index = data[cols[0]].value_counts().index

        if sort_alphabetically:
            value_labels, value_counts_data = _sort_labels_alphabetically(value_labels, value_counts_data)
        
        ordered_colors = _sort_colors(color_dict, value_labels, generic_colors)

        # if there are too many categories to plot
        if len(value_labels) > max_categories:
            axs.barh(y = value_labels[:max_categories], width = value_counts[:10])
        else:
            axs.barh(y = value_labels, width = value_counts)

        axs.set_title(f'{title_suffix}{cols[0]}')
        axs.set_xlabel(xlabel)

    else:
        for col, ax in zip(cols, axs.flatten()):
            value_labels = [str(label) for label in data[col].value_counts().index]

            value_counts = data[col].value_counts().values

            if sort_alphabetically:
                sorted_pairs = sorted(zip(value_labels, value_counts))
                value_labels, value_counts = zip(*sorted_pairs)

            # if there are too many categories to plot
            if len(value_labels) > max_categories:
                ax.barh(y = value_labels[:max_categories], width = value_counts[:10])
            else:
                ax.barh(y = value_labels, width = value_counts)

            ax.set_title(f'{title_suffix}{col}')
            ax.set_xlabel(xlabel)

    fig.suptitle(fig_title)
    plt.tight_layout()
    return fig, axs