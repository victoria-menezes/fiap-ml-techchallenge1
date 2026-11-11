import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _get_number_of_rows(
        amount_of_plots : int,
        ncols : int
) -> int:
    """Calculates how many rows are needed to accomodate amount_of_plots over ncols"""
    nrows : int = int(np.ceil(amount_of_plots / ncols))
    return nrows

# def _get_(value_counts_series : pd.Series):
#     """Returns the labels of the value counts, and the value
#     counts of the given value_counts series"""
    
#     value_counts_data : pd.Series = value_counts_series.value_counts()
#     value_labels : list | pd.Index = value_counts_series.value_counts().index

#     return value_labels ,value_counts_data

def _sort_labels_alphabetically(
            value_labels : pd.Index,
            value_counts_data : pd.Series
    ):
        """Returns value_labels, value_counts_data sorted alphabetically"""
        sorted_pairs : list[tuple] = sorted(zip(value_labels, value_counts_data))
        ordered_value_labels, ordered_value_counts_data = zip(*sorted_pairs)

        return ordered_value_labels, ordered_value_counts_data

def _sort_colors(
        color_dict : dict | None,
        value_labels : list | pd.Index | tuple,
        generic_colors : list
    ):
    """Returns a list of ordered colors according to the labels and the color dictionary"""
    
    ordered_colors : list = []
    if color_dict is None:
        ordered_colors = generic_colors
    else:
        generic_color_count : int = 0
        for label in value_labels:
            if label in color_dict.keys():
                ordered_colors.append(color_dict[label])
            else:
                ordered_colors.append(generic_colors[generic_color_count])
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
        color_dict : dict | None = None
):
    """Plots columns' value counts in pie charts
    
    Parameters
    ---
        data : pd.Dataframe
        figsize : tuple, default (16, 6)
        ncols : int
        max_categories : int
        sort_alphabetically : bool
            If True, the category axis will be sorter alphabetically
        title_suffix : str, default 'Distribuição de clientes por '
            Text that will be concatenated before the column name in the graph's title
        color_dict : dict | None, default None
            label:color dictionary to color pie slices according to their label
    
    Returns
    ---
        fig, axs
    """
    generic_colors : list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    nrows : int = _get_number_of_rows(len(data.columns), ncols)
    cols = [str(col) for col in data.columns]
    fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
    ordered_colors : list

    iterable : zip
    if nrows == 1 and ncols == 1:
        iterable = zip([cols[0]], [axs])
    else:
        iterable = zip(cols, axs.flatten())
        

    for col, ax in iterable:
        value_counts_series : pd.Series = data[col].value_counts()
        value_labels_index : pd.Index = data[col].value_counts().index

        value_labels : tuple
        value_counts : tuple

        if sort_alphabetically:
            value_labels, value_counts = _sort_labels_alphabetically(value_labels_index, value_counts_series)
        else:
            value_labels = tuple(value_labels)
            value_counts = tuple(value_counts)

        ordered_colors = _sort_colors(color_dict, value_labels, generic_colors)        

        ax.pie(
            x = value_counts,
            labels = value_labels,
            autopct='%1.1f%%',
            colors = ordered_colors
        )

        ax.set_title(f'{title_suffix}{col}')
    
    fig.suptitle(fig_title)
    plt.tight_layout()
    return fig, axs
    if nrows == 1 and ncols == 1:
        value_counts_data : pd.Series = data[cols[0]].value_counts()
        value_labels : list | pd.Index = data[cols[0]].value_counts().index

        if sort_alphabetically:
            value_labels, value_counts_data = _sort_labels_alphabetically(value_labels, value_counts_data)

        ordered_colors = _sort_colors(color_dict, value_labels, generic_colors)

        axs.pie(
            x = value_counts_data,
            labels = value_labels,
            autopct='%1.1f%%'
        )

        axs.set_title(f'{title_suffix}{cols[0]}')

    else:
        count : int = 0
        for col, ax in zip(cols, axs.flatten()):
            value_counts_data : pd.Series = data[col].value_counts()
            value_labels : list | pd.Index = data[col].value_counts().index

            if sort_alphabetically:
                value_labels, value_counts_data = _sort_labels_alphabetically(value_labels, value_counts_data)
            
            ordered_colors = _sort_colors(color_dict, value_labels, generic_colors)

            ax.pie(
                x = value_counts_data,
                labels = value_labels,
                autopct='%1.1f%%',
                colors=ordered_colors
            )

            ax.set_title(f'{title_suffix}{col}')
            count +=1
    
    fig.suptitle(fig_title)
    plt.tight_layout()
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