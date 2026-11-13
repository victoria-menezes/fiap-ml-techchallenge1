import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_visualizer import (
    _get_number_of_rows,
    _sort_labels_alphabetically,
    _group_overflow,
    _sort_colors,
    _get_value_counts_and_sort_labels,
    _validation_batch_plotters,
    plot_value_counts_pie,
    plot_value_counts,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')  # avoids leaking figures between tests

class TestGetNumberOfRows:
    @pytest.mark.parametrize('plots, ncols, expected', [
        (1, 3, 1),
        (3, 3, 1),
        (4, 3, 2),
        (6, 3, 2),
        (7, 3, 3),
        (5, 1, 5),
    ])
    def test_rounds_up(self, plots, ncols, expected):
        assert _get_number_of_rows(plots, ncols) == expected

class TestSortLabelsAlphabetically:
    def test_keeps_counts_paired(self):
        labels, counts = _sort_labels_alphabetically(['b', 'c', 'a'], [2, 3, 1])
        assert labels == ('a', 'b', 'c')
        assert counts == (1, 2, 3)

    def test_is_case_insensitive(self):
        labels, _ = _sort_labels_alphabetically(['Zebra', 'apple'], [1, 2])
        assert labels == ('apple', 'Zebra')

    def test_handles_mixed_types(self):
        labels, counts = _sort_labels_alphabetically([10, 'a', 2], [1, 2, 3])
        assert labels == (10, 2, 'a')  # compared as '10' < '2' < 'a'
        assert counts == (1, 3, 2)

    def test_empty_input(self):
        assert _sort_labels_alphabetically([], []) == ((), ())

    def test_accepts_pandas_objects(self):
        labels, counts = _sort_labels_alphabetically(
            pd.Index(['b', 'a']), pd.Series([5, 7])
        )
        assert labels == ('a', 'b')
        assert counts == (7, 5)


class TestGroupOverflow:
    def test_returns_unchanged_when_under_limit(self):
        series = pd.Series({'A': 5, 'B': 3})
        pd.testing.assert_series_equal(_group_overflow(series, 5), series)

    def test_returns_unchanged_when_exactly_at_limit(self):
        # no relabeling at exactly max_categories
        series = pd.Series({'A': 50, 'B': 30, 'C': 10, 'D': 6, 'E': 4})
        result = _group_overflow(series, 5)
        pd.testing.assert_series_equal(result, series)
        assert 'Outros' not in result.index

    def test_groups_tail(self):
        series = pd.Series({'A': 50, 'B': 30, 'C': 10, 'D': 6, 'E': 4})
        result = _group_overflow(series, 3)
        assert list(result.index) == ['A', 'B', 'Outros']
        assert list(result.values) == [50, 30, 20]

    def test_result_length_equals_max_categories(self):
        series = pd.Series(range(20, 0, -1), index=[f'c{i}' for i in range(20)])
        assert len(_group_overflow(series, 7)) == 7

    def test_preserves_total(self):
        series = pd.Series({'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 4})
        assert _group_overflow(series, 3).sum() == series.sum()

    def test_custom_label(self):
        series = pd.Series({'A': 3, 'B': 2, 'C': 1})
        assert 'Other' in _group_overflow(series, 2, other_label='Other').index

class TestSortColors:
    GENERIC = ['g0', 'g1', 'g2']

    def test_none_returns_generic(self):
        assert _sort_colors(None, ['a', 'b'], self.GENERIC) == self.GENERIC

    def test_uses_dict_and_falls_back_in_order(self):
        result = _sort_colors({'a': 'red'}, ['a', 'b', 'c'], self.GENERIC)
        assert result == ['red', 'g0', 'g1']

    def test_wraps_around_generic(self):
        result = _sort_colors({}, list('abcde'), self.GENERIC)
        assert result == ['g0', 'g1', 'g2', 'g0', 'g1']

    def test_mapped_labels_do_not_consume_generic(self):
        result = _sort_colors({'b': 'blue'}, ['a', 'b', 'c'], self.GENERIC)
        assert result == ['g0', 'blue', 'g1']

class TestPlotArgValidation:

    def test_no_arguments_does_not_raise(self):
        _validation_batch_plotters()

    def test_valid_arguments_do_not_raise(self):
        _validation_batch_plotters(
            data=pd.DataFrame({'a': ['x']}), max_categories=2, ncols=1
        )

    @pytest.mark.parametrize('empty', [
        pd.DataFrame(),                # no rows, no columns
        pd.DataFrame({'a': []}),       # column but no rows
    ])
    def test_rejects_empty_dataframe(self, empty):
        with pytest.raises(ValueError, match='empty'):
            _validation_batch_plotters(data=empty)

    @pytest.mark.parametrize('bad', [1, 0, -5])
    def test_rejects_small_max_categories(self, bad):
        with pytest.raises(ValueError, match='max_categories'):
            _validation_batch_plotters(max_categories=bad)

    @pytest.mark.parametrize('bad', [0, -1])
    def test_rejects_small_ncols(self, bad):
        with pytest.raises(ValueError, match='ncols'):
            _validation_batch_plotters(ncols=bad)

    def test_only_checks_arguments_that_were_given(self):
        # an invalid ncols is reported even when the others are omitted
        with pytest.raises(ValueError, match='ncols'):
            _validation_batch_plotters(ncols=0)


@pytest.mark.parametrize('plot', [plot_value_counts_pie, plot_value_counts])
class TestPlotFunctionsValidation:
    """Both public functions must validate before drawing anything."""

    def test_rejects_empty_dataframe(self, plot):
        with pytest.raises(ValueError, match='empty'):
            plot(pd.DataFrame())

    def test_rejects_small_max_categories(self, plot):
        with pytest.raises(ValueError, match='max_categories'):
            plot(pd.DataFrame({'a': ['x']}), max_categories=1)

    def test_rejects_small_ncols(self, plot):
        with pytest.raises(ValueError, match='ncols'):
            plot(pd.DataFrame({'a': ['x']}), ncols=0)

    def test_no_figure_is_created_when_validation_fails(self, plot):
        with pytest.raises(ValueError):
            plot(pd.DataFrame({'a': ['x']}), ncols=0)
        assert plt.get_fignums() == []

class TestGetValueLabelsAndCounts:
    def test_returns_tuples_ordered_by_frequency(self):
        labels, counts = _get_value_counts_and_sort_labels(pd.Series(list('aaabbc')))
        assert labels == ('a', 'b', 'c')
        assert counts == (3, 2, 1)

    def test_groups_overflow_into_other(self):
        labels, counts = _get_value_counts_and_sort_labels(
            pd.Series(list('aaabbc')), max_categories=2
        )
        assert labels == ('a', 'Outros')
        assert counts == (3, 3)

    def test_custom_other_label(self):
        labels, _ = _get_value_counts_and_sort_labels(
            pd.Series(list('aaabbc')), max_categories=2, other_label='Misc'
        )
        assert labels == ('a', 'Misc')

    def test_sort_alphabetically_keeps_counts_paired(self):
        labels, counts = _get_value_counts_and_sort_labels(
            pd.Series(list('bbbaac')), sort_alphabetically=True
        )
        assert labels == ('a', 'b', 'c')
        assert counts == (2, 3, 1)

    def test_other_goes_last_when_sorted_alphabetically(self):
        series = pd.Series(['z'] * 5 + ['b'] * 4 + ['a'] + ['c'])
        labels, counts = _get_value_counts_and_sort_labels(
            series, max_categories=3, sort_alphabetically=True
        )
        assert labels == ('b', 'z', 'Outros')
        assert counts == (4, 5, 2)

    def test_na_shown_with_default_label(self):
        labels, counts = _get_value_counts_and_sort_labels(pd.Series(['x', 'x', None]))
        assert labels == ('x', 'Sem dado')
        assert counts == (2, 1)

    def test_custom_na_label(self):
        labels, _ = _get_value_counts_and_sort_labels(
            pd.Series(['x', None]), na_label='Missing'
        )
        assert 'Missing' in labels

    def test_dropna_removes_na(self):
        labels, counts = _get_value_counts_and_sort_labels(
            pd.Series(['x', 'x', None]), dropna=True
        )
        assert labels == ('x',)
        assert counts == (2,)

    def test_ignores_unobserved_categorical_levels(self):
        cat = pd.Series(pd.Categorical(['a', 'b'], categories=['a', 'b', 'unused']))
        labels, _ = _get_value_counts_and_sort_labels(cat)
        assert 'unused' not in labels

    @pytest.mark.parametrize('sort', [False, True])
    def test_nothing_to_count_returns_empty_tuples(self, sort):
        series = pd.Series([None, None], dtype=object)
        result = _get_value_counts_and_sort_labels(
            series, dropna=True, sort_alphabetically=sort
        )
        assert result == ((), ())

    def test_total_is_preserved_when_grouping(self):
        series = pd.Series(list('aaaabbbcc') + ['d', 'e'])
        _, counts = _get_value_counts_and_sort_labels(series, max_categories=3)
        assert sum(counts) == len(series)

    def test_does_not_modify_input(self):
        series = pd.Series(['x', 'x', None])
        before = series.copy()
        _get_value_counts_and_sort_labels(series)
        pd.testing.assert_series_equal(series, before)


###

class _SharedPlotTests:
    """Behavior common to plot_value_counts_pie and plot_value_counts.

    Subclasses must define `plot`, `labels(ax)` and `sizes(ax)`.
    The leading underscore keeps pytest from collecting this class directly;
    only the Test* subclasses run.
    """

    @staticmethod
    def _plot(*args, **kwargs):
        raise NotImplementedError('_plot not implemented')

    @staticmethod
    def _get_labels(ax):
        raise NotImplementedError('_get_labels not implemented')

    @staticmethod
    def _get_sizes(ax):
        """Values proportional to each category's count, in drawing order."""
        raise NotImplementedError('_get_sizes not implemented')

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'color': ['red'] * 5 + ['blue'] * 3 + ['green'] * 2,
            'size':  ['S'] * 6 + ['M'] * 3 + ['L'] * 1,
        })

    # figure structure

    def test_returns_fig_and_2d_axes(self, df):
        fig, axs = self._plot(df)
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 3)

    def test_single_column_still_2d(self):
        _, axs = self._plot(pd.DataFrame({'a': ['x', 'y']}), ncols=1)
        assert axs.shape == (1, 1)

    def test_row_count(self):
        data = pd.DataFrame({f'c{i}': ['a', 'b'] for i in range(5)})
        _, axs = self._plot(data, ncols=2)
        assert axs.shape == (3, 2)

    def test_hides_unused_axes(self, df):
        _, axs = self._plot(df, ncols=3)
        assert axs[0, 0].get_visible()
        assert axs[0, 1].get_visible()
        assert not axs[0, 2].get_visible()

    # titles

    def test_titles_use_prefix_and_column_name(self, df):
        _, axs = self._plot(df, title_prefix='By ')
        assert axs[0, 0].get_title() == 'By color'
        assert axs[0, 1].get_title() == 'By size'

    def test_sets_figure_title(self, df):
        fig, _ = self._plot(df, fig_title='Hello')
        assert fig._suptitle.get_text() == 'Hello'

    # ordering

    def test_ordered_by_frequency_by_default(self, df):
        _, axs = self._plot(df)
        assert self._get_labels(axs[0, 0]) == ['red', 'blue', 'green']

    def test_sort_alphabetically_keeps_labels_and_sizes_aligned(self, df):
        # regression: labels sorted but values left in frequency order
        _, axs = self._plot(df, sort_alphabetically=True)
        ax = axs[0, 1]  # size: S=6, M=3, L=1
        assert self._get_labels(ax) == ['L', 'M', 'S']
        sizes = self._get_sizes(ax)
        assert sizes[0] < sizes[1] < sizes[2]   # L < M < S

    # grouping

    def test_groups_overflow_into_other(self, df):
        _, axs = self._plot(df, max_categories=2)
        assert self._get_labels(axs[0, 0]) == ['red', 'Outros']

    def test_other_goes_last_when_sorted_alphabetically(self):
        data = pd.DataFrame({'a': ['z'] * 5 + ['b'] * 4 + ['a'] + ['c']})
        _, axs = self._plot(data, max_categories=3, sort_alphabetically=True)
        assert self._get_labels(axs[0, 0]) == ['b', 'z', 'Outros']

    def test_custom_other_label(self):
        data = pd.DataFrame({'a': list('aaabbc')})
        _, axs = self._plot(data, max_categories=2, other_label='Misc')
        assert 'Misc' in self._get_labels(axs[0, 0])

    # missing / unobserved values

    def test_na_shown_with_label_by_default(self):
        data = pd.DataFrame({'a': ['x', 'x', None]})
        _, axs = self._plot(data)
        assert 'Sem dado' in self._get_labels(axs[0, 0])

    def test_dropna_removes_na(self):
        data = pd.DataFrame({'a': ['x', 'x', None]})
        _, axs = self._plot(data, dropna=True)
        assert self._get_labels(axs[0, 0]) == ['x']

    def test_custom_na_label(self):
        data = pd.DataFrame({'a': ['x', None]})
        _, axs = self._plot(data, na_label='Missing')
        assert 'Missing' in self._get_labels(axs[0, 0])

    def test_ignores_unobserved_categorical_levels(self):
        cat = pd.Categorical(['a', 'b'], categories=['a', 'b', 'unused'])
        _, axs = self._plot(pd.DataFrame({'a': cat}))
        assert 'unused' not in self._get_labels(axs[0, 0])

    # column names

    def test_supports_non_string_column_names(self):
        data = pd.DataFrame({0: ['a', 'b'], 2024: ['x', 'x']})
        _, axs = self._plot(data)
        assert axs[0, 0].get_title().endswith('0')

    def test_duplicate_column_names_are_plotted_separately(self):
        # regression: data[col] returned a DataFrame for repeated names
        data = pd.DataFrame({'c': ['a', 'a', 'b'], 'd': ['x', 'y', 'y']})
        data.columns = ['c', 'c']
        _, axs = self._plot(data)
        assert self._get_labels(axs[0, 0]) == ['a', 'b']
        assert self._get_labels(axs[0, 1]) == ['y', 'x']

    # validation

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match='empty'):
            self._plot(pd.DataFrame())

    def test_rejects_small_max_categories(self):
        with pytest.raises(ValueError, match='max_categories'):
            self._plot(pd.DataFrame({'a': ['x']}), max_categories=1)

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            self._plot(pd.DataFrame({'a': ['x']}), ncols=0)

    def test_no_figure_is_created_when_validation_fails(self):
        with pytest.raises(ValueError):
            self._plot(pd.DataFrame({'a': ['x']}), ncols=0)
        assert plt.get_fignums() == []


class TestPlotValueCountsPie(_SharedPlotTests):
    _plot = staticmethod(plot_value_counts_pie)

    @staticmethod
    def _get_labels(ax):
        return [t.get_text() for t in ax.texts if not t.get_text().endswith('%')]

    @staticmethod
    def _get_sizes(ax):
        return [p.theta2 - p.theta1 for p in ax.patches]

    def test_applies_color_dict(self):
        data = pd.DataFrame({'a': ['x'] * 3 + ['y'] * 2})
        _, axs = plot_value_counts_pie(data, color_dict={'x': '#ff0000'})
        first_wedge = axs[0, 0].patches[0]
        assert matplotlib.colors.to_hex(first_wedge.get_facecolor()) == '#ff0000'

    # colors

    def test_applies_color_dict(self):
        data = pd.DataFrame({'a': ['x'] * 3 + ['y'] * 2})
        _, axs = plot_value_counts_pie(data, color_dict={'x': '#ff0000'})
        first_wedge = axs[0, 0].patches[0]
        assert matplotlib.colors.to_hex(first_wedge.get_facecolor()) == '#ff0000'

class TestPlotValueCounts(_SharedPlotTests):
    _plot = staticmethod(plot_value_counts)

    @staticmethod
    def _get_labels(ax):
        ax.figure.canvas.draw()
        return [t.get_text() for t in ax.get_yticklabels()]

    @staticmethod
    def _get_sizes(ax):
        return [p.get_width() for p in ax.patches]

    def test_first_category_is_on_top(self, df):
        _, axs = plot_value_counts(df)
        assert axs[0, 0].yaxis_inverted()

    def test_sets_xlabel(self, df):
        _, axs = plot_value_counts(df, xlabel='Count')
        assert axs[0, 0].get_xlabel() == 'Count'
        assert axs[0, 1].get_xlabel() == 'Count'

    def test_integer_labels_are_drawn(self):
        data = pd.DataFrame({'n': [1, 1, 2]})
        _, axs = plot_value_counts(data)
        assert self._get_labels(axs[0, 0]) == ['1', '2']
        assert self._get_sizes(axs[0, 0]) == [2, 1]

    # colors

    def test_uses_generic_color_by_default(self, df):
        _, axs = plot_value_counts(df)
        for bar in axs[0, 0].patches:
            assert matplotlib.colors.to_hex(bar.get_facecolor()) == '#0000ff'

    def test_custom_generic_color(self, df):
        _, axs = plot_value_counts(df, generic_color='#00ff00')
        for bar in axs[0, 0].patches:
            assert matplotlib.colors.to_hex(bar.get_facecolor()) == '#00ff00'

    def test_color_dict_overrides_generic_color(self):
        data = pd.DataFrame({'a': ['x'] * 3 + ['y'] * 2})
        _, axs = plot_value_counts(
            data, color_dict={'x': '#ff0000'}, generic_color='#0000ff'
        )
        bars = axs[0, 0].patches
        assert matplotlib.colors.to_hex(bars[0].get_facecolor()) == '#ff0000'
        assert matplotlib.colors.to_hex(bars[1].get_facecolor()) == '#0000ff'

    def test_color_dict_keys_match_original_label_type(self):
        # colors are matched before labels are converted to strings
        data = pd.DataFrame({'n': [1, 1, 2]})
        _, axs = plot_value_counts(
            data, color_dict={1: '#ff0000'}, generic_color='#0000ff'
        )
        bars = axs[0, 0].patches
        assert matplotlib.colors.to_hex(bars[0].get_facecolor()) == '#ff0000'
        assert matplotlib.colors.to_hex(bars[1].get_facecolor()) == '#0000ff'

    