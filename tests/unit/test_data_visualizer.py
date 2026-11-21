import inspect

import matplotlib
matplotlib.use('Agg')

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_visualizer import (
    defaults_ax_title,
    defaults_fig_title,
    create_fig,
    fig_suptitle_format,
    draw_scatter,
    draw_line,
    make_color_zone,
    _get_number_of_rows,
    _sort_labels_alphabetically,
    _group_overflow,
    _group_overflow_and_sort,
    _sort_colors,
    _hide_unused_axes,
    _get_common_columns,
    _titled_series,
    _paired_series,
    _value_counts_panels,
    _group_sum_panels,
    _get_value_counts_and_sort_labels,
    _get_group_by_and_sort_labels,
    _validation_batch_plotters,
    _validation_batch_groupers,
    _validation_batch_comparers,
    _title_format,
    _visual_setup,
    _draw_pie,
    _draw_barh,
    _draw_legend,
    _annotate_axes,
    _plot_grid,
    _plot_category_grid,
    _plot_hist_grid,
    plot_value_counts_pie,
    plot_value_counts,
    plot_hist,
    plot_alternating_hist,
    plot_group_variable_sum_into_pie,
    plot_group_variable_sum_into_barh,
    plot_alternating_value_counts_barh,
    plot_comparing_box,
    plot_line_chart,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')  # avoids leaking figures between tests


# Helpers to read what was drawn. Pies and bars expose labels and sizes differently

def _pie_labels(ax):
    return [t.get_text() for t in ax.texts if not t.get_text().endswith('%')]

def _pie_sizes(ax):
    return [p.theta2 - p.theta1 for p in ax.patches]

def _barh_labels(ax):
    ax.figure.canvas.draw()
    return [t.get_text() for t in ax.get_yticklabels()]

def _barh_sizes(ax):
    return [p.get_width() for p in ax.patches]

def _fractions(sizes):
    """Each size as a share of the total, so pies (angles) and bars (lengths) compare alike."""
    total = sum(sizes)
    return [s / total for s in sizes]

def _assert_fractions(sizes, expected):
    actual = _fractions(sizes)
    assert len(actual) == len(expected)
    assert all(abs(a - e) < 1e-9 for a, e in zip(actual, expected))

def _hex(color):
    return matplotlib.colors.to_hex(color)

def _visible(axs):
    return [ax for ax in axs.flat if ax.get_visible()]


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

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            _get_number_of_rows(3, 0)

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

    def test_rejects_small_max_categories_even_without_grouping(self):
        with pytest.raises(ValueError, match='max_categories'):
            _group_overflow(pd.Series({'A': 1}), 1)

class TestGroupOverflowAndSort:
    """Shared tail of both _get_*_and_sort_labels helpers.

    It now also replaces the missing-category label and orders by value,
    so it must cope with input in any order.
    """

    @staticmethod
    def _run(values, sort_alphabetically=False, max_categories=10,
             other_label='Outros', na_label='Sem dado'):
        return _group_overflow_and_sort(
            values, sort_alphabetically, max_categories, other_label, na_label
        )

    def test_orders_by_descending_value_whatever_the_input_order(self):
        labels, values = self._run(pd.Series({'a': 1, 'b': 3, 'c': 2}))
        assert labels == ('b', 'c', 'a')
        assert values == (3, 2, 1)

    def test_ties_keep_their_original_order(self):
        labels, _ = self._run(pd.Series({'x': 2, 'y': 2, 'z': 2}))
        assert labels == ('x', 'y', 'z')

    def test_ties_keep_their_original_order_with_many_categories(self):
        # small or constant arrays come out stable from any algorithm, so use
        # a larger one with interleaved ties
        counts = [i % 3 for i in range(90)]
        index = [f'c{i:02d}' for i in range(90)]
        labels, _ = self._run(pd.Series(counts, index=index), max_categories=100)
        expected = sorted(index, key=lambda label: -counts[index.index(label)])
        assert list(labels) == expected

    def test_replaces_missing_label(self):
        labels, values = self._run(pd.Series([2, 1], index=['x', np.nan]))
        assert labels == ('x', 'Sem dado')
        assert values == (2, 1)

    def test_custom_na_label(self):
        labels, _ = self._run(pd.Series([2, 1], index=['x', np.nan]), na_label='Missing')
        assert labels == ('x', 'Missing')

    def test_groups_overflow(self):
        labels, values = self._run(pd.Series({'a': 5, 'b': 4, 'c': 1, 'd': 1}), max_categories=3)
        assert labels == ('a', 'b', 'Outros')
        assert values == (5, 4, 2)

    def test_other_goes_last_when_sorted_alphabetically(self):
        series = pd.Series({'z': 5, 'b': 4, 'a': 1, 'c': 1})
        labels, values = self._run(series, sort_alphabetically=True, max_categories=3)
        assert labels == ('b', 'z', 'Outros')
        assert values == (4, 5, 2)

    @pytest.mark.parametrize('sort', [False, True])
    def test_empty_input(self, sort):
        assert self._run(pd.Series([], dtype=float), sort_alphabetically=sort) == ((), ())

    def test_does_not_modify_input(self):
        series = pd.Series([2, 1], index=['x', np.nan])
        before = series.copy()
        self._run(series)
        pd.testing.assert_series_equal(series, before)

    def test_rejects_small_max_categories(self):
        with pytest.raises(ValueError, match='max_categories'):
            self._run(pd.Series({'a': 1}), max_categories=1)

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

class TestValidationBatchGroupers:
    def test_does_not_raise_on_valid(self):
        df = pd.DataFrame({'cat':['a', 'b'], 'val': [10, 20]})
        _validation_batch_groupers(df, target_col='val')

    def test_rejects_duplicate_columns(self):
        df = pd.DataFrame([[1,2,3]], columns=['a', 'b', 'a'])
        with pytest.raises(ValueError, match='duplicate column names'):
            _validation_batch_groupers(df, target_col='b')

    def test_rejects_missing_target_col(self):
        df = pd.DataFrame({'cat': ['a', 'b'], 'val': [10, 20]})
        with pytest.raises(ValueError, match='not found in data'):
            _validation_batch_groupers(df, target_col='missing')

    def test_rejects_non_numeric_target_col(self):
        df = pd.DataFrame({'cat': ['a', 'b'], 'val': ['x', 'y']})
        with pytest.raises(ValueError, match='must be numeric'):
            _validation_batch_groupers(df, target_col='val')

    def test_rejects_insufficient_columns(self):
        df = pd.DataFrame({'val': [10, 20]})
        with pytest.raises(ValueError, match='needs at least one column besides target_col'):
            _validation_batch_groupers(df, target_col='val')

class TestGetCommonColumns:
    def test_returns_shared_columns_in_data1_order(self):
        d1 = pd.DataFrame(columns=['a', 'b', 'c'])
        d2 = pd.DataFrame(columns=['c', 'a', 'z'])
        assert _get_common_columns(d1, d2) == ['a', 'c']

    def test_ignores_columns_present_in_only_one(self):
        d1 = pd.DataFrame(columns=['a', 'only1'])
        d2 = pd.DataFrame(columns=['a', 'only2'])
        assert _get_common_columns(d1, d2) == ['a']

    def test_rejects_no_shared_columns(self):
        with pytest.raises(ValueError, match='no columns in common'):
            _get_common_columns(pd.DataFrame(columns=['a']), pd.DataFrame(columns=['b']))

    @pytest.mark.parametrize('which', [0, 1])
    def test_rejects_duplicate_column_names(self, which):
        dup = pd.DataFrame([[1, 2]], columns=['a', 'a'])
        ok = pd.DataFrame([[1]], columns=['a'])
        frames = [ok, ok]
        frames[which] = dup
        with pytest.raises(ValueError, match='duplicate column names'):
            _get_common_columns(*frames)

class TestValidationBatchComparers:
    D1 = pd.DataFrame({'a': [1, 2], 'only1': [1, 2]})
    D2 = pd.DataFrame({'a': [3, 4], 'only2': [3, 4]})

    def test_returns_the_shared_columns(self):
        assert _validation_batch_comparers(self.D1, self.D2, max_categories=2, ncols=1) == ['a']

    def test_optional_arguments_can_be_omitted(self):
        assert _validation_batch_comparers(self.D1, self.D2) == ['a']

    @pytest.mark.parametrize('which', [0, 1])
    def test_rejects_empty_dataframe(self, which):
        frames = [self.D1, self.D2]
        frames[which] = pd.DataFrame()
        with pytest.raises(ValueError, match='empty'):
            _validation_batch_comparers(*frames)

    def test_rejects_small_max_categories(self):
        with pytest.raises(ValueError, match='max_categories'):
            _validation_batch_comparers(self.D1, self.D2, max_categories=1)

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            _validation_batch_comparers(self.D1, self.D2, ncols=0)

    def test_rejects_no_shared_columns(self):
        with pytest.raises(ValueError, match='no columns in common'):
            _validation_batch_comparers(self.D1, self.D2.rename(columns={'a': 'zzz'}))

@pytest.mark.parametrize('plot', [plot_value_counts_pie, plot_value_counts])
class TestPlotFunctionsValidation:
    """Functions must validate before drawing anything."""

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

class TestPlotHistValidation:
    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match='empty'):
            plot_hist(pd.DataFrame())

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            plot_hist(pd.DataFrame({'a': [1, 2]}), ncols=0)

    def test_no_figure_is_created_when_validation_fails(self):
        with pytest.raises(ValueError):
            plot_hist(pd.DataFrame({'a': [1, 2]}), ncols=0)
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

    @pytest.mark.parametrize('bad', [1, 0])
    def test_rejects_small_max_categories(self, bad):
        # the check lives in _group_overflow now, so it must still surface from here
        with pytest.raises(ValueError, match='max_categories'):
            _get_value_counts_and_sort_labels(pd.Series(list('abc')), max_categories=bad)

class TestGetGroupByAndSortLabels:
    def test_sums_correctly_per_category(self):
        df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B', 'C'], 'val': [10, 5, 20, 10, 50]})
        labels, sums = _get_group_by_and_sort_labels(df, group_by='cat', target_col='val')
        assert labels == ('C', 'B', 'A')
        assert sums == (50, 30, 15)

    def test_group_by_overflow_and_sorting(self):
        df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D'], 'val': [10, 40, 20, 30]})
        labels, sums = _get_group_by_and_sort_labels(
            df, group_by='cat', target_col='val', max_categories=2
        )
        assert labels == ('B', 'Outros')
        assert sums == (40, 60)


    def test_sort_alphabetically_keeps_sums_paired(self):
        df = pd.DataFrame({'cat': ['b', 'a', 'c'], 'val': [3, 2, 1]})
        labels, sums = _get_group_by_and_sort_labels(
            df, group_by='cat', target_col='val', sort_alphabetically=True
        )
        assert labels == ('a', 'b', 'c')
        assert sums == (2, 3, 1)

    def test_other_goes_last_when_sorted_alphabetically(self):
        df = pd.DataFrame({'cat': list('zbac'), 'val': [5, 4, 1, 1]})
        labels, sums = _get_group_by_and_sort_labels(
            df, group_by='cat', target_col='val', max_categories=3, sort_alphabetically=True
        )
        assert labels == ('b', 'z', 'Outros')
        assert sums == (4, 5, 2)

    def test_missing_category_shown_with_default_label(self):
        df = pd.DataFrame({'cat': ['x', None], 'val': [5, 2]})
        labels, sums = _get_group_by_and_sort_labels(df, group_by='cat', target_col='val')
        assert labels == ('x', 'Sem dado')
        assert sums == (5, 2)

    def test_custom_na_label(self):
        df = pd.DataFrame({'cat': ['x', None], 'val': [5, 2]})
        labels, _ = _get_group_by_and_sort_labels(
            df, group_by='cat', target_col='val', na_label='Missing'
        )
        assert labels == ('x', 'Missing')

    def test_dropna_removes_missing_category(self):
        df = pd.DataFrame({'cat': ['x', None], 'val': [5, 2]})
        labels, sums = _get_group_by_and_sort_labels(
            df, group_by='cat', target_col='val', dropna=True
        )
        assert labels == ('x',)
        assert sums == (5,)

    def test_ignores_unobserved_categorical_levels(self):
        cat = pd.Categorical(['a', 'b'], categories=['a', 'b', 'unused'])
        df = pd.DataFrame({'cat': cat, 'val': [1, 2]})
        labels, _ = _get_group_by_and_sort_labels(df, group_by='cat', target_col='val')
        assert 'unused' not in labels

    def test_missing_target_values_are_skipped(self):
        df = pd.DataFrame({'cat': ['a', 'a', 'b'], 'val': [1, np.nan, 3]})
        labels, sums = _get_group_by_and_sort_labels(df, group_by='cat', target_col='val')
        assert labels == ('b', 'a')
        assert sums == (3, 1)

    @pytest.mark.parametrize('bad', [1, 0])
    def test_rejects_small_max_categories(self, bad):
        df = pd.DataFrame({'cat': list('abc'), 'val': [1, 2, 3]})
        with pytest.raises(ValueError, match='max_categories'):
            _get_group_by_and_sort_labels(df, 'cat', 'val', max_categories=bad)


class TestTitledSeries:
    def test_pairs_each_column_with_its_title(self):
        data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = _titled_series(data, 'By ')
        assert [title for _, title in result] == ['By a', 'By b']
        assert [series.name for series, _ in result] == ['a', 'b']

    def test_duplicate_column_names_yield_one_series_each(self):
        data = pd.DataFrame([[1, 2]], columns=['c', 'c'])
        result = _titled_series(data, '')
        assert len(result) == 2
        assert [series.iloc[0] for series, _ in result] == [1, 2]

class TestPairedSeries:
    D1 = pd.DataFrame({'a': [1], 'b': [2]})
    D2 = pd.DataFrame({'b': [20], 'a': [10]})

    def test_alternates_samples_for_each_column(self):
        result = _paired_series(['a', 'b'], self.D1, self.D2, 'S1', 'S2', 'P ')
        assert [title for _, title in result] == [
            'P a - S1', 'P a - S2', 'P b - S1', 'P b - S2'
        ]

    def test_matches_columns_by_name_not_position(self):
        result = _paired_series(['a', 'b'], self.D1, self.D2, 'S1', 'S2', '')
        assert [series.iloc[0] for series, _ in result] == [1, 10, 2, 20]

    def test_follows_the_given_column_order(self):
        result = _paired_series(['b'], self.D1, self.D2, 'S1', 'S2', '')
        assert [title for _, title in result] == ['b - S1', 'b - S2']

class TestPanelBuilders:
    def test_value_counts_panels_return_labels_counts_and_title(self):
        panels = _value_counts_panels([(pd.Series(list('aab')), 'T1'), (pd.Series(list('xyy')), 'T2')])
        assert panels == [
            (('a', 'b'), (2, 1), 'T1'),
            (('y', 'x'), (2, 1), 'T2'),
        ]

    def test_value_counts_panels_forward_options(self):
        panels = _value_counts_panels(
            [(pd.Series(list('aaabbc')), 'T')], max_categories=2, other_label='Misc'
        )
        assert panels == [(('a', 'Misc'), (3, 3), 'T')]

    def test_value_counts_panels_reject_small_max_categories(self):
        with pytest.raises(ValueError, match='max_categories'):
            _value_counts_panels([(pd.Series(list('ab')), 'T')], max_categories=1)

    def test_group_sum_panels_skip_the_target_column(self):
        data = pd.DataFrame({'k': list('aab'), 'j': list('xyy'), 'v': [1, 2, 4]})
        panels = _group_sum_panels(data, 'v', 'By ')
        assert [title for _, _, title in panels] == ['By k', 'By j']
        assert panels[0][:2] == (('b', 'a'), (4, 3))
        assert panels[1][:2] == (('y', 'x'), (6, 1))

    def test_group_sum_panels_forward_options(self):
        data = pd.DataFrame({'k': list('abcd'), 'v': [4, 3, 2, 1]})
        panels = _group_sum_panels(data, 'v', '', max_categories=2, other_label='Misc')
        assert panels == [(('a', 'Misc'), (4, 6), 'k')]


class TestPlotGrid:
    def test_draws_each_item_once_in_order(self):
        calls = []
        _, axs = _plot_grid(
            items=['a', 'b', 'c'],
            draw=lambda ax, item: calls.append((ax, item)),
            ncols=2,
            figsize=(4, 4)
        )
        assert [item for _, item in calls] == ['a', 'b', 'c']
        assert [ax for ax, _ in calls] == list(axs.flatten()[:3])   # row-major

    def test_returns_fig_and_2d_axes(self):
        fig, axs = _plot_grid([1, 2, 3], lambda ax, item: None, ncols=2, figsize=(4, 4))
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (2, 2)

    def test_hides_unused_axes(self):
        _, axs = _plot_grid([1, 2, 3], lambda ax, item: None, ncols=2, figsize=(4, 4))
        assert [ax.get_visible() for ax in axs.flat] == [True, True, True, False]

    def test_accepts_a_generator(self):
        calls = []
        _, axs = _plot_grid((i for i in range(3)), lambda ax, item: calls.append(item), ncols=3, figsize=(4, 4))
        assert calls == [0, 1, 2]
        assert axs.shape == (1, 3)

    def test_sets_figure_title_and_style(self):
        fig, _ = _plot_grid(
            [1], lambda ax, item: None, ncols=1, figsize=(4, 4),
            fig_title='Hello', suptitle_kwargs={'fontsize': 9}
        )
        assert fig._suptitle.get_text() == 'Hello'
        assert fig._suptitle.get_fontsize() == 9

    def test_rejects_small_ncols_without_creating_a_figure(self):
        with pytest.raises(ValueError, match='ncols'):
            _plot_grid([1], lambda ax, item: None, ncols=0, figsize=(4, 4))
        assert plt.get_fignums() == []

class TestPlotCategoryGrid:
    PANELS = [(('a', 'b'), (3, 1), 'First'), (('x', 'y', 'z'), (1, 1, 2), 'Second')]

    def test_pie_draws_one_wedge_per_label(self):
        _, axs = _plot_category_grid(self.PANELS, 'pie', ncols=2, figsize=(6, 3))
        assert [len(ax.patches) for ax in axs.flat] == [2, 3]
        assert [ax.get_title() for ax in axs.flat] == ['First', 'Second']

    def test_barh_draws_one_bar_per_label(self):
        _, axs = _plot_category_grid(self.PANELS, 'barh', ncols=2, figsize=(6, 3), xlabel='N')
        assert [_barh_sizes(ax) for ax in axs.flat] == [[3, 1], [1, 1, 2]]
        assert [ax.get_xlabel() for ax in axs.flat] == ['N', 'N']

    def test_rejects_unknown_chart_before_creating_a_figure(self):
        with pytest.raises(ValueError, match='chart'):
            _plot_category_grid(self.PANELS, 'donut', ncols=2, figsize=(6, 3))
        assert plt.get_fignums() == []

    def test_default_colors_follow_matplotlib_cycle(self):
        _, axs = _plot_category_grid(self.PANELS, 'pie', ncols=2, figsize=(6, 3))
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        assert [_hex(p.get_facecolor()) for p in axs[0, 0].patches] == [_hex(cycle[0]), _hex(cycle[1])]

    def test_generic_colors_are_used_and_cycled(self):
        _, axs = _plot_category_grid(
            self.PANELS, 'barh', ncols=2, figsize=(6, 3), generic_colors=['#ff0000']
        )
        assert all(_hex(p.get_facecolor()) == '#ff0000' for p in axs[0, 1].patches)

    def test_color_dict_takes_priority(self):
        _, axs = _plot_category_grid(
            self.PANELS, 'barh', ncols=2, figsize=(6, 3),
            color_dict={'a': '#00ff00'}, generic_colors=['#ff0000']
        )
        colors = [_hex(p.get_facecolor()) for p in axs[0, 0].patches]
        assert colors == ['#00ff00', '#ff0000']

    def test_chart_kwargs_reach_the_chart(self):
        _, axs = _plot_category_grid(
            self.PANELS, 'barh', ncols=2, figsize=(6, 3), chart_kwargs={'height': 0.3}
        )
        assert axs[0, 0].patches[0].get_height() == 0.3

class TestPlotHistGrid:
    PANELS = [(pd.Series([1, 2, 3, 4]), 'First'), (pd.Series([10, 20]), 'Second')]

    def test_draws_one_histogram_per_panel(self):
        _, axs = _plot_hist_grid(self.PANELS, 'red', ncols=2, figsize=(6, 3))
        assert [ax.get_title() for ax in axs.flat] == ['First', 'Second']
        assert all(_hex(p.get_facecolor()) == '#ff0000' for p in axs[0, 0].patches)

    def test_hist_kwargs_reach_the_histogram(self):
        _, axs = _plot_hist_grid(self.PANELS, 'red', ncols=2, figsize=(6, 3), hist_kwargs={'bins': 3})
        assert len(axs[0, 0].patches) == 3

    def test_hides_unused_axes(self):
        _, axs = _plot_hist_grid(self.PANELS, 'red', ncols=3, figsize=(6, 3))
        assert [ax.get_visible() for ax in axs.flat] == [True, True, False]


class TestCreateFigAndHideUnusedAxes:
    def test_axes_are_always_2d(self):
        _, axs = create_fig(1, 1, (4, 4))
        assert axs.shape == (1, 1)

    def test_grid_is_large_enough(self):
        _, axs = create_fig(5, 2, (4, 4))
        assert axs.shape == (3, 2)

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            create_fig(3, 0, (4, 4))

    def test_hide_unused_axes_hides_only_the_tail(self):
        _, axs = create_fig(5, 2, (4, 4))
        _hide_unused_axes(axs, 3)
        assert [ax.get_visible() for ax in axs.flat] == [True] * 3 + [False] * 3


class TestTitleFormatting:
    def test_axes_title_uses_defaults(self):
        _, ax = plt.subplots()
        _title_format(ax, 'T')
        assert ax.get_title() == 'T'
        assert ax.title.get_fontsize() == defaults_ax_title['fontsize']
        assert ax.title.get_fontweight() == 'bold'

    def test_axes_title_overrides(self):
        _, ax = plt.subplots()
        _title_format(ax, 'T', {'fontsize': 20})
        assert ax.title.get_fontsize() == 20

    def test_figure_title_uses_defaults_and_overrides(self):
        fig = plt.figure()
        fig_suptitle_format(fig, 'T')
        assert fig._suptitle.get_fontsize() == defaults_fig_title['fontsize']
        fig_suptitle_format(fig, 'T', {'fontsize': 9})
        assert fig._suptitle.get_fontsize() == 9

    def test_overrides_do_not_leak_into_the_defaults(self):
        _, ax = plt.subplots()
        _title_format(ax, 'T', {'fontsize': 99})
        fig_suptitle_format(ax.figure, 'T', {'fontsize': 99})
        assert defaults_ax_title['fontsize'] == 10
        assert defaults_fig_title['fontsize'] == 14

    def test_label_in_overrides_does_not_collide(self):
        # the title text is part of the merged dict, so this must not raise TypeError
        _, ax = plt.subplots()
        _title_format(ax, 'T', {'label': 'other'})

class TestVisualSetup:
    def test_sets_background_and_hides_top_right_spines(self):
        _, ax = plt.subplots()
        _visual_setup(ax, facecolor='#123456')
        assert _hex(ax.get_facecolor()) == '#123456'
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()

class TestDrawHelpersKwargs:
    def test_pie_kwargs_may_override_labels(self):
        _, ax = plt.subplots()
        _draw_pie(ax, [1, 2], ['a', 'b'], 't', ['r', 'g'], pie_kwargs={'labels': ['X', 'Y']})
        assert _pie_labels(ax) == ['X', 'Y']

    def test_barh_kwargs_are_applied(self):
        _, ax = plt.subplots()
        _draw_barh(ax, ['a', 'b'], [1, 2], 't', 'x', ['r'], barh_kwargs={'height': 0.2})
        assert ax.patches[0].get_height() == 0.2

    def test_barh_default_height_is_unchanged(self):
        _, ax = plt.subplots()
        _draw_barh(ax, ['a', 'b'], [1, 2], 't', 'x', ['r'])
        assert ax.patches[0].get_height() == 0.8


class TestLegendAndAxisHelpers:
    def test_draw_legend_accepts_overrides(self):
        _, ax = plt.subplots()
        ax.plot([1, 2], label='l')
        _draw_legend(ax, {'title': 'LT'})
        assert ax.get_legend().get_title().get_text() == 'LT'

    def test_annotate_axes_sets_title_and_labels(self):
        _, ax = plt.subplots()
        _annotate_axes(ax, 'T', 'X', 'Y')
        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ('T', 'X', 'Y')

    def test_annotate_axes_has_no_legend_without_label(self):
        _, ax = plt.subplots()
        ax.plot([1, 2], label='l')
        _annotate_axes(ax, 'T', 'X', 'Y')
        assert ax.get_legend() is None

    def test_annotate_axes_draws_legend_with_label(self):
        _, ax = plt.subplots()
        ax.plot([1, 2], label='l')
        _annotate_axes(ax, 'T', 'X', 'Y', label='l', legend_kwargs={'title': 'LT'})
        assert ax.get_legend().get_title().get_text() == 'LT'

class TestDrawScatter:
    def test_sets_title_and_axis_labels(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2, 3], [3, 1, 2], 'X', 'Y', 'T', 'red')
        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ('T', 'X', 'Y')

    def test_axis_labels_are_set_without_a_legend(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2], [1, 2], 'X', 'Y', 'T', 'red')
        assert ax.get_legend() is None
        assert ax.get_xlabel() == 'X'

    def test_draws_legend_with_label(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2], [1, 2], 'X', 'Y', 'T', 'red', label='pts')
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['pts']

    def test_uses_the_given_color(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2], [1, 2], 'X', 'Y', 'T', 'red')
        assert _hex(ax.collections[0].get_facecolor()[0]) == '#ff0000'

    def test_scatter_kwargs_can_color_by_value(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2, 3], [1, 2, 3], 'X', 'Y', 'T', 'red',
                     scatter_kwargs={'c': [1, 2, 3], 'color': None})
        assert len(ax.collections) == 1

    def test_setup_and_title_kwargs(self):
        _, ax = plt.subplots()
        draw_scatter(ax, [1, 2], [1, 2], 'X', 'Y', 'T', 'red',
                     title_kwargs={'fontsize': 21}, setup_kwargs={'facecolor': '#123456'})
        assert ax.title.get_fontsize() == 21
        assert _hex(ax.get_facecolor()) == '#123456'

class TestDrawLine:
    def test_horizontal_line_at_coord(self):
        _, ax = plt.subplots()
        draw_line(ax, 2, 'h')
        assert list(ax.lines[0].get_ydata()) == [2, 2]

    def test_vertical_line_at_coord(self):
        _, ax = plt.subplots()
        draw_line(ax, 2, 'v')
        assert list(ax.lines[0].get_xdata()) == [2, 2]

    def test_unknown_linetype_draws_nothing(self):
        _, ax = plt.subplots()
        draw_line(ax, 2, 'x')
        assert len(ax.lines) == 0

    def test_legend_needs_both_show_label_and_a_label(self):
        _, ax = plt.subplots()
        draw_line(ax, 1, label='ref')                    # show_label False
        assert ax.get_legend() is None
        draw_line(ax, 2, show_label=True)                # no label
        assert ax.get_legend() is None
        draw_line(ax, 3, label='ref', show_label=True)
        assert ax.get_legend() is not None

class TestMakeColorZone:
    def test_adds_a_filled_area(self):
        _, ax = plt.subplots()
        make_color_zone(ax, [0, 1], [0, 0], [1, 1], 'red', {'alpha': 0.5})
        assert len(ax.collections) == 1
        assert ax.collections[0].get_alpha() == 0.5


class TestAdditionalPlotFunctions:
    def test_plot_hist(self):
        df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]})
        fig, axs = plot_hist(df)
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 3)

    def test_plot_alternating_hist(self):
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [4, 5, 6]})
        fig, axs = plot_alternating_hist(df1, df2)
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 2)

    def test_plot_group_variable_sum_into_pie(self):
        df = pd.DataFrame({'cat': ['A', 'B', 'A', 'B'], 'val': [10, 20, 30, 40]})
        fig, axs = plot_group_variable_sum_into_pie(df, target_col='val')
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 3)

    def test_plot_group_variable_sum_into_barh(self):
        df = pd.DataFrame({'cat': ['A', 'B', 'A', 'B'], 'val': [10, 20, 30, 40]})
        fig, axs = plot_group_variable_sum_into_barh(df, target_col='val')
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 3)

    def test_plot_alternating_value_counts_barh(self):
        df1 = pd.DataFrame({'cat': ['A', 'B', 'A']})
        df2 = pd.DataFrame({'cat': ['B', 'B', 'C']})
        fig, axs = plot_alternating_value_counts_barh(df1, df2)
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 2)

    def test_plot_comparing_box(self):
        df1 = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({'val': [2, 3, 4, 5, 6]})
        fig, axs = plot_comparing_box(df1, df2)
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 2)

    def test_plot_line_chart_axes_provided(self):
        fig_base, ax_base = plt.subplots()
        x = np.linspace(0, 10, 5)
        y = np.sin(x)
        ax = plot_line_chart(x=x, y=y, xlabel='X', ylabel='Y', label='Sine', color='red', ax=ax_base)
        assert ax == ax_base

    def test_plot_line_chart_new_figure(self):
        x = np.linspace(0, 10, 5)
        y = np.sin(x)
        fig, axs = plot_line_chart(x=x, y=y, xlabel='X', ylabel='Y', label='Sine', color='red')
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, np.ndarray)


class TestPlotLineChart:
    X = np.arange(5)

    def _plot(self, **kwargs):
        args = dict(x=self.X, y=self.X ** 2, xlabel='X', ylabel='Y', label='sq', color='red')
        args.update(kwargs)
        return plot_line_chart(**args)

    def test_new_figure_has_a_single_2d_subplot(self):
        fig, axs = self._plot()
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 1)

    def test_default_title_is_ylabel_por_xlabel(self):
        _, axs = self._plot()
        assert axs[0, 0].get_title() == 'Y por X'

    def test_default_title_on_a_given_axes(self):
        _, ax = plt.subplots()
        assert self._plot(ax=ax).get_title() == 'Y por X'

    def test_custom_title(self):
        _, axs = self._plot(title='Mine')
        assert axs[0, 0].get_title() == 'Mine'

    def test_labels_and_legend_in_a_new_figure(self):
        _, axs = self._plot()
        ax = axs[0, 0]
        assert (ax.get_xlabel(), ax.get_ylabel()) == ('X', 'Y')
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['sq']

    def test_labels_and_legend_on_a_given_axes(self):
        _, ax = plt.subplots()
        self._plot(ax=ax)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ('X', 'Y')
        assert ax.get_legend() is not None

    def test_line_kwargs_and_legend_kwargs_apply_in_a_new_figure(self):
        _, axs = self._plot(line_kwargs={'lw': 3}, legend_kwargs={'title': 'LT'})
        ax = axs[0, 0]
        assert ax.lines[0].get_linewidth() == 3
        assert ax.get_legend().get_title().get_text() == 'LT'

    def test_default_line_width(self):
        _, axs = self._plot()
        assert axs[0, 0].lines[0].get_linewidth() == 0.75

    def test_figure_title_figsize_and_style(self):
        fig, _ = self._plot(fig_title='Hello', figsize=(8, 3), suptitle_kwargs={'fontsize': 9})
        assert fig._suptitle.get_text() == 'Hello'
        assert fig._suptitle.get_fontsize() == 9
        assert tuple(fig.get_size_inches()) == (8, 3)

    def test_suptitle_kwargs_none_is_accepted(self):
        self._plot(fig_title='Hello', suptitle_kwargs=None)

    def test_given_axes_is_returned_and_no_figure_is_created(self):
        _, ax = plt.subplots()
        before = plt.get_fignums()
        assert self._plot(ax=ax, fig_title='ignored') is ax
        assert plt.get_fignums() == before
        assert ax.figure._suptitle is None

    def test_setup_and_title_kwargs(self):
        _, axs = self._plot(title_kwargs={'fontsize': 21}, setup_kwargs={'facecolor': '#123456'})
        assert axs[0, 0].title.get_fontsize() == 21
        assert _hex(axs[0, 0].get_facecolor()) == '#123456'


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
    _get_labels = staticmethod(_pie_labels)
    _get_sizes = staticmethod(_pie_sizes)

    # colors

    def test_applies_color_dict(self):
        data = pd.DataFrame({'a': ['x'] * 3 + ['y'] * 2})
        _, axs = plot_value_counts_pie(data, color_dict={'x': '#ff0000'})
        first_wedge = axs[0, 0].patches[0]
        assert _hex(first_wedge.get_facecolor()) == '#ff0000'

    def test_generic_colors_are_cycled(self):
        data = pd.DataFrame({'a': list('xxyyz')})
        _, axs = plot_value_counts_pie(data, generic_colors=['#ff0000', '#00ff00'])
        colors = [_hex(p.get_facecolor()) for p in axs[0, 0].patches]
        assert colors == ['#ff0000', '#00ff00', '#ff0000']

    def test_pie_kwargs_are_applied(self):
        data = pd.DataFrame({'a': list('xxyz')})
        _, axs = plot_value_counts_pie(data, pie_kwargs={'autopct': '%d%%'})
        assert '50%' in [t.get_text() for t in axs[0, 0].texts]

class TestPlotValueCounts(_SharedPlotTests):
    _plot = staticmethod(plot_value_counts)
    _get_labels = staticmethod(_barh_labels)
    _get_sizes = staticmethod(_barh_sizes)

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

    def test_barh_kwargs_are_applied(self, df):
        _, axs = plot_value_counts(df, barh_kwargs={'height': 0.3})
        assert axs[0, 0].patches[0].get_height() == 0.3

    # colors

    def test_uses_generic_color_by_default(self, df):
        _, axs = plot_value_counts(df)
        for bar in axs[0, 0].patches:
            assert _hex(bar.get_facecolor()) == '#0000ff'

    def test_custom_generic_color(self, df):
        _, axs = plot_value_counts(df, generic_color='#00ff00')
        for bar in axs[0, 0].patches:
            assert _hex(bar.get_facecolor()) == '#00ff00'

    def test_color_dict_overrides_generic_color(self):
        data = pd.DataFrame({'a': ['x'] * 3 + ['y'] * 2})
        _, axs = plot_value_counts(
            data, color_dict={'x': '#ff0000'}, generic_color='#0000ff'
        )
        bars = axs[0, 0].patches
        assert _hex(bars[0].get_facecolor()) == '#ff0000'
        assert _hex(bars[1].get_facecolor()) == '#0000ff'

    def test_color_dict_keys_match_original_label_type(self):
        # colors are matched before labels are converted to strings
        data = pd.DataFrame({'n': [1, 1, 2]})
        _, axs = plot_value_counts(
            data, color_dict={1: '#ff0000'}, generic_color='#0000ff'
        )
        bars = axs[0, 0].patches
        assert _hex(bars[0].get_facecolor()) == '#ff0000'
        assert _hex(bars[1].get_facecolor()) == '#0000ff'


class _SharedGroupedPlotTests:
    """Behavior common to plot_group_variable_sum_into_pie and _into_barh.

    They mirror the value-count plots: same layout, titles, ordering,
    grouping and missing-value handling, but sizes are sums of a target column.
    """

    @staticmethod
    def _plot(*args, **kwargs):
        raise NotImplementedError('_plot not implemented')

    @staticmethod
    def _get_labels(ax):
        raise NotImplementedError('_get_labels not implemented')

    @staticmethod
    def _get_sizes(ax):
        raise NotImplementedError('_get_sizes not implemented')

    @pytest.fixture
    def df(self):
        # color sums: red=10, blue=3, green=2 / size sums: S=10, M=5 / total 15
        return pd.DataFrame({
            'color': ['red', 'red', 'blue', 'green'],
            'size':  ['S', 'M', 'S', 'S'],
            'qty':   [5, 5, 3, 2],
        })

    # figure structure

    def test_returns_fig_and_2d_axes(self, df):
        fig, axs = self._plot(df, target_col='qty')
        assert isinstance(fig, plt.Figure)
        assert axs.shape == (1, 3)

    def test_target_column_is_not_plotted(self, df):
        _, axs = self._plot(df, target_col='qty')
        titles = [ax.get_title() for ax in _visible(axs)]
        assert titles == ['Agrupamento por color', 'Agrupamento por size']

    def test_hides_unused_axes(self, df):
        _, axs = self._plot(df, target_col='qty', ncols=3)
        assert [ax.get_visible() for ax in axs.flat] == [True, True, False]

    def test_row_count(self):
        data = pd.DataFrame({f'c{i}': ['a', 'b'] for i in range(5)})
        data['v'] = [1, 2]
        _, axs = self._plot(data, target_col='v', ncols=2)
        assert axs.shape == (3, 2)

    # titles

    def test_titles_use_prefix_and_column_name(self, df):
        _, axs = self._plot(df, target_col='qty', title_prefix='By ')
        assert axs[0, 0].get_title() == 'By color'
        assert axs[0, 1].get_title() == 'By size'

    def test_sets_figure_title(self, df):
        fig, _ = self._plot(df, target_col='qty', fig_title='Hello')
        assert fig._suptitle.get_text() == 'Hello'

    # ordering and sizes

    def test_ordered_by_sum_by_default(self, df):
        _, axs = self._plot(df, target_col='qty')
        assert self._get_labels(axs[0, 0]) == ['red', 'blue', 'green']
        _assert_fractions(self._get_sizes(axs[0, 0]), [10 / 15, 3 / 15, 2 / 15])

    def test_sort_alphabetically_keeps_labels_and_sizes_aligned(self, df):
        _, axs = self._plot(df, target_col='qty', sort_alphabetically=True)
        ax = axs[0, 0]
        assert self._get_labels(ax) == ['blue', 'green', 'red']
        _assert_fractions(self._get_sizes(ax), [3 / 15, 2 / 15, 10 / 15])

    def test_missing_target_values_are_skipped(self):
        data = pd.DataFrame({'k': ['a', 'a', 'b'], 'v': [1, np.nan, 3]})
        _, axs = self._plot(data, target_col='v')
        assert self._get_labels(axs[0, 0]) == ['b', 'a']
        _assert_fractions(self._get_sizes(axs[0, 0]), [0.75, 0.25])

    # grouping

    def test_groups_overflow_into_other(self, df):
        _, axs = self._plot(df, target_col='qty', max_categories=2)
        assert self._get_labels(axs[0, 0]) == ['red', 'Outros']
        _assert_fractions(self._get_sizes(axs[0, 0]), [10 / 15, 5 / 15])

    def test_other_goes_last_when_sorted_alphabetically(self):
        data = pd.DataFrame({'k': list('zbac'), 'v': [5, 4, 1, 1]})
        _, axs = self._plot(data, target_col='v', max_categories=3, sort_alphabetically=True)
        assert self._get_labels(axs[0, 0]) == ['b', 'z', 'Outros']

    def test_custom_other_label(self, df):
        _, axs = self._plot(df, target_col='qty', max_categories=2, other_label='Misc')
        assert 'Misc' in self._get_labels(axs[0, 0])

    # missing / unobserved values

    def test_missing_category_shown_with_label_by_default(self):
        data = pd.DataFrame({'k': ['x', 'x', None], 'v': [1, 2, 3]})
        _, axs = self._plot(data, target_col='v')
        assert 'Sem dado' in self._get_labels(axs[0, 0])

    def test_dropna_removes_missing_category(self):
        data = pd.DataFrame({'k': ['x', 'x', None], 'v': [1, 2, 3]})
        _, axs = self._plot(data, target_col='v', dropna=True)
        assert self._get_labels(axs[0, 0]) == ['x']

    def test_custom_na_label(self):
        data = pd.DataFrame({'k': ['x', None], 'v': [1, 2]})
        _, axs = self._plot(data, target_col='v', na_label='Missing')
        assert 'Missing' in self._get_labels(axs[0, 0])

    def test_ignores_unobserved_categorical_levels(self):
        cat = pd.Categorical(['a', 'b'], categories=['a', 'b', 'unused'])
        _, axs = self._plot(pd.DataFrame({'k': cat, 'v': [1, 2]}), target_col='v')
        assert 'unused' not in self._get_labels(axs[0, 0])

    # column names

    def test_supports_non_string_group_column_names(self):
        data = pd.DataFrame({0: ['a', 'b'], 'v': [1, 2]})
        _, axs = self._plot(data, target_col='v')
        assert axs[0, 0].get_title().endswith('0')

    # validation

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match='empty'):
            self._plot(pd.DataFrame(), target_col='v')

    def test_rejects_small_max_categories(self, df):
        with pytest.raises(ValueError, match='max_categories'):
            self._plot(df, target_col='qty', max_categories=1)

    def test_rejects_small_ncols(self, df):
        with pytest.raises(ValueError, match='ncols'):
            self._plot(df, target_col='qty', ncols=0)

    def test_rejects_duplicate_column_names(self):
        data = pd.DataFrame([[1, 2, 3]], columns=['a', 'v', 'a'])
        with pytest.raises(ValueError, match='duplicate column names'):
            self._plot(data, target_col='v')

    def test_rejects_missing_target_column(self, df):
        with pytest.raises(ValueError, match='not found in data'):
            self._plot(df, target_col='missing')

    def test_rejects_non_numeric_target_column(self, df):
        with pytest.raises(ValueError, match='must be numeric'):
            self._plot(df, target_col='color')

    def test_rejects_target_as_only_column(self):
        with pytest.raises(ValueError, match='needs at least one column'):
            self._plot(pd.DataFrame({'v': [1, 2]}), target_col='v')

    def test_no_figure_is_created_when_validation_fails(self, df):
        with pytest.raises(ValueError):
            self._plot(df, target_col='missing')
        with pytest.raises(ValueError):
            self._plot(df, target_col='qty', ncols=0)
        assert plt.get_fignums() == []

class TestPlotGroupVariableSumIntoPie(_SharedGroupedPlotTests):
    _plot = staticmethod(plot_group_variable_sum_into_pie)
    _get_labels = staticmethod(_pie_labels)
    _get_sizes = staticmethod(_pie_sizes)

    def test_applies_color_dict(self, df):
        _, axs = self._plot(df, target_col='qty', color_dict={'red': '#ff0000'})
        assert _hex(axs[0, 0].patches[0].get_facecolor()) == '#ff0000'

    def test_generic_colors_are_cycled(self, df):
        _, axs = self._plot(df, target_col='qty', generic_colors=['#ff0000', '#00ff00'])
        colors = [_hex(p.get_facecolor()) for p in axs[0, 0].patches]
        assert colors == ['#ff0000', '#00ff00', '#ff0000']

    def test_pie_kwargs_are_applied(self, df):
        _, axs = self._plot(df, target_col='qty', pie_kwargs={'autopct': '%d%%'})
        assert '66%' in [t.get_text() for t in axs[0, 0].texts]   # red: 10 of 15

    def test_negative_sum_is_rejected_by_matplotlib(self):
        data = pd.DataFrame({'k': ['a', 'b'], 'v': [-1, 2]})
        with pytest.raises(ValueError):
            self._plot(data, target_col='v')

class TestPlotGroupVariableSumIntoBarh(_SharedGroupedPlotTests):
    _plot = staticmethod(plot_group_variable_sum_into_barh)
    _get_labels = staticmethod(_barh_labels)
    _get_sizes = staticmethod(_barh_sizes)

    def test_first_category_is_on_top(self, df):
        _, axs = self._plot(df, target_col='qty')
        assert axs[0, 0].yaxis_inverted()

    def test_bar_lengths_are_the_sums(self, df):
        _, axs = self._plot(df, target_col='qty')
        assert self._get_sizes(axs[0, 0]) == [10, 3, 2]

    def test_sets_xlabel(self, df):
        _, axs = self._plot(df, target_col='qty', xlabel='Soma')
        assert axs[0, 0].get_xlabel() == 'Soma'
        assert axs[0, 1].get_xlabel() == 'Soma'

    def test_barh_kwargs_are_applied(self, df):
        _, axs = self._plot(df, target_col='qty', barh_kwargs={'height': 0.3})
        assert axs[0, 0].patches[0].get_height() == 0.3

    def test_uses_generic_color_by_default(self, df):
        _, axs = self._plot(df, target_col='qty')
        assert all(_hex(b.get_facecolor()) == '#0000ff' for b in axs[0, 0].patches)

    def test_custom_generic_color(self, df):
        _, axs = self._plot(df, target_col='qty', generic_color='#00ff00')
        assert all(_hex(b.get_facecolor()) == '#00ff00' for b in axs[0, 0].patches)

    def test_color_dict_overrides_generic_color(self, df):
        _, axs = self._plot(df, target_col='qty', color_dict={'red': '#ff0000'})
        colors = [_hex(b.get_facecolor()) for b in axs[0, 0].patches]
        assert colors == ['#ff0000', '#0000ff', '#0000ff']


class _SharedTwoSampleTests:
    """Behavior common to the three two-sample functions.

    Columns are matched by name (in data1's order); columns present in only
    one DataFrame are skipped. Subclasses define `_plot`, `_frames()`
    (two valid DataFrames sharing columns 'a' and 'b', with 'b' listed first
    in data2, and one column that exists in only one of them) and
    `_panels_per_column`.
    """

    _panels_per_column = 0

    @staticmethod
    def _plot(*args, **kwargs):
        raise NotImplementedError('_plot not implemented')

    @staticmethod
    def _frames():
        raise NotImplementedError('_frames not implemented')

    # layout

    def test_default_ncols_is_two(self):
        # keeps the pair of subplots of a column on the same row
        assert inspect.signature(self._plot).parameters['ncols'].default == 2

    def test_returns_fig_and_2d_axes(self):
        fig, axs = self._plot(*self._frames())
        assert isinstance(fig, plt.Figure)
        assert axs.ndim == 2

    def test_columns_present_in_only_one_frame_are_skipped(self):
        _, axs = self._plot(*self._frames())
        assert len(_visible(axs)) == 2 * self._panels_per_column
        assert not any('only' in ax.get_title() for ax in axs.flat)

    def test_hides_unused_axes(self):
        d1, d2 = self._frames()
        _, axs = self._plot(d1[['a']], d2[['a']], ncols=3)
        assert len(_visible(axs)) == self._panels_per_column
        assert axs.size > self._panels_per_column

    def test_sets_figure_title(self):
        fig, _ = self._plot(*self._frames(), fig_title='Hello')
        assert fig._suptitle.get_text() == 'Hello'

    # validation

    @pytest.mark.parametrize('which', [0, 1])
    def test_rejects_empty_dataframe(self, which):
        frames = list(self._frames())
        frames[which] = pd.DataFrame()
        with pytest.raises(ValueError, match='empty'):
            self._plot(*frames)

    def test_rejects_small_ncols(self):
        with pytest.raises(ValueError, match='ncols'):
            self._plot(*self._frames(), ncols=0)

    def test_rejects_frames_without_shared_columns(self):
        d1, d2 = self._frames()
        with pytest.raises(ValueError, match='no columns in common'):
            self._plot(d1[['only1']], d2[['only2']])

    @pytest.mark.parametrize('which', [0, 1])
    def test_rejects_duplicate_column_names(self, which):
        frames = list(self._frames())
        frames[which] = frames[which][['a', 'a']]
        with pytest.raises(ValueError, match='duplicate column names'):
            self._plot(*frames)

    def test_no_figure_is_created_when_validation_fails(self):
        d1, d2 = self._frames()
        with pytest.raises(ValueError):
            self._plot(d1, d2, ncols=0)
        with pytest.raises(ValueError):
            self._plot(d1[['only1']], d2[['only2']])
        assert plt.get_fignums() == []

class TestPlotAlternatingHist(_SharedTwoSampleTests):
    _plot = staticmethod(plot_alternating_hist)
    _panels_per_column = 2

    @staticmethod
    def _frames():
        d1 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [101, 102, 103, 104], 'only1': [0, 0, 0, 0]})
        d2 = pd.DataFrame({'b': [101, 102, 103], 'a': [1, 2, 3], 'only2': [0, 0, 0]})
        return d1, d2

    def test_defaults_match_plot_hist(self):
        # the two histogram functions should read the same by default
        alt = inspect.signature(plot_alternating_hist).parameters
        single = inspect.signature(plot_hist).parameters
        assert alt['title_prefix'].default == single['title_prefix'].default

    def test_samples_alternate_for_each_column_in_data1_order(self):
        _, axs = plot_alternating_hist(*self._frames(), data1_label='S1', data2_label='S2', title_prefix='P ')
        assert [ax.get_title() for ax in _visible(axs)] == [
            'P a - S1', 'P a - S2', 'P b - S1', 'P b - S2'
        ]

    def test_default_labels(self):
        _, axs = plot_alternating_hist(*self._frames(), title_prefix='')
        assert axs[0, 0].get_title() == 'a - Sample 1'
        assert axs[0, 1].get_title() == 'a - Sample 2'

    def test_columns_are_matched_by_name_not_position(self):
        # regression: data2 lists 'b' first, so pairing by position mixed the columns up
        _, axs = plot_alternating_hist(*self._frames(), title_prefix='')
        a_sample2, b_sample2 = axs[0, 1], axs[1, 1]
        assert a_sample2.get_xlim()[1] < 50     # values 1..3
        assert b_sample2.get_xlim()[0] > 50     # values 101..103

    def test_hist_kwargs_reach_both_samples(self):
        _, axs = plot_alternating_hist(*self._frames(), hist_kwargs={'bins': 3})
        assert [len(ax.patches) for ax in _visible(axs)] == [3, 3, 3, 3]

    def test_color(self):
        _, axs = plot_alternating_hist(*self._frames(), color='#00ff00')
        assert all(_hex(p.get_facecolor()) == '#00ff00' for p in axs[0, 0].patches)

class TestPlotAlternatingValueCountsBarh(_SharedTwoSampleTests):
    _plot = staticmethod(plot_alternating_value_counts_barh)
    _panels_per_column = 2

    @staticmethod
    def _frames():
        d1 = pd.DataFrame({'a': list('xxy'), 'b': list('pqq'), 'only1': list('zzz')})
        d2 = pd.DataFrame({'b': list('rrs'), 'a': list('xyy'), 'only2': list('zzz')})
        return d1, d2

    def test_samples_alternate_for_each_column_in_data1_order(self):
        _, axs = plot_alternating_value_counts_barh(
            *self._frames(), data1_label='S1', data2_label='S2', title_prefix='P '
        )
        assert [ax.get_title() for ax in _visible(axs)] == [
            'P a - S1', 'P a - S2', 'P b - S1', 'P b - S2'
        ]

    def test_columns_are_matched_by_name_not_position(self):
        # regression: data2 lists 'b' first, so pairing by position mixed the columns up
        _, axs = plot_alternating_value_counts_barh(*self._frames())
        assert _barh_labels(axs[0, 0]) == ['x', 'y']    # a, sample 1
        assert _barh_labels(axs[0, 1]) == ['y', 'x']    # a, sample 2
        assert _barh_labels(axs[1, 0]) == ['q', 'p']    # b, sample 1
        assert _barh_labels(axs[1, 1]) == ['r', 's']    # b, sample 2

    def test_counts_are_drawn(self):
        _, axs = plot_alternating_value_counts_barh(*self._frames())
        assert _barh_sizes(axs[0, 0]) == [2, 1]

    def test_sets_xlabel_everywhere(self):
        _, axs = plot_alternating_value_counts_barh(*self._frames(), xlabel='N')
        assert all(ax.get_xlabel() == 'N' for ax in _visible(axs))

    def test_groups_overflow_into_other(self):
        d = pd.DataFrame({'a': list('aaabbc')})
        _, axs = plot_alternating_value_counts_barh(d, d, max_categories=2)
        assert _barh_labels(axs[0, 0]) == ['a', 'Outros']
        assert _barh_labels(axs[0, 1]) == ['a', 'Outros']

    def test_sort_alphabetically_treats_the_missing_label_as_a_regular_category(self):
        # only the aggregated 'Outros' entry is forced last
        d = pd.DataFrame({'a': ['z', 'z', 'b', None]})
        _, axs = plot_alternating_value_counts_barh(d, d, sort_alphabetically=True)
        assert _barh_labels(axs[0, 0]) == ['b', 'Sem dado', 'z']

    def test_dropna_and_custom_na_label(self):
        d = pd.DataFrame({'a': ['z', 'z', 'b', None]})
        _, axs = plot_alternating_value_counts_barh(d, d, dropna=True)
        assert _barh_labels(axs[0, 0]) == ['z', 'b']
        _, axs = plot_alternating_value_counts_barh(d, d, na_label='Missing')
        assert 'Missing' in _barh_labels(axs[0, 0])

    def test_rejects_small_max_categories(self):
        with pytest.raises(ValueError, match='max_categories'):
            plot_alternating_value_counts_barh(*self._frames(), max_categories=1)

    def test_color_dict_and_generic_color(self):
        _, axs = plot_alternating_value_counts_barh(
            *self._frames(), color_dict={'x': '#ff0000'}, generic_color='#0000ff'
        )
        assert [_hex(p.get_facecolor()) for p in axs[0, 0].patches] == ['#ff0000', '#0000ff']

    def test_barh_kwargs_reach_both_samples(self):
        _, axs = plot_alternating_value_counts_barh(*self._frames(), barh_kwargs={'height': 0.3})
        assert all(ax.patches[0].get_height() == 0.3 for ax in _visible(axs))

class TestPlotComparingBox(_SharedTwoSampleTests):
    _plot = staticmethod(plot_comparing_box)
    _panels_per_column = 1

    @staticmethod
    def _frames():
        d1 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [101, 102, 103, 104], 'only1': [0, 0, 0, 0]})
        d2 = pd.DataFrame({'b': [101, 102, 103], 'a': [1, 2, 3], 'only2': [0, 0, 0]})
        return d1, d2

    def test_one_subplot_per_shared_column_in_data1_order(self):
        _, axs = plot_comparing_box(*self._frames(), title_prefix='P ')
        assert [ax.get_title() for ax in _visible(axs)] == ['P a', 'P b']

    def test_both_samples_share_a_subplot_with_tick_labels(self):
        _, axs = plot_comparing_box(*self._frames(), data1_label='S1', data2_label='S2')
        ax = axs[0, 0]
        ax.figure.canvas.draw()
        assert [t.get_text() for t in ax.get_xticklabels()] == ['S1', 'S2']

    def test_columns_are_matched_by_name_not_position(self):
        # regression: data2 lists 'b' first, so pairing by position mixed the columns up
        _, axs = plot_comparing_box(*self._frames(), title_prefix='')
        assert axs[0, 0].get_ylim()[1] < 50     # column a: values 1..4
        assert axs[0, 1].get_ylim()[0] > 50     # column b: values 101..104

    def test_title_uses_the_shared_column_name(self):
        _, axs = plot_comparing_box(*self._frames(), title_prefix='By ')
        assert axs[0, 0].get_title() == 'By a'

    def test_boxes_are_drawn_side_by_side(self):
        _, axs = plot_comparing_box(*self._frames(), boxplot_kwargs={'patch_artist': True})
        centers = sorted(p.get_path().vertices[:, 0].mean() for p in axs[0, 0].patches)
        assert abs(centers[0] - 0) < 0.1 and abs(centers[1] - 1) < 0.1

    def test_boxplot_kwargs_reach_both_boxes(self):
        _, axs = plot_comparing_box(*self._frames(), boxplot_kwargs={'patch_artist': True})
        assert len(axs[0, 0].patches) == 2

    def test_max_categories_is_accepted_but_has_no_effect(self):
        # kept in the signature so existing calls keep working
        _, axs = plot_comparing_box(*self._frames(), max_categories=1)
        assert len(_visible(axs)) == 2


# The same style arguments must behave the same in every plot_* function.

_CAT = pd.DataFrame({'a': list('xxyz'), 'b': list('pqqq'), 'v': [1, 2, 3, 4]})
_NUM1 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
_NUM2 = pd.DataFrame({'a': [2, 3, 4], 'b': [6, 7, 8]})

_ALL_PLOTS = {
    'value_counts_pie':    lambda **kw: plot_value_counts_pie(_CAT[['a', 'b']], **kw),
    'value_counts':        lambda **kw: plot_value_counts(_CAT[['a', 'b']], **kw),
    'hist':                lambda **kw: plot_hist(_NUM1, **kw),
    'alternating_hist':    lambda **kw: plot_alternating_hist(_NUM1, _NUM2, **kw),
    'group_pie':           lambda **kw: plot_group_variable_sum_into_pie(_CAT, 'v', **kw),
    'group_barh':          lambda **kw: plot_group_variable_sum_into_barh(_CAT, 'v', **kw),
    'alternating_barh':    lambda **kw: plot_alternating_value_counts_barh(_CAT[['a', 'b']], _CAT[['a', 'b']], **kw),
    'comparing_box':       lambda **kw: plot_comparing_box(_NUM1, _NUM2, **kw),
}

@pytest.mark.parametrize('call', list(_ALL_PLOTS.values()), ids=list(_ALL_PLOTS))
class TestAllPlotFunctions:
    def test_returns_fig_and_2d_axes(self, call):
        fig, axs = call()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, np.ndarray) and axs.ndim == 2

    def test_figsize_is_applied(self, call):
        fig, _ = call(figsize=(7, 3))
        assert tuple(fig.get_size_inches()) == (7, 3)

    def test_figure_title_and_style(self, call):
        fig, _ = call(fig_title='Hello', suptitle_kwargs={'fontsize': 9})
        assert fig._suptitle.get_text() == 'Hello'
        assert fig._suptitle.get_fontsize() == 9

    def test_default_figure_title_style(self, call):
        fig, _ = call(fig_title='Hello')
        assert fig._suptitle.get_fontsize() == defaults_fig_title['fontsize']

    def test_title_kwargs_apply_to_every_subplot(self, call):
        _, axs = call(title_kwargs={'fontsize': 21})
        assert all(ax.title.get_fontsize() == 21 for ax in _visible(axs))

    def test_setup_kwargs_apply_to_every_subplot(self, call):
        _, axs = call(setup_kwargs={'facecolor': '#123456'})
        assert all(_hex(ax.get_facecolor()) == '#123456' for ax in _visible(axs))

    def test_every_visible_subplot_has_a_title(self, call):
        _, axs = call()
        assert all(ax.get_title() for ax in _visible(axs))

    def test_invalid_ncols_fails_before_any_figure_is_created(self, call):
        with pytest.raises(ValueError, match='ncols'):
            call(ncols=0)
        assert plt.get_fignums() == []

    def test_ncols_sets_the_grid_width(self, call):
        _, axs = call(ncols=1)
        assert axs.shape[1] == 1