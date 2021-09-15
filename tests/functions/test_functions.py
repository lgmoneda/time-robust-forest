import numpy as np
import pandas as pd
import pytest
from time_robust_forest.functions import (
    check_categoricals_match,
    check_min_sample_periods,
    check_min_sample_periods_dict,
    check_numerical_match,
    fill_right_dict,
    gini_impurity_score_by_period,
    initialize_period_dict,
    score_by_period,
    std_agg,
    std_score_by_period,
)


@pytest.mark.parametrize(
    ("cnt", "s1", "s2", "expected"),
    [
        (5, 2, 3, 0.6633249580710799),
        (1, 1, 1, 0),
        (5, -1, -1, 0),
        (10, 2, 3, 0.5099019513592785),
    ],
)
def test_std_agg(cnt, s1, s2, expected):
    assert std_agg(cnt, s1, s2) == expected


@pytest.mark.parametrize(
    ("count_dict", "min_sample_periods", "expected"),
    [
        ({0: 1, 1: 1}, 1, True),
        ({0: 1, 1: 1}, 2, False),
        ({0: 10, 1: 5}, 2, True),
        ({0: 10, 1: 2}, 3, False),
        ({0: 10}, 3, True),
    ],
)
def test_check_min_sample_periods_dict(
    count_dict, min_sample_periods, expected
):
    assert (
        check_min_sample_periods_dict(count_dict, min_sample_periods)
        == expected
    )


@pytest.mark.parametrize(
    ("X", "time_column", "min_sample_periods", "expected"),
    [
        (
            pd.DataFrame([[1, 0], [1, 1]], columns=["value", "time_column"]),
            "time_column",
            1,
            1,
        ),
        (
            pd.DataFrame([[1, 0], [1, 1]], columns=["value", "time_column"]),
            "time_column",
            2,
            0,
        ),
    ],
)
def test_check_min_sample_periods(X, time_column, min_sample_periods, expected):
    assert (
        check_min_sample_periods(X, time_column, min_sample_periods) == expected
    )


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        (
            [0, 1],
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
        ),
        (
            ["2020", "2021"],
            {
                "2020": {"count": 0, "sum": 0, "squared_sum": 0},
                "2021": {"count": 0, "sum": 0, "squared_sum": 0},
            },
        ),
    ],
)
def test_initialize_period_dict(periods, expected):
    assert initialize_period_dict(periods) == expected


@pytest.mark.parametrize(
    ("periods", "target", "weights", "right_dict", "expected"),
    [
        (
            np.array([0, 0, 1]),
            np.array([1, 0, 1]),
            np.array([1, 1, 1]),
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
        ),
        (
            np.array(["2020", "2020", "2021"]),
            np.array([1, 0, 1]),
            np.array([2, 1, 1]),
            {
                "2020": {"count": 0, "sum": 0, "squared_sum": 0},
                "2021": {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                "2020": {"count": 3, "sum": 2, "squared_sum": 2},
                "2021": {"count": 1, "sum": 1, "squared_sum": 1},
            },
        ),
    ],
)
def test_fill_right_dict(periods, target, weights, right_dict, expected):
    assert fill_right_dict(periods, target, weights, right_dict) == expected


@pytest.mark.parametrize(
    ("right_dict", "left_dict", "criterion", "period_criterion", "expected"),
    [
        (
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "std",
            "avg",
            0.5,
        ),
        (
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "std",
            "max",
            1.0,
        ),
        (
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "std_norm",
            "avg",
            0.25,
        ),
        (
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "std_norm",
            "max",
            0.5,
        ),
        (
            {
                0: {"count": 4, "sum": 3, "squared_sum": 3},
                1: {"count": 4, "sum": 3, "squared_sum": 3},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "gini",
            "avg",
            0.35833333333333334,
        ),
        (
            {
                0: {"count": 4, "sum": 3, "squared_sum": 3},
                1: {"count": 4, "sum": 3, "squared_sum": 3},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            "gini",
            "max",
            0.41666666666666663,
        ),
    ],
)
def test_score_by_period(
    right_dict, left_dict, criterion, period_criterion, expected
):
    assert (
        score_by_period(right_dict, left_dict, criterion, period_criterion)
        == expected
    )


@pytest.mark.parametrize(
    ("right_dict", "left_dict", "expected"),
    [
        (
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            [0.5, 0.0],
        ),
        (
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            {
                0: {"count": 0, "sum": 0, "squared_sum": 0},
                1: {"count": 0, "sum": 0, "squared_sum": 0},
            },
            [0.5, 0.0],
        ),
        (
            {
                0: {"count": 4, "sum": 3, "squared_sum": 3},
                1: {"count": 4, "sum": 3, "squared_sum": 3},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            [0.4553418012614795, 0.34641016151377546],
        ),
    ],
)
def test_std_score_by_period(right_dict, left_dict, expected):
    assert std_score_by_period(right_dict, left_dict, norm=True) == expected


@pytest.mark.parametrize(
    ("right_dict", "left_dict", "expected"),
    [
        (
            {
                0: {"count": 1, "sum": 0, "squared_sum": 0},
                1: {"count": 1, "sum": 0, "squared_sum": 0},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            [0.3333333333333333, 0.0],
        ),
        (
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            {
                0: {"count": 1, "sum": 0, "squared_sum": 0},
                1: {"count": 1, "sum": 0, "squared_sum": 0},
            },
            [0.3333333333333333, 0.0],
        ),
        (
            {
                0: {"count": 4, "sum": 3, "squared_sum": 3},
                1: {"count": 4, "sum": 3, "squared_sum": 3},
            },
            {
                0: {"count": 2, "sum": 1, "squared_sum": 1},
                1: {"count": 1, "sum": 1, "squared_sum": 1},
            },
            [0.41666666666666663, 0.30000000000000004],
        ),
    ],
)
def test_gini_impurity_score_by_period(right_dict, left_dict, expected):
    assert gini_impurity_score_by_period(right_dict, left_dict) == expected


@pytest.mark.parametrize(
    ("data", "categorical_features", "environment_column", "expected"),
    [
        (
            pd.DataFrame([[1, 1, 0], [1, 2, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            0.75,
        ),
        (
            pd.DataFrame([[1, 1, 0], [1, 1, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            1,
        ),
        (
            pd.DataFrame([[1, 1, 0], [2, 2, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            0.5,
        ),
    ],
)
def test_check_categoricals_match(
    data, categorical_features, environment_column, expected
):
    assert (
        check_categoricals_match(data, categorical_features, environment_column)
        == expected
    )


@pytest.mark.parametrize(
    ("data", "numerical_features", "environment_column", "expected"),
    [
        (
            pd.DataFrame([[1, 1, 0], [1, 2, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            0.75,
        ),
        (
            pd.DataFrame([[1, 1, 0], [1, 1, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            1,
        ),
        (
            pd.DataFrame([[1, 1, 0], [2, 2, 1]], columns=["c1", "c2", "env"]),
            ["c1", "c2"],
            "env",
            0.5,
        ),
        (
            pd.DataFrame(
                [[10, 5.5, 0], [-2, 2, 1]], columns=["c1", "c2", "env"]
            ),
            ["c1", "c2"],
            "env",
            0.5,
        ),
        (
            pd.DataFrame(
                [[10, 5.5, 0], [-2, 2, 0], [1, 10, 1], [3, 3, 1]],
                columns=["c1", "c2", "env"],
            ),
            ["c1", "c2"],
            "env",
            0.75,
        ),
    ],
)
def test_check_numerical_match(
    data, numerical_features, environment_column, expected
):
    assert (
        check_numerical_match(
            data, numerical_features, environment_column, n_q=2
        )
        == expected
    )
