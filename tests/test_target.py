"""Checks on the target leg's client-side half, which is the join and its guard.

The sampling and the point reads can only be checked against Earth Engine. What is pinned
here is the one thing that fails silently: the two exports each re-derive the point set, so
a fitted pixel-year without a weather row means the sampling was not deterministic, and a
merge would simply drop the row rather than say so.
"""

import numpy
import pandas
import pytest

from qdann import target, weather


def _frames(pids=("a", "b"), weather_pids=None, n_observations=(30, 30)):
    keys = pandas.DataFrame(
        {"fips": "19169", "pid": list(pids), "year": 2018}
    )
    gcvi = keys.assign(
        n_observations=list(n_observations),
        **{
            name: numpy.where(numpy.asarray(n_observations) >= 20, 1.0, numpy.nan)
            for name in weather.HARMONIC_COLUMNS
        },
    )
    table = pandas.DataFrame(
        {"fips": "19169", "pid": list(weather_pids or pids), "year": 2018}
    ).assign(**dict.fromkeys(weather.weather_columns("maize"), 2.0))
    return gcvi, table


def test_the_join_keeps_the_27_features_and_no_label():
    table = target.join(*_frames(), crop="maize")
    assert list(table.columns) == [
        *target.KEY_COLUMNS,
        *weather.HARMONIC_COLUMNS,
        *weather.weather_columns("maize"),
    ]
    assert len(table.columns) - len(target.KEY_COLUMNS) == weather.N_FEATURES
    assert weather.LABEL_COLUMN not in table.columns


def test_a_pixel_year_below_the_floor_is_dropped_rather_than_joined_as_nan():
    table = target.join(*_frames(n_observations=(30, 7)), crop="maize")
    assert list(table["pid"]) == ["a"]


def test_the_two_legs_sampling_different_points_raises():
    with pytest.raises(AssertionError, match="different points"):
        target.join(*_frames(weather_pids=("a", "c")), crop="maize")


def test_the_retained_fraction_counts_observations_per_pixel_year():
    long = pandas.DataFrame(
        {
            "fips": "19169",
            "pid": ["a"] * 30 + ["b"] * 10,
            "year": 2018,
            "date": [*range(30), *range(10)],
        }
    )
    table = target.retained(long).set_index("floor")
    assert table.loc[7, "pixel_years"] == 2
    assert table.loc[20, "pixel_years"] == 1
    assert table.loc[30, "retained"] == 0.5
