"""Checks on the parts of the Earth Engine leg that run without Earth Engine.

The reduction itself can only be checked against the service, so what is pinned here is
everything that would corrupt the features silently: the crop window t is normalized over,
the rows that must not reach Eq. 2, and the repeated-date guard that proves the same-day
mosaic ran.
"""

import numpy
import pandas
import pytest

from qdann import gee


def _long(dates, gcvi=None, fips="19169", year=2018):
    green = numpy.full(len(dates), 0.05)
    nir = (
        numpy.asarray(gcvi if gcvi is not None else numpy.arange(len(dates))) + 1
    ) * green
    return pandas.DataFrame(
        {"fips": fips, "year": year, "date": dates, "green": green, "nir": nir}
    )


def test_the_winter_wheat_window_starts_in_the_previous_autumn():
    assert gee.window(crop="winter_wheat", year=2018) == ("2017-09-01", "2018-09-01")
    assert gee.window(crop="maize", year=2018) == ("2018-01-01", "2019-01-01")


def test_normalized_time_spans_the_crop_window_not_the_calendar_year():
    """A wheat date in the October before harvest is early in its window, not late."""
    october = pandas.Series(["2017-10-01"])
    assert gee.normalized_time(october, crop="winter_wheat", year=2018).iloc[0] < 0.15
    july = pandas.Series(["2018-07-01"])
    assert (
        0.8 < gee.normalized_time(july, crop="winter_wheat", year=2018).iloc[0] < 0.95
    )


def test_fully_masked_county_dates_are_dropped():
    """A county-date with no unmasked pixel comes back with the bands simply absent."""
    long = _long(["2018-05-01", "2018-06-01", "2018-07-01"])
    long.loc[1, ["green", "nir"]] = numpy.nan
    assert list(gee.clean(long)["date"]) == ["2018-05-01", "2018-07-01"]


def test_a_reduction_that_returned_no_bands_at_all_is_still_cleanable():
    long = pandas.DataFrame({"fips": ["19169"], "year": [2018], "date": ["2018-05-01"]})
    assert gee.clean(long).empty


def test_near_zero_green_is_dropped_before_the_ratio():
    long = _long(["2018-05-01", "2018-06-01"])
    long.loc[0, "green"] = 1e-6
    cleaned = gee.clean(long)
    assert len(cleaned) == 1
    # left in, that row's GCVI would be in the tens of thousands
    assert cleaned["gcvi"].abs().max() < 10


def test_gcvi_is_eq_1_on_the_county_means():
    cleaned = gee.clean(_long(["2018-07-01"], gcvi=[2.5]))
    assert cleaned["gcvi"].iloc[0] == pytest.approx(2.5)


def test_an_unmosaicked_repeated_date_is_caught_not_fitted():
    dates = [f"2018-{m:02d}-01" for m in range(1, 13)]
    long = gee.clean(_long([*dates, "2018-07-01"]))
    with pytest.raises(AssertionError, match="repeated date"):
        gee.fit_table(long=long, crop="maize")


def test_fit_table_shape_and_observation_count():
    dates = [f"2018-{m:02d}-{d:02d}" for m in range(1, 13) for d in (3, 19)]
    table = gee.fit_table(long=gee.clean(_long(dates)), crop="maize")
    assert list(table.columns) == [
        "fips",
        "year",
        "n_observations",
        "c",
        "a1",
        "b1",
        "a2",
        "b2",
        "a3",
        "b3",
    ]
    assert len(table) == 1
    assert int(table["n_observations"].iloc[0]) == len(dates)
    # an integer round-trip would drop the leading zero of every state below 10
    assert pandas.api.types.is_string_dtype(table["fips"])
    assert table["fips"].iloc[0] == "19169"


def test_fit_table_keys_a_pixel_year_not_just_a_county_year():
    """The target leg fits one series per pixel, and two pixels share every date."""
    dates = [f"2018-{m:02d}-{d:02d}" for m in range(1, 13) for d in (3, 19)]
    keys = ("fips", "pid", "year")
    long = pandas.concat(
        [
            _long(dates, gcvi=numpy.arange(len(dates)) + offset).assign(pid=pid)
            for pid, offset in (("19169_2018_0", 0), ("19169_2018_1", 1))
        ]
    )
    table = gee.fit_table(long=gee.clean(long, keys=keys), crop="maize", keys=keys)
    assert list(table.columns)[:3] == list(keys)
    assert len(table) == 2
    # the offset lands entirely in the intercept, so the fit really was done per pixel
    assert table["c"].diff().iloc[1] == pytest.approx(1.0)


def test_the_observation_floor_returns_nan_rather_than_an_interpolating_fit():
    """Seven dates fit seven coefficients exactly; Phase 3's floor is what rejects that."""
    dates = [f"2018-{m:02d}-01" for m in range(1, 8)]
    long = gee.clean(_long(dates))
    assert gee.fit_table(long=long, crop="maize")["c"].notna().all()
    floored = gee.fit_table(long=long, crop="maize", min_observations=12)
    assert floored["c"].isna().all()
    assert int(floored["n_observations"].iloc[0]) == 7
