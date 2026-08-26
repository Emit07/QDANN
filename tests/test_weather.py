"""The weather leg and the join that assembles the source table."""

import pandas
import pytest

from qdann import weather


def labels(fips: list[str], year: int = 2018) -> pandas.DataFrame:
    return pandas.DataFrame(
        {"fips": fips, "year": year, "yield_t_ha": [11.0] * len(fips)}
    )


def features(fips: list[str], crop: str = "maize", year: int = 2018) -> tuple:
    keys = {"fips": fips, "year": year}
    gcvi = pandas.DataFrame(
        keys | {"n_observations": 40} | dict.fromkeys(weather.HARMONIC_COLUMNS, 1.0)
    )
    weather_frame = pandas.DataFrame(
        keys | dict.fromkeys(weather.weather_columns(crop), 1.0)
    )
    return gcvi, weather_frame


def test_the_growing_season_is_crop_dependent():
    assert weather.weather_columns("maize")[:5] == [
        "pr_5",
        "srad_5",
        "tmmn_5",
        "tmmx_5",
        "vpd_5",
    ]
    # winter wheat is sown the previous autumn, so its season runs March to June
    assert weather.weather_columns("winter_wheat")[-1] == "vpd_6"
    for crop in weather.CROP_TO_MONTHS:
        assert len(weather.weather_columns(crop)) == 20


def test_the_joined_table_carries_exactly_the_27_features():
    gcvi, weather_frame = features(["19169", "19153"])
    joined = weather.join(
        labels=labels(["19169", "19153"]),
        gcvi=gcvi,
        weather=weather_frame,
        crop="maize",
    )
    assert len(joined) == 2
    assert len(joined.columns) == weather.N_FEATURES + 3
    # n_observations is a diagnostic of the satellite leg, not a predictor
    assert "n_observations" not in joined.columns


def test_a_county_renamed_mid_window_fails_the_join_rate():
    # NASS labels Oglala Lakota County 46102 from 2015; a features table built on a TIGER
    # vintage older than the rename still calls it Shannon County, 46113
    others = [f"461{n:02d}" for n in (1, 5, 7, 9, 11, 15, 17, 19, 21, 23)]
    gcvi, weather_frame = features([*others, "46113"])
    with pytest.raises(AssertionError, match="46102"):
        weather.join(
            labels=labels([*others, "46102"]),
            gcvi=gcvi,
            weather=weather_frame,
            crop="maize",
        )


def test_a_county_year_missing_one_leg_is_dropped_not_filled():
    # a county whose Landsat record is too cloudy to fit has no harmonics, so it is not a
    # sample at all -- carrying it with NaN features would poison the standardization
    fips = [f"191{n:02d}" for n in range(20)]
    gcvi, weather_frame = features(fips)
    joined = weather.join(
        labels=labels(fips),
        gcvi=gcvi[gcvi["fips"] != fips[0]],
        weather=weather_frame,
        crop="maize",
    )
    assert len(joined) == 19
    assert fips[0] not in set(joined["fips"])
    assert not joined.isna().to_numpy().any()
