"""Reduction of raw Quick Stats records to county-year yields."""

import pytest

from qdann import nass


def row(**overrides: str) -> dict[str, str]:
    """One Quick Stats CSV record, carrying only the fields the reduction reads."""
    return {
        "state_fips_code": "19",
        "county_ansi": "169",
        "year": "2018",
        "class_desc": "ALL CLASSES",
        "util_practice_desc": "GRAIN",
        "unit_desc": "ACRES",
        "Value": "1",
    } | overrides


def test_bushels_per_acre_becomes_tonnes_per_hectare() -> None:
    """180 bu/ac of corn is 11.30 t/ha at 56 lb/bu (7 CFR 810)."""
    df = nass.to_yield_t_ha(
        area_rows=[row()],
        production_rows=[row(unit_desc="BU", Value="180")],
        crop="maize",
    )
    assert df["yield_t_ha"].to_list() == pytest.approx([11.2982], abs=1e-4)


def test_thousands_separators_survive_the_parse() -> None:
    df = nass.to_yield_t_ha(
        area_rows=[row(Value="1,000")],
        production_rows=[row(unit_desc="BU", Value="180,000")],
        crop="maize",
    )
    assert df["yield_t_ha"].to_list() == pytest.approx([11.2982], abs=1e-4)


def test_suppressed_and_pooled_and_unpaired_rows_are_dropped() -> None:
    """Only a county-year with an ANSI code and both halves of the ratio keeps a label."""
    counties = ("001", "003", "005", "   ")
    df = nass.to_yield_t_ha(
        area_rows=[
            row(county_ansi="001"),
            row(county_ansi="003", Value="(D)"),  # area withheld
            row(county_ansi="005"),
            row(county_ansi="   "),  # OTHER (COMBINED) COUNTIES
        ],
        production_rows=[
            row(county_ansi=ansi, unit_desc="BU", Value="180")
            if ansi != "005"
            else row(county_ansi=ansi, unit_desc="BU", Value="(D)")  # production withheld
            for ansi in counties
        ],
        crop="maize",
    )
    # the leading zero survives: a county is "19001", never "191"
    assert df["fips"].to_list() == ["19001"]


def test_corn_silage_is_not_grain() -> None:
    """Silage is harvested whole-plant and reported in TONS, so it carries no bushel weight."""
    df = nass.to_yield_t_ha(
        area_rows=[row(util_practice_desc="SILAGE")],
        production_rows=[row(util_practice_desc="SILAGE", unit_desc="BU", Value="180")],
        crop="maize",
    )
    assert df.empty


def test_winter_wheat_keeps_only_the_winter_class() -> None:
    """Below the state level wheat is published per class, and the paper wants only WINTER."""
    rows = [
        (class_desc, county_ansi)
        for class_desc, county_ansi in (("WINTER", "001"), ("SPRING", "003"))
    ]
    df = nass.to_yield_t_ha(
        area_rows=[
            row(
                class_desc=class_desc,
                county_ansi=county_ansi,
                util_practice_desc="ALL UTILIZATION PRACTICES",
            )
            for class_desc, county_ansi in rows
        ],
        production_rows=[
            row(
                class_desc=class_desc,
                county_ansi=county_ansi,
                util_practice_desc="ALL UTILIZATION PRACTICES",
                unit_desc="BU",
                Value="60",
            )
            for class_desc, county_ansi in rows
        ],
        crop="winter_wheat",
    )
    assert df["fips"].to_list() == ["19001"]
