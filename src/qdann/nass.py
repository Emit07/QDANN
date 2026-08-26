"""County-year yields from USDA NASS Quick Stats: the labels of the source domain.

Yield is rebuilt as PRODUCTION / AREA HARVESTED rather than read from the YIELD statistic.
Below the state level wheat is published only per class (WINTER, SPRING, DURUM) and the
ALL CLASSES yield used at state level does not exist, so a county wheat yield has to be
reconstructed; doing the same for corn and soybeans keeps one code path and covers more
counties than the YIELD statistic does.

  uv run python -m qdann.nass --crop maize --years 2008-2018 --state 19
"""

import argparse
import collections
import csv
import io
import logging
import os
import pathlib
import urllib.error
import urllib.parse
import urllib.request

import pandas

logger = logging.getLogger(__name__)


API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
# the API errors with "exceeds limit = 50000" rather than truncating
RECORD_LIMIT = 50000
# a query matching no record is answered with HTTP 400 and this body rather than an empty
# CSV, so an unrecognised parameter value cannot be told apart from an empty result
EMPTY_RESULT_BODY = '{"error":["bad request - invalid query"]}'

HA_PER_ACRE = 0.40468564
KG_PER_LB = 0.45359237
KG_PER_TONNE = 1000
# 7 CFR 810 (US Grain Standards Act)
COMMODITY_TO_LB_PER_BUSHEL = {"CORN": 56, "SOYBEANS": 60, "WHEAT": 60}

# the paper's vocabulary on the left, the only spellings Quick Stats accepts on the right
CROP_TO_COMMODITY_DESC = {
    "maize": "CORN",
    "soybean": "SOYBEANS",
    "winter_wheat": "WHEAT",
}
# at county level corn and soybeans are published as ALL CLASSES; only wheat splits by class
CROP_TO_CLASS_DESC = {
    "maize": "ALL CLASSES",
    "soybean": "ALL CLASSES",
    "winter_wheat": "WINTER",
}
# corn is published for grain and for silage, which is harvested whole-plant and reported in
# TONS, so it carries no bushel weight
KEPT_UTIL_PRACTICE_DESCS = frozenset({"GRAIN", "ALL UTILIZATION PRACTICES"})
SUPPRESSED_VALUES = frozenset({"(D)", "(Z)", "(S)", "(NA)"})

DATA_DIR = pathlib.Path(__file__).parents[2] / "data"


def load_api_key() -> str:
    """Read USDA_NASS_API_KEY from the environment, falling back to the repo's .env."""
    if key := os.environ.get("USDA_NASS_API_KEY"):
        return key
    dot_env = DATA_DIR.parent / ".env"
    if dot_env.exists():
        for line in dot_env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "USDA_NASS_API_KEY":
                return value.strip().strip("\"'")
    raise SystemExit(f"USDA_NASS_API_KEY is in neither the environment nor {dot_env}")


def get_rows(params: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    """GET one Quick Stats query, returning [] for the 400 that means "no records"."""
    try:
        with urllib.request.urlopen(
            f"{API_URL}?{urllib.parse.urlencode(params)}"
        ) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        if exc.code == 400 and body.strip() == EMPTY_RESULT_BODY:
            return []
        # the key rides in the query string, so report the parameters without it
        redacted = [(name, value) for name, value in params if name != "key"]
        raise RuntimeError(
            f"Quick Stats {exc.code:d} on {redacted}: {body[:200]}"
        ) from None
    rows = list(csv.DictReader(io.StringIO(body)))
    assert len(rows) < RECORD_LIMIT, (
        f"{len(rows):d} records hit the cap; split the query"
    )
    return rows


def get_fips(row: dict[str, str]) -> str | None:
    """Zero-padded five-digit county FIPS, or None for a row that names no county.

    `OTHER (COMBINED) COUNTIES` pools the counties withheld for confidentiality within an
    agricultural district and carries no ANSI code, so it cannot be placed on the map.
    """
    county_ansi = row["county_ansi"].strip()
    if not county_ansi:
        return None
    return f"{int(row['state_fips_code']):02d}{int(county_ansi):03d}"


def to_yield_t_ha(
    area_rows: list[dict[str, str]],
    production_rows: list[dict[str, str]],
    crop: str,
) -> pandas.DataFrame:
    """Reduce raw AREA HARVESTED and PRODUCTION records to one yield per (county, year)."""
    totals: dict[tuple[str, int], list[float]] = collections.defaultdict(
        lambda: [0.0, 0.0]
    )
    for index, rows, unit_desc in ((0, area_rows, "ACRES"), (1, production_rows, "BU")):
        for row in rows:
            fips = get_fips(row)
            if (
                fips is None
                or row["unit_desc"] != unit_desc
                or row["class_desc"] != CROP_TO_CLASS_DESC[crop]
                or row["util_practice_desc"] not in KEPT_UTIL_PRACTICE_DESCS
                or row["Value"] in SUPPRESSED_VALUES
            ):
                continue
            # published values carry thousands separators
            totals[(fips, int(row["year"]))][index] += float(
                row["Value"].replace(",", "")
            )

    lb_per_bushel = COMMODITY_TO_LB_PER_BUSHEL[CROP_TO_COMMODITY_DESC[crop]]
    # a county-year keeps its label only where both halves of the ratio survived suppression:
    # bare area whose production is withheld would otherwise read as a yield of zero
    return pandas.DataFrame.from_records(
        [
            {
                "fips": fips,
                "year": year,
                "yield_t_ha": bushels
                * lb_per_bushel
                * KG_PER_LB
                / (acres * HA_PER_ACRE * KG_PER_TONNE),
            }
            for (fips, year), (acres, bushels) in sorted(totals.items())
            if acres > 0 and bushels > 0
        ],
        columns=["fips", "year", "yield_t_ha"],
    )


def fetch(crop: str, years: range, state_fips: tuple[str, ...]) -> pandas.DataFrame:
    """County-year yields for one crop, as columns fips / year / yield_t_ha."""
    common = (
        ("key", load_api_key()),
        ("source_desc", "SURVEY"),
        ("sector_desc", "CROPS"),
        ("agg_level_desc", "COUNTY"),
        ("freq_desc", "ANNUAL"),
        # keeps in-season forecasts out
        ("reference_period_desc", "YEAR"),
        # without this, irrigated and non-irrigated come back as separate rows and every
        # Great Plains wheat county triplicates
        ("prodn_practice_desc", "ALL PRODUCTION PRACTICES"),
        ("commodity_desc", CROP_TO_COMMODITY_DESC[crop]),
        *[("year", f"{year:d}") for year in years],
        *[("state_fips_code", fips) for fips in state_fips],
        ("format", "CSV"),
    )
    area_rows, production_rows = (
        get_rows(common + (("statisticcat_desc", statisticcat_desc),))
        for statisticcat_desc in ("AREA HARVESTED", "PRODUCTION")
    )
    logger.info(
        f"{len(area_rows):d} area and {len(production_rows):d} production records"
    )
    return to_yield_t_ha(
        area_rows=area_rows, production_rows=production_rows, crop=crop
    )


def year_range(text: str) -> range:
    start, _, end = text.partition("-")
    return range(int(start), int(end or start) + 1)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop", choices=sorted(CROP_TO_COMMODITY_DESC), default="maize"
    )
    parser.add_argument("--years", type=year_range, default=year_range("2008-2018"))
    parser.add_argument("--state", nargs=argparse.ONE_OR_MORE, default=["19"])
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    df = fetch(crop=args.crop, years=args.years, state_fips=tuple(args.state))
    out = args.out or DATA_DIR / f"nass_{args.crop}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(df.to_string())
    print(f"\n{len(df):d} rows -> {out}")
    print(df["yield_t_ha"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
