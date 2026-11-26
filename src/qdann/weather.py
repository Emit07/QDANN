"""County-level monthly weather from gridMET, and the joined source table.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 2.3. See SPEC.md
section 1. Produces the twenty weather features of every source-domain sample, then joins
them to the seven satellite features and the NASS labels.

  uv run python -m qdann.weather --crop maize --years 2008-2018 --state 19
  uv run python -m qdann.weather --crop maize --join

The first is a live Earth Engine query; the second is local, and needs the parquet each of
the two earlier legs writes.
"""

import argparse
import calendar
import hashlib
import logging
import math
import pathlib
import typing
import urllib.request

import ee
import numpy
import pandas
import rasterio
import rasterio.windows
from rasterio.transform import Affine

from qdann import env, gee

logger = logging.getLogger(__name__)


DATA_DIR = pathlib.Path(__file__).parents[2] / "data"

GRIDMET = "IDAHO_EPSCOR/GRIDMET"
# section 2.3 writes ppt for precipitation; the gridMET band is pr. The paper performs no
# feature selection, so this list is fixed rather than tuned
VARIABLES = ("pr", "srad", "tmmn", "tmmx", "vpd")
CROP_TO_MONTHS = {
    "maize": range(5, 9),
    "soybean": range(5, 9),
    "winter_wheat": range(3, 7),
}

# the Climatology Lab's annual files, the source Earth Engine ingests GRIDMET from
GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data/{variable}_{year:d}.nc"
GRIDMET_NAMES = {
    "pr": "precipitation_amount",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "vpd": "mean_vapor_pressure_deficit",
}

HARMONIC_COLUMNS = ("c", "a1", "b1", "a2", "b2", "a3", "b3")
KEY_COLUMNS = ("fips", "year")
LABEL_COLUMN = "yield_t_ha"
N_FEATURES = 27

# NASS uses each year's contemporary county codes while TIGER/2018 is a single vintage, so
# a county renamed mid-window drops out silently -- Shannon County, SD (46113) became
# Oglala Lakota County (46102) in 2015. Failing on the rate catches that and every other
# join bug without enumerating renames
MIN_JOIN_RATE = 0.95


def weather_columns(crop: str) -> list[str]:
    return [
        f"{variable}_{month:d}"
        for month in CROP_TO_MONTHS[crop]
        for variable in VARIABLES
    ]


def monthly_means(crop: str, year: int) -> ee.Image:
    """One image whose bands are <variable>_<month>, each the mean of that month's days."""
    return ee.Image.cat(
        [
            ee.ImageCollection(GRIDMET)
            .filterDate(
                ee.Date.fromYMD(year, month, 1),
                ee.Date.fromYMD(year, month, 1).advance(1, "month"),
            )
            .select(VARIABLES)
            .mean()
            .rename([f"{variable}_{month:d}" for variable in VARIABLES])
            for month in CROP_TO_MONTHS[crop]
        ]
    )


def native_scale() -> ee.Number:
    """gridMET's own ~4638 m grid, which both the county mean and the point read use."""
    return (
        ee.ImageCollection(GRIDMET)
        .first()
        .select(VARIABLES[0])
        .projection()
        .nominalScale()
    )


def county_means(
    crop: str, year: int, counties: ee.FeatureCollection
) -> ee.FeatureCollection:
    """The twenty features per county, reduced at gridMET's own ~4638 m scale.

    SPEC.md section 1 records the paper resampling gridMET to 30 m, which is what the
    subfield samples need -- each pixel takes its 4 km cell's value. A county mean is the
    same quantity either way, so it is taken natively rather than off an upsampled grid.
    """
    image = monthly_means(crop=crop, year=year)
    scale = native_scale()

    def per_county(county: ee.Feature) -> ee.Feature:
        means = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=county.geometry(),
            scale=scale,
            maxPixels=int(1e13),
            tileScale=4,
        )
        return ee.Feature(
            None, means.combine({"fips": county.get("GEOID"), "year": year})
        )

    return counties.map(per_county)


def fetch(crop: str, years: range, state_fips: str) -> pandas.DataFrame:
    """County-year weather features, one interactive query per year.

    A county holds ~70 gridMET pixels, so unlike the Landsat leg this fits comfortably in
    getInfo and needs no batch export.
    """
    counties = gee.state_counties(state_fips)
    frames = []
    for year in years:
        features = county_means(crop=crop, year=year, counties=counties).getInfo()
        frames.append(pandas.DataFrame([f["properties"] for f in features["features"]]))
        logger.info(f"{year:d}: {len(frames[-1]):d} counties")
    ret = pandas.concat(frames, ignore_index=True)
    ret["fips"] = ret["fips"].astype(str)
    return ret.loc[:, [*KEY_COLUMNS, *weather_columns(crop)]].sort_values(
        list(KEY_COLUMNS)
    )


class Cells(typing.NamedTuple):
    """The twenty features on gridMET's own lat/lon grid over a window, (rows, cols) each."""

    values: dict[str, numpy.ndarray]
    transform: Affine

    def at(self, lon: numpy.ndarray, lat: numpy.ndarray) -> pandas.DataFrame:
        """Each point's containing cell, as `target.weather_values` reads it in Earth
        Engine: the value a 30 m upsample would give, without the upsample."""
        cols, rows = ~self.transform @ (numpy.asarray(lon), numpy.asarray(lat))
        rows, cols = numpy.floor(rows).astype(int), numpy.floor(cols).astype(int)
        return pandas.DataFrame({name: v[rows, cols] for name, v in self.values.items()})


def gridmet_file(variable: str, year: int) -> pathlib.Path:
    path = DATA_DIR / "gridmet" / f"{variable}_{year:d}.nc"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".nc.part")
        urllib.request.urlretrieve(GRIDMET_URL.format(variable=variable, year=year), part)
        part.rename(path)
    return path


def cell_means(
    crop: str, year: int, bounds: tuple[float, float, float, float]
) -> Cells:
    """`monthly_means` from the downloaded files, over `bounds` (west, south, east, north
    in degrees): each month the mean of its days, units as Earth Engine serves them."""
    values, transform = {}, None
    for variable in VARIABLES:
        name = f'NETCDF:"{gridmet_file(variable, year)}":{GRIDMET_NAMES[variable]}'
        with rasterio.open(name) as src:
            # whole cells, plus one of margin, so an edge pixel's cell is always inside
            exact = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            col, row = math.floor(exact.col_off) - 1, math.floor(exact.row_off) - 1
            window = rasterio.windows.Window(
                col,
                row,
                math.ceil(exact.col_off + exact.width) + 1 - col,
                math.ceil(exact.row_off + exact.height) + 1 - row,
            )
            transform = src.window_transform(window)
            assert src.count in (365, 366), f"{name} has {src.count:d} days"
            for month in CROP_TO_MONTHS[crop]:
                first = sum(calendar.monthrange(year, m)[1] for m in range(1, month))
                days = range(first + 1, first + calendar.monthrange(year, month)[1] + 1)
                raw = src.read(list(days), window=window, masked=True).astype(float)
                # the files pack shorts with a per-variable scale and offset
                scaled = raw * src.scales[0] + src.offsets[0]
                values[f"{variable}_{month:d}"] = scaled.mean(axis=0).filled(numpy.nan)
    return Cells(values=values, transform=transform)


def join(
    labels: pandas.DataFrame,
    gcvi: pandas.DataFrame,
    weather: pandas.DataFrame,
    crop: str,
) -> pandas.DataFrame:
    """One row per labelled county-year: 27 features and the yield.

    Labels are the sparse side -- Title 7 suppression drops county-years -- while features
    are computed for every county in the TIGER vintage, so the rate is matched rows over
    label rows and not the reverse.
    """
    keys = list(KEY_COLUMNS)
    features = gcvi.loc[:, [*keys, *HARMONIC_COLUMNS]].merge(
        weather, on=keys, how="inner"
    )
    ret = labels.merge(features, on=keys, how="inner")

    missing = labels.merge(features[keys], on=keys, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    assert len(ret) >= MIN_JOIN_RATE * len(labels), (
        f"only {len(ret):d} of {len(labels):d} label rows found features; "
        f"unmatched: {sorted(set(missing['fips']))[:10]}"
    )
    if len(missing):
        logger.warning(f"{len(missing):d} label rows have no features")
    return ret.loc[:, [*keys, LABEL_COLUMN, *HARMONIC_COLUMNS, *weather_columns(crop)]]


def read(path: pathlib.Path) -> pandas.DataFrame:
    """The parquet at `path`, with `attrs["source"]` naming the file and its sha256 so a
    model trained on it can say which table it came from (AMBIGUITIES #27). pandas 3 keeps
    attrs through filtering and column selection, but merge or concat with an unstamped
    frame drops them: re-stamp after combining tables."""
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the leg that writes it first")
    table = pandas.read_parquet(path).astype({"fips": str})
    table.attrs["source"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return table


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(CROP_TO_MONTHS), default="maize")
    parser.add_argument(
        "--years", type=gee.year_range, default=gee.year_range("2008-2018")
    )
    parser.add_argument(
        "--state", default="19", help="two-digit state FIPS, e.g. 19 for Iowa"
    )
    parser.add_argument(
        "--join", action="store_true", help="skip the query and build the source table"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"gridmet_{args.crop}.parquet"
    if not args.join:
        ee.Initialize(project=env.get("EE_PROJECT"))
        weather = fetch(crop=args.crop, years=args.years, state_fips=args.state)
        weather.to_parquet(out, index=False)
        print(weather.describe().to_string())
        print(f"\n{len(weather):d} county-years -> {out}")
        return 0

    table = join(
        labels=read(DATA_DIR / f"nass_{args.crop}.parquet"),
        gcvi=read(DATA_DIR / f"gcvi_{args.crop}.parquet"),
        weather=read(out),
        crop=args.crop,
    )
    source = DATA_DIR / f"source_{args.crop}.parquet"
    table.to_parquet(source, index=False)
    print(table.describe().to_string())
    print(f"\n{len(table):d} samples, {len(table.columns) - 3:d} features -> {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
