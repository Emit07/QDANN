"""County-level GCVI harmonics from Landsat Collection 2, via Microsoft Planetary Computer.

The same features as `gee`, read from MPC's free STAC catalogue instead of Earth Engine, so
the pipeline runs without an Earth Engine quota or a Cloud bucket. The masking, mosaic and
fitting rules are `gee`'s, reused rather than restated.

  uv run python -m qdann.mpc --crop maize --years 2018 --county 19169   # one county
  uv run python -m qdann.mpc --crop maize --years 2008-2018 --state 19  # days of reads
  uv run python -m qdann.mpc --crop maize --fit data/mpc/19

A read writes one long parquet per year under data/mpc/<county or state>/ and skips the
years already there, so an interrupted run resumes. --fit fits Eq. 2 to them. Nothing here
writes the GEE tables (gcvi_maize.parquet and friends): every output is named *_mpc_*.

Two differences from `gee` are deliberate. Earth Engine reduces on the date mosaic's
default EPSG:4326 grid, resampling both Landsat and CDL; here each county is read on its
own UTM zone's Landsat grid, so a pixel is the one Landsat measured. And each row keeps
n_clear and n_crop, the counts behind the mean, which AMBIGUITIES.md #20's clear-fraction
rule needs and Earth Engine never returned.
"""

import argparse
import contextlib
import itertools
import logging
import math
import pathlib
import time
import typing
import urllib.request

import numpy
import pandas
import planetary_computer
import pyogrio.raw
import pyproj
import pystac_client
import rasterio
import rasterio.features
import shapely
import shapely.ops
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

from qdann import gee

logger = logging.getLogger(__name__)

# the vintage Earth Engine's TIGER/2018/Counties was built from, in NAD83 (EPSG:4269)
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip"
TIGER_PATH = gee.DATA_DIR / "tl_2018_us_county.zip"
TIGER_CRS = "EPSG:4269"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT = "landsat-c2-l2"
CDL = "usda-cdl"
# gee.SENSOR_BANDS's merge order: a later platform lands on top of the same-day mosaic
PLATFORMS = ("landsat-5", "landsat-7", "landsat-8", "landsat-9")
# MPC names bands by wavelength, so one set covers the four sensors gee.SENSOR_BANDS renames
ASSETS = ("green", "nir08", "qa_pixel", "qa_radsat")

PIXEL_M = gee.NATIVE_SCALE_M
# Landsat's UTM grids put pixel edges at 15 m mod 30
GRID_OFFSET_M = 15

RETRIES = 5
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    # neighbouring counties' boxes overlap, so the same COG blocks are asked for twice
    "GDAL_CACHEMAX": 512,
}


class Grid(typing.NamedTuple):
    crs: pyproj.CRS
    transform: Affine
    width: int
    height: int


def download(url: str, path: pathlib.Path) -> pathlib.Path:
    """Fetch `url` to `path` once; a partial download never takes the final name."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def state_counties(
    state_fips: str, geoids: tuple[str, ...] = ()
) -> dict[str, shapely.Geometry]:
    """County polygons by GEOID, in EPSG:4269, mirroring `gee.state_counties`."""
    # STATEFP has to be read for the filter to see it: GDAL ignores the unread fields,
    # and a filter on one then matches nothing rather than raising. The fields come back in
    # file order, not `columns` order, so GEOID is found by name
    meta, _, wkb, fields = pyogrio.raw.read(
        download(TIGER_URL, TIGER_PATH),
        columns=["GEOID", "STATEFP"],
        where=f"STATEFP = '{state_fips}'",
    )
    ids = fields[list(meta["fields"]).index("GEOID")]
    ret = dict(zip(ids, shapely.from_wkb(wkb)))
    return {g: ret[g] for g in geoids} if geoids else ret


def retry[T](fn: typing.Callable[[], T], attempts: int = RETRIES) -> T:
    """Call `fn`, backing off on network errors; the last failure is raised, never skipped."""
    for attempt in range(attempts):
        try:
            return fn()
        except (OSError, pystac_client.exceptions.APIError) as error:
            if attempt == attempts - 1:
                raise
            delay = 10 * 2**attempt
            logger.warning("%s; retrying in %d s", error, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def catalog() -> pystac_client.Client:
    # hrefs are signed when opened, not here: a token lasts about an hour, a run for days
    return pystac_client.Client.open(STAC_URL)


def date(item) -> str:
    """The acquisition date as Earth Engine formats it, `image.date()` in UTC."""
    return item.datetime.strftime("%Y-%m-%d")


def scenes(crop: str, year: int, region: shapely.Geometry) -> list:
    """Every Tier 1 scene touching `region` in the window, ordered date, platform, id.

    Within a date that order is the mosaic's: the last valid pixel wins, as with
    Earth Engine's mosaic() over gee.reflectance's merge.
    """
    start, end = gee.window(crop=crop, year=year)
    # a loose outline keeps the request small; the exact test is client-side below
    loose = region.buffer(0.01).simplify(0.005)

    def search() -> list:
        return list(
            catalog()
            .search(
                collections=[LANDSAT],
                intersects=shapely.geometry.mapping(loose),
                datetime=f"{start}/{end}",
                query={"landsat:collection_category": {"eq": "T1"}},
            )
            .items()
        )

    items = [
        item
        for item in retry(search)
        # the server warns it does not conform to QUERY, so the tier is checked here too
        if item.properties["landsat:collection_category"] == "T1"
        and item.properties["platform"] in PLATFORMS
        # STAC's end is inclusive, the window's is not
        and start <= date(item) < end
        and shapely.geometry.shape(item.geometry).intersects(region)
    ]
    return sorted(
        items,
        key=lambda item: (
            date(item),
            PLATFORMS.index(item.properties["platform"]),
            item.id,
        ),
    )


def cdl_tiles(year: int, region: shapely.Geometry) -> list:
    def search() -> list:
        return list(
            catalog()
            .search(
                collections=[CDL],
                intersects=shapely.geometry.mapping(region.envelope),
                datetime=str(year),
            )
            .items()
        )

    return [
        item
        for item in retry(search)
        if item.id.startswith(f"cropland_{year}_")
        and shapely.geometry.shape(item.geometry).intersects(region)
    ]


def utm(polygon: shapely.Geometry) -> pyproj.CRS:
    """The UTM zone of the county's centroid, WGS84 as Landsat's grids are."""
    return pyproj.CRS.from_epsg(32600 + int((polygon.centroid.x + 180) // 6) + 1)


def project(polygon: shapely.Geometry, crs: pyproj.CRS) -> shapely.Geometry:
    to = pyproj.Transformer.from_crs(TIGER_CRS, crs, always_xy=True).transform
    return shapely.ops.transform(to, polygon)


def grid(polygon: shapely.Geometry) -> Grid:
    """The Landsat-aligned 30 m grid covering the county, in its centroid's UTM zone."""
    crs = utm(polygon)
    minx, miny, maxx, maxy = project(polygon, crs).bounds

    def snap(value: float, up: bool) -> float:
        rounded = math.ceil if up else math.floor
        return rounded((value - GRID_OFFSET_M) / PIXEL_M) * PIXEL_M + GRID_OFFSET_M

    left, top = snap(minx, up=False), snap(maxy, up=True)
    right, bottom = snap(maxx, up=True), snap(miny, up=False)
    return Grid(
        crs=crs,
        transform=Affine(PIXEL_M, 0, left, 0, -PIXEL_M, top),
        width=round((right - left) / PIXEL_M),
        height=round((top - bottom) / PIXEL_M),
    )


def inside(polygon: shapely.Geometry, on: Grid) -> numpy.ndarray:
    """True for the pixels whose centre falls in the county."""
    return rasterio.features.geometry_mask(
        [project(polygon, on.crs)],
        out_shape=(on.height, on.width),
        transform=on.transform,
        invert=True,
    )


def fill(src: rasterio.DatasetReader) -> int:
    """What a pixel outside `src` reads as: its own nodata, else its dtype's maximum.

    Never 0 for a band without nodata. qa_pixel and qa_radsat would read 0 as clear and
    unsaturated, and the warper rewrites a valid value equal to the fill (a real 0 comes
    out as 1, which in CDL is maize). The maximum is all QA bits set, saturated, and no
    CDL class, so it can only ever mask.
    """
    if src.nodata is not None:
        return int(src.nodata)
    return int(numpy.iinfo(src.dtypes[0]).max)


def read(src: rasterio.DatasetReader, on: Grid) -> numpy.ndarray:
    """Band 1 of `src` on the county grid, `fill(src)` outside it. Nearest only: CDL is
    categorical, and across UTM zones any other kernel would blend Landsat pixels no
    sensor measured."""
    with WarpedVRT(
        src,
        crs=on.crs,
        transform=on.transform,
        width=on.width,
        height=on.height,
        resampling=Resampling.nearest,
        nodata=fill(src),
    ) as vrt:
        return vrt.read(1)


def reflectance(
    green: numpy.ndarray, nir: numpy.ndarray, qa: numpy.ndarray, radsat: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Scaled green and NIR, NaN wherever gee.reflectance masks the pixel."""
    green = green * gee.SCALE_FACTOR + gee.SCALE_OFFSET
    nir = nir * gee.SCALE_FACTOR + gee.SCALE_OFFSET
    valid = (
        (green >= 0)
        & (green <= 1)
        & (nir >= 0)
        & (nir <= 1)
        & ((qa & gee.QA_MASK) == 0)
        & (radsat == 0)
    )
    return numpy.where(valid, green, numpy.nan), numpy.where(valid, nir, numpy.nan)


def mosaic(
    layers: typing.Iterable[tuple[numpy.ndarray, numpy.ndarray]],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Collapse one date's scenes, the last valid pixel winning as in Earth Engine."""
    green = nir = None
    for g, n in layers:
        if green is None:
            green, nir = numpy.full_like(g, numpy.nan), numpy.full_like(n, numpy.nan)
        valid = ~numpy.isnan(g)
        green[valid], nir[valid] = g[valid], n[valid]
    return green, nir


def opened(href: str) -> rasterio.DatasetReader:
    return rasterio.open(planetary_computer.sign(href))


def cropland(
    crop: str, polygon: shapely.Geometry, on: Grid, tiles: list
) -> numpy.ndarray:
    """True for the county's pixels CDL labels as the crop."""
    classes = numpy.zeros((on.height, on.width), dtype=numpy.uint8)
    footprint = polygon.envelope
    for tile in tiles:
        if not shapely.geometry.shape(tile.geometry).intersects(footprint):
            continue
        with opened(tile.assets["cropland"].href) as src:
            values = retry(lambda src=src: read(src, on))
            outside = fill(src)
        # the tiles abut without overlapping
        classes = numpy.where(values != outside, values, classes)
    return (classes == gee.CROP_TO_CDL_CODE[crop]) & inside(polygon, on)


def date_means(
    items: list,
    counties: dict[str, shapely.Geometry],
    grids: dict[str, Grid],
    crops: dict[str, numpy.ndarray],
) -> list[dict]:
    """One row per county the date's scenes touch. Each scene's assets are opened once
    and read per county, so only one county-date is ever held in memory."""
    footprints = [shapely.geometry.shape(item.geometry) for item in items]
    rows = []
    with contextlib.ExitStack() as stack:
        sources = [
            [stack.enter_context(opened(item.assets[name].href)) for name in ASSETS]
            for item in items
        ]
        for fips, polygon in counties.items():
            touching = [
                bands
                for bands, footprint in zip(sources, footprints)
                if footprint.intersects(polygon)
            ]
            if not touching:
                continue
            on, crop = grids[fips], crops[fips]
            green, nir = mosaic(
                reflectance(*(read(src, on) for src in bands)) for bands in touching
            )
            clear = crop & ~numpy.isnan(green)
            n_clear = int(clear.sum())
            rows.append(
                {
                    "fips": fips,
                    "date": date(items[0]),
                    "green": green[clear].mean(dtype=numpy.float64)
                    if n_clear
                    else numpy.nan,
                    "nir": nir[clear].mean(dtype=numpy.float64)
                    if n_clear
                    else numpy.nan,
                    "n_clear": n_clear,
                    "n_crop": int(crop.sum()),
                }
            )
    return rows


def county_means(
    crop: str, year: int, counties: dict[str, shapely.Geometry]
) -> tuple[pandas.DataFrame, list[str]]:
    """The long table (fips, year, date, green, nir, n_clear, n_crop) and the scene ids."""
    region = shapely.union_all(list(counties.values()))
    items = scenes(crop=crop, year=year, region=region)
    tiles = cdl_tiles(year=year, region=region)
    grids = {fips: grid(polygon) for fips, polygon in counties.items()}
    crops = {
        fips: cropland(crop=crop, polygon=polygon, on=grids[fips], tiles=tiles)
        for fips, polygon in counties.items()
    }
    logger.info("%d: %d scenes, %d CDL tiles", year, len(items), len(tiles))
    rows = []
    for day, group in itertools.groupby(items, key=date):
        group = list(group)
        rows += retry(lambda group=group: date_means(group, counties, grids, crops))
        logger.info("%d: %s done (%d scenes)", year, day, len(group))
    long = pandas.DataFrame(
        rows, columns=["fips", "date", "green", "nir", "n_clear", "n_crop"]
    )
    long.insert(1, "year", year)
    return long, [item.id for item in items]


def year_path(out: pathlib.Path, crop: str, year: int) -> pathlib.Path:
    return out / f"gcvi_mpc_{crop}_{year}.parquet"


def fit(out: pathlib.Path, crop: str) -> pandas.DataFrame:
    """Fit Eq. 2 to every year read into `out`; the long tables are kept, so the counts
    stay available for #20's rule after `gee.clean` drops them."""
    paths = sorted(out.glob(f"gcvi_mpc_{crop}_[0-9][0-9][0-9][0-9].parquet"))
    long = pandas.concat([pandas.read_parquet(path) for path in paths])
    return gee.fit_table(long=gee.clean(long), crop=crop)


# the Phase 1.4 gate against Earth Engine; a date under SLIVER clear is reported, not gated
SLIVER = 0.05
MAX_ONE_SIDED_CLEAR = 100
MIN_IDENTICAL_DATES = 0.95
MAX_MEDIAN_REL = 0.003
MAX_P95_REL = 0.01
MAX_COEF_SD = 0.05
COEFFICIENTS = ["c", "a1", "b1", "a2", "b2", "a3", "b3"]


def compare(
    gee_raw: pandas.DataFrame, mpc_raw: pandas.DataFrame, sd: pandas.Series, crop: str
) -> dict:
    """Tiers B and C1 for one county-year: the per-date means, then Eq. 2 refitted on both
    sides without the sliver dates, whose handful of pixels the two grids pick differently."""
    g, m = gee.clean(gee_raw), gee.clean(mpc_raw)
    clear = mpc_raw.set_index("date")["n_clear"]
    frac = clear / mpc_raw.set_index("date")["n_crop"]
    one_sided = set(g["date"]) ^ set(m["date"])
    both = g.merge(m, on="date", suffixes=("_g", "_m"))
    rel = (both["gcvi_m"] / both["gcvi_g"] - 1).abs()
    sliver = both["date"].map(frac) < SLIVER
    gated = rel[~sliver]
    keep = set(both["date"][~sliver])

    def worst(g, m):
        fg, fm = (gee.fit_table(long=x, crop=crop).iloc[0] for x in (g, m))
        return float(((fg[COEFFICIENTS] - fm[COEFFICIENTS]).abs() / sd).max())

    return {
        "dates_gee": len(g),
        "dates_mpc": len(m),
        "one_sided": len(one_sided),
        # a date missing from MPC's own rows has no count, so it can't pass as a sliver
        "one_sided_clear": max((clear.get(d, math.inf) for d in one_sided), default=0),
        "rel_median": float(gated.median()),
        "rel_p95": float(gated.quantile(0.95)),
        "rel_max_sliver": float(rel.max()),
        "coef_all": worst(g, m),
        "coef_filtered": worst(g[g["date"].isin(keep)], m[m["date"].isin(keep)]),
    }


def gate(results: pandas.DataFrame) -> list[str]:
    """The failed criteria of the Tier B/C1 table `compare` rows make, empty if it passes."""
    failed = []
    if (results["one_sided"] == 0).mean() < MIN_IDENTICAL_DATES:
        failed.append("B: date sets identical in too few county-years")
    if (results["one_sided_clear"] >= MAX_ONE_SIDED_CLEAR).any():
        failed.append("B: a one-sided date has >= 100 clear pixels")
    if (results["rel_median"] > MAX_MEDIAN_REL).any():
        failed.append("B: GCVI median relative gap > 0.3%")
    if (results["rel_p95"] > MAX_P95_REL).any():
        failed.append("B: GCVI p95 relative gap > 1%")
    if (results["coef_filtered"] > MAX_COEF_SD).any():
        failed.append("C1: sliver-filtered coefficient gap > 0.05 SD")
    return failed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(gee.CROP_TO_CDL_CODE), required=True)
    parser.add_argument("--years", type=gee.year_range)
    parser.add_argument(
        "--state", default="19", help="two-digit state FIPS, e.g. 19 for Iowa"
    )
    parser.add_argument("--county", help="five-digit county FIPS, e.g. 19169")
    parser.add_argument("--out", type=pathlib.Path, help="default data/mpc/<area>")
    parser.add_argument("--fit", type=pathlib.Path, help="a directory of read years")
    parser.add_argument(
        "--gate",
        type=pathlib.Path,
        help="a file of 'fips year' lines to check against GEE",
    )
    args = parser.parse_args()

    if args.gate is not None:
        reference = gee.DATA_DIR / "reference_gee"
        sd = pandas.read_parquet(reference / f"source_{args.crop}.parquet")[
            COEFFICIENTS
        ].std()
        rows = []
        for fips, year in (line.split() for line in args.gate.read_text().splitlines()):
            mpc_raw = pandas.read_parquet(
                year_path(
                    out=gee.DATA_DIR / "mpc" / fips, crop=args.crop, year=int(year)
                )
            )
            gee_raw = pandas.read_csv(
                reference / "tier_b" / f"gee_{args.crop}_{fips}_{year}.csv",
                dtype={"fips": str},
            )
            rows.append(
                {"fips": fips, "year": int(year)}
                | compare(gee_raw, mpc_raw, sd, args.crop)
            )
        results = pandas.DataFrame(rows)
        print(results.to_string(index=False, float_format="{:.4f}".format))
        failed = gate(results)
        print("\n" + ("\n".join(failed) if failed else "Tiers B and C1 pass"))
        return 1 if failed else 0

    if args.fit is not None:
        table = fit(out=args.fit, crop=args.crop)
        path = args.fit / f"gcvi_mpc_{args.crop}.parquet"
        table.to_parquet(path, index=False)
        print(table["n_observations"].describe().to_string())
        print(f"\n{len(table):d} county-years -> {path}")
        return 0

    state = args.county[:2] if args.county else args.state
    counties = state_counties(state, (args.county,) if args.county else ())
    out = args.out or gee.DATA_DIR / "mpc" / (args.county or state)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(**GDAL_ENV):
        for year in args.years:
            path = year_path(out=out, crop=args.crop, year=year)
            if path.exists():
                logger.info("%d: %s exists, skipping", year, path)
                continue
            started = time.monotonic()
            long, ids = county_means(crop=args.crop, year=year, counties=counties)
            (out / f"scenes_mpc_{args.crop}_{year}.txt").write_text("\n".join(ids))
            # written last, so a year only counts as done once all of it is on disk
            long.to_parquet(path, index=False)
            logger.info(
                "%d: %d rows in %.0f s -> %s",
                year,
                len(long),
                time.monotonic() - started,
                path,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
