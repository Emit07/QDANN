"""County-level GCVI harmonics from Landsat Collection 2, via Microsoft Planetary Computer.

The same features as `gee`, read from MPC's free STAC catalogue instead of Earth Engine, so
the pipeline runs without an Earth Engine quota or a Cloud bucket. The masking, mosaic and
fitting rules are `gee`'s, reused rather than restated.

  uv run python -m qdann.mpc --crop maize --years 2018 --county 19169   # one county
  uv run python -m qdann.mpc --crop maize --years 2008-2018 --state 19  # days of reads
  uv run python -m qdann.mpc --crop maize --years 2008-2018 --state 19 --target
  uv run python -m qdann.mpc --crop maize --fit data/mpc/19

A read writes one long parquet per year under data/mpc/<county or state>/ and skips the
years already there, so an interrupted run resumes. --target also reads the stored target
pixels from the same scenes, so the one download serves both legs. --fit fits Eq. 2 to
them. Nothing here writes the GEE tables (gcvi_maize.parquet and friends): every output
is named *_mpc_*.

Two differences from `gee` are deliberate. Earth Engine reduces on the date mosaic's
default EPSG:4326 grid, resampling both Landsat and CDL; here each county is read on its
own UTM zone's Landsat grid, so a pixel is the one Landsat measured. And each row keeps
n_clear and n_crop, the counts behind the mean, which AMBIGUITIES.md #20's clear-fraction
rule needs and Earth Engine never returned.
"""

import argparse
import concurrent.futures
import contextlib
import itertools
import logging
import math
import os
import pathlib
import re
import shutil
import socket
import time
import typing
import urllib.request

import numpy
import pandas
import planetary_computer
import pyogrio.raw
import pyproj
import pystac.errors
import pystac_client
import rasterio
import rasterio.cache
import rasterio.errors
import rasterio.features
import rasterio.warp
import shapely
import shapely.ops
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

from qdann import gee, target

logger = logging.getLogger(__name__)

# the vintage Earth Engine's TIGER/2018/Counties was built from, in NAD83 (EPSG:4269)
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip"
TIGER_PATH = gee.DATA_DIR / "tl_2018_us_county.zip"
TIGER_CRS = "EPSG:4269"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT = "landsat-c2-l2"
CDL = "usda-cdl"
# the usda-cdl collection's temporal extent ends 2021-12-31 (checked 2026-11-26); later
# years come from CropScape's per-state files
MPC_CDL_LAST = 2021
CROPSCAPE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
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
    # GDAL waits forever by default: a socket dropped by a network change hung a read for hours
    "GDAL_HTTP_TIMEOUT": "60",
    # neighbouring counties' boxes overlap, so the same COG blocks are asked for twice
    "GDAL_CACHEMAX": 512,
}


class Grid(typing.NamedTuple):
    crs: pyproj.CRS
    transform: Affine
    width: int
    height: int


class Pixels(typing.NamedTuple):
    """One county-year's crop pixels: green and NIR are (crop pixels, dates), NaN where
    masked, the pixels in the order of `crop`'s True cells."""

    on: Grid
    crop: numpy.ndarray
    dates: numpy.ndarray
    green: numpy.ndarray
    nir: numpy.ndarray


def download(url: str, path: pathlib.Path) -> pathlib.Path:
    """Fetch `url` to `path` once; a partial download never takes the final name."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # per process and node: Sherlock's array jobs can fetch the same file at once
        part = path.with_suffix(f"{path.suffix}.{socket.gethostname()}.{os.getpid()}.part")
        try:
            # a timeout, unlike urlretrieve: a stalled socket would hang a job to its limit
            with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f)
                expected = r.headers.get("Content-Length")
            if expected is not None and part.stat().st_size != int(expected):
                raise OSError(f"{url}: {part.stat().st_size} of {expected} bytes")
            part.rename(path)
        finally:
            part.unlink(missing_ok=True)
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
        except (OSError, pystac_client.exceptions.APIError, pystac.errors.STACError) as error:
            if attempt == attempts - 1:
                raise
            delay = 10 * 2**attempt
            logger.warning("%s; retrying in %d s", error, delay)
            # a truncated response stays in GDAL's HTTP cache, and every retry would reread it
            rasterio.cache.invalidate_all()
            time.sleep(delay)
    raise AssertionError("unreachable")


def catalog() -> pystac_client.Client:
    # hrefs are signed when opened, not here: a token lasts about an hour, a run for days
    return pystac_client.Client.open(STAC_URL)


def date(item) -> str:
    """The acquisition date as Earth Engine formats it, `image.date()` in UTC."""
    return item.datetime.strftime("%Y-%m-%d")


def scenes(
    crop: str, year: int, region: shapely.Geometry, landsat7_last: int | None = None
) -> list:
    """Every Tier 1 scene touching `region` in the window, ordered date, platform, id.

    Within a date that order is the mosaic's: the last valid pixel wins, as with
    Earth Engine's mosaic() over gee.reflectance's merge. With `landsat7_last`, Landsat 7
    is left out after that year (AMBIGUITIES #30: its overpass drifts earlier).
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
        and not (
            item.properties["platform"] == "landsat-7"
            and landsat7_last is not None
            and year > landsat7_last
        )
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


def cropscape(year: int, state_fips: str) -> pathlib.Path:
    """The state's CDL GeoTIFF from CropScape, the same 30 m Albers product MPC tiles,
    downloaded once to data/cdl/ (about 210 MB for Iowa)."""
    path = gee.DATA_DIR / "cdl" / f"CDL_{year}_{state_fips}.tif"
    if not path.exists():
        def ask() -> bytes:
            query = f"{CROPSCAPE_URL}?year={year}&fips={state_fips}"
            with urllib.request.urlopen(query, timeout=120) as r:
                return r.read()

        url = re.search(rb"<returnURL>(.+?)</returnURL>", retry(ask))
        if url is None:
            raise ValueError(f"CropScape has no CDL for state {state_fips} in {year}")
        retry(lambda: download(url.group(1).decode(), path))
    return path


def cdl_tiles(
    year: int, counties: dict[str, shapely.Geometry]
) -> list[tuple[str, shapely.Geometry]]:
    """(href, footprint) of the CDL rasters covering `counties`: MPC's tiles to 2021,
    CropScape's state files after, whose footprint is the state's counties here."""
    region = shapely.union_all(list(counties.values()))
    if year > MPC_CDL_LAST:
        return [
            (
                str(cropscape(year, state)),
                shapely.union_all([p for f, p in counties.items() if f[:2] == state]),
            )
            for state in sorted({fips[:2] for fips in counties})
        ]

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

    ret = [
        (item.assets["cropland"].href, shapely.geometry.shape(item.geometry))
        for item in retry(search)
        if item.id.startswith(f"cropland_{year}_")
        and shapely.geometry.shape(item.geometry).intersects(region)
    ]
    # an empty list would give every county an empty crop mask, silently
    if not ret:
        raise ValueError(f"no CDL tiles on MPC for {year} over this region")
    return ret


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


def sample(
    src: rasterio.DatasetReader, lon: numpy.ndarray, lat: numpy.ndarray
) -> numpy.ndarray:
    """Band 1 at the pixel containing each point, `fill(src)` off the raster.

    The pixel Landsat measured, never a resampled neighbour: Earth Engine's sampleRegions
    read one about 40% of the time (AMBIGUITIES.md #19). A point on a pixel edge is the
    pixel to its east or south, as `rowcol` floors.
    """
    xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, lon, lat)
    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
    ret = numpy.full(len(lon), fill(src), dtype=src.dtypes[0])
    for i, (row, col) in enumerate(zip(rows, cols)):
        if 0 <= row < src.height and 0 <= col < src.width:
            ret[i] = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    return ret


def opened(href: str) -> rasterio.DatasetReader:
    return rasterio.open(planetary_computer.sign(href))


def readable(item) -> bool:
    """Whether every asset the reader uses opens, after the usual retries."""
    try:
        for name in ASSETS:
            retry(lambda name=name: opened(item.assets[name].href).close())
    except rasterio.errors.RasterioIOError:
        return False
    return True


def cropland(
    crop: str, polygon: shapely.Geometry, on: Grid, tiles: list[tuple[str, shapely.Geometry]]
) -> numpy.ndarray:
    """True for the county's pixels CDL labels as the crop."""
    classes = numpy.zeros((on.height, on.width), dtype=numpy.uint8)
    footprint = polygon.envelope
    for href, tile in tiles:
        if not tile.intersects(footprint):
            continue
        with opened(href) as src:
            values = retry(lambda src=src: read(src, on))
            outside = fill(src)
        # the tiles abut without overlapping. Class 0 is CDL's background, which is what
        # a state file holds beyond its border: it must not overwrite the neighbour's classes
        classes = numpy.where((values != outside) & (values != 0), values, classes)
    return (classes == gee.CROP_TO_CDL_CODE[crop]) & inside(polygon, on)


def date_means(
    items: list,
    counties: dict[str, shapely.Geometry],
    grids: dict[str, Grid],
    crops: dict[str, numpy.ndarray],
    points: pandas.DataFrame,
    full: bool = False,
) -> tuple[list[dict], pandas.DataFrame, dict[str, tuple[numpy.ndarray, numpy.ndarray]]]:
    """One row per county the date's scenes touch, and one per clear target point. Each
    scene's assets are opened once and read per county, so only one county-date is ever
    held in memory, and the points reuse the blocks the counties just fetched. With
    `full`, also each county's green and NIR at every crop pixel, NaN where masked, in the
    order of `crops[fips]`'s True pixels."""
    footprints = [shapely.geometry.shape(item.geometry) for item in items]
    rows, series = [], {}
    # the reads wait on HTTP, not the CPU, and each thread holds its own dataset; the
    # options rasterio.Env set on the main thread are global, so the workers see them
    with (
        contextlib.ExitStack() as stack,
        concurrent.futures.ThreadPoolExecutor(len(ASSETS)) as pool,
    ):
        hrefs = [item.assets[name].href for item in items for name in ASSETS]
        flat = [stack.enter_context(src) for src in pool.map(opened, hrefs)]
        sources = list(itertools.batched(flat, len(ASSETS)))
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
                reflectance(*pool.map(lambda src: read(src, on), bands))
                for bands in touching
            )
            clear = crop & ~numpy.isnan(green)
            n_clear = int(clear.sum())
            if full:
                series[fips] = (
                    green[crop].astype(numpy.float32),
                    nir[crop].astype(numpy.float32),
                )
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
        # no CDL mask: the points were drawn on that year's crop pixels (AMBIGUITIES.md #19)
        green, nir = mosaic(
            reflectance(
                *pool.map(lambda src: sample(src, points["lon"], points["lat"]), bands)
            )
            for bands in sources
        )
    clear = ~numpy.isnan(green)
    return (
        rows,
        points.loc[clear, ["fips", "pid"]].assign(
            date=date(items[0]), green=green[clear], nir=nir[clear]
        ),
        series,
    )


def county_means(
    crop: str,
    year: int,
    counties: dict[str, shapely.Geometry],
    points: pandas.DataFrame | None = None,
    full: bool = False,
    landsat7_last: int | None = None,
) -> tuple[pandas.DataFrame, list[str], pandas.DataFrame, dict[str, Pixels]]:
    """The long table (fips, year, date, green, nir, n_clear, n_crop), the scene ids, the
    target's long table (fips, pid, year, date, green, nir) at `points` (pid, fips, lon,
    lat), read from the same scenes in the same pass, and with `full` every crop pixel's
    series per county (empty without)."""
    if points is None:
        points = pandas.DataFrame(columns=["fips", "pid", "lon", "lat"])
    region = shapely.union_all(list(counties.values()))
    items = scenes(crop=crop, year=year, region=region, landsat7_last=landsat7_last)
    tiles = cdl_tiles(year=year, counties=counties)
    grids = {fips: grid(polygon) for fips, polygon in counties.items()}
    crops = {
        fips: cropland(crop=crop, polygon=polygon, on=grids[fips], tiles=tiles)
        for fips, polygon in counties.items()
    }
    logger.info("%d: %d scenes, %d CDL tiles", year, len(items), len(tiles))
    # dates one at a time: 4 in flight and GDAL_HTTP_MERGE_CONSECUTIVE_RANGES were each
    # slower on a 4-CPU laptop (2026-11-25, 12 dates x 3 rounds), and the blobs speak
    # HTTP/1.1 only, so GDAL_HTTP_MULTIPLEX has nothing to multiplex
    rows, pixels, dates, series, used = [], [], [], {}, []
    for day, group in itertools.groupby(items, key=date):
        group = list(group)
        try:
            means, values, stacks = retry(
                lambda group=group: date_means(group, counties, grids, crops, points, full)
            )
        except rasterio.errors.RasterioIOError:
            # MPC holds a few truncated assets (LC08 027/030 2022-07-23's nir08 is 12 kB),
            # which fail on every retry: drop the scene, never the county-year
            bad = [item for item in group if not readable(item)]
            if not bad:
                raise
            logger.warning("%s: skipping unreadable scene(s) %s", day, [i.id for i in bad])
            group = [item for item in group if item not in bad]
            if not group:
                continue
            means, values, stacks = retry(
                lambda group=group: date_means(group, counties, grids, crops, points, full)
            )
        used += group
        rows += means
        pixels.append(values)
        dates.append(day)
        for fips in counties:
            series.setdefault(fips, []).append(stacks.get(fips))
        logger.info("%d: %s done (%d scenes)", year, day, len(group))
    long = pandas.DataFrame(
        rows, columns=["fips", "date", "green", "nir", "n_clear", "n_crop"]
    )
    long.insert(1, "year", year)
    full_series = {}
    for fips, stacks in series.items() if full else ():
        # a date whose scenes miss the county leaves every pixel unseen
        unseen = numpy.full(int(crops[fips].sum()), numpy.nan, dtype=numpy.float32)
        full_series[fips] = Pixels(
            on=grids[fips],
            crop=crops[fips],
            dates=numpy.array(dates),
            green=numpy.stack([s[0] if s else unseen for s in stacks], axis=1),
            nir=numpy.stack([s[1] if s else unseen for s in stacks], axis=1),
        )
    pixels = pandas.concat(pixels, ignore_index=True) if pixels else points.iloc[:0]
    pixels.insert(2, "year", year)
    return long, [item.id for item in used], pixels, full_series


def year_path(out: pathlib.Path, crop: str, year: int) -> pathlib.Path:
    return out / f"gcvi_mpc_{crop}_{year}.parquet"


def target_path(out: pathlib.Path, crop: str, year: int) -> pathlib.Path:
    return out / f"target_mpc_{crop}_{year}.parquet"


def points(crop: str, year: int, fips: typing.Iterable[str]) -> pandas.DataFrame:
    """The target pixels Earth Engine drew (AMBIGUITIES.md #19), stored with their weather,
    so every pixel read here still finds its weather row in `target.join`."""
    stored = pandas.read_parquet(
        gee.DATA_DIR / f"gridmet_target_{crop}.parquet",
        columns=["fips", "pid", "year", "lon", "lat"],
    )
    stored = stored[(stored["year"] == year) & stored["fips"].isin(list(fips))]
    return stored.drop(columns="year").reset_index(drop=True)


def read_years(
    out: pathlib.Path, prefix: str, crop: str, keys: typing.Iterable[str]
) -> pandas.DataFrame:
    """Every year file under `out` at any depth, so a run split one county per job
    (--out data/mpc/19/<fips>, see sherlock.sbatch) fits as one. A county read into two
    of those directories would enter its fit twice, so a repeated row is an error."""
    paths = sorted(out.rglob(f"{prefix}_{crop}_[0-9][0-9][0-9][0-9].parquet"))
    long = pandas.concat([pandas.read_parquet(path) for path in paths])
    assert not long.duplicated([*keys, "date"]).any(), f"a {prefix} row repeats under {out}"
    return long


# AMBIGUITIES.md #20: a county-date under this clear fraction is a mean over a sliver, not
# the county, so the fit drops it and the gate forgives the two grids disagreeing on it
SLIVER = 0.05


def fit(out: pathlib.Path, crop: str) -> pandas.DataFrame:
    """Fit Eq. 2 to every year read into `out`, without #20's sliver dates."""
    long = read_years(out=out, prefix="gcvi_mpc", crop=crop, keys=gee.KEY_COLUMNS)
    long = long[long["n_clear"] >= SLIVER * long["n_crop"]]
    return gee.fit_table(long=gee.clean(long), crop=crop)


def fit_target(out: pathlib.Path, crop: str) -> pandas.DataFrame:
    """Eq. 2 per target pixel-year, under the target's observation floor."""
    long = gee.clean(
        read_years(out=out, prefix="target_mpc", crop=crop, keys=target.KEY_COLUMNS),
        keys=target.KEY_COLUMNS,
    )
    print(target.retained(long).to_string(index=False))
    return gee.fit_table(
        long=long,
        crop=crop,
        keys=target.KEY_COLUMNS,
        min_observations=target.MIN_OBSERVATIONS,
    )


# the Phase 1.4 gate against Earth Engine; a date under SLIVER clear is reported, not gated
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
    # `fit` never sees a sliver, so only a date it would keep can disagree; one missing
    # from MPC's own rows has no count, so it can't pass as a sliver
    one_sided = {
        d for d in set(g["date"]) ^ set(m["date"]) if frac.get(d, math.inf) >= SLIVER
    }
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
    parser.add_argument(
        "--target",
        action="store_true",
        help="also read the stored target pixels, in the same pass",
    )
    parser.add_argument(
        "--landsat7-last",
        type=int,
        help="leave Landsat 7 out after this year (AMBIGUITIES #30); default keeps it",
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
                # sherlock.sbatch's layout, data/mpc/<state>/<county>
                year_path(
                    out=gee.DATA_DIR / "mpc" / fips[:2] / fips,
                    crop=args.crop,
                    year=int(year),
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
        if any(args.fit.rglob(f"target_mpc_{args.crop}_*.parquet")):
            table = fit_target(out=args.fit, crop=args.crop)
            path = args.fit / f"gcvi_target_mpc_{args.crop}.parquet"
            table.to_parquet(path, index=False)
            print(f"\n{len(table):d} pixel-years -> {path}")
        return 0

    state = args.county[:2] if args.county else args.state
    counties = state_counties(state, (args.county,) if args.county else ())
    out = args.out or gee.DATA_DIR / "mpc" / (args.county or state)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(**GDAL_ENV):
        for year in args.years:
            path = year_path(out=out, crop=args.crop, year=year)
            pixels_path = target_path(out=out, crop=args.crop, year=year)
            if path.exists() and (pixels_path.exists() or not args.target):
                logger.info("%d: %s exists, skipping", year, path)
                continue
            started = time.monotonic()
            long, ids, pixels, _ = county_means(
                crop=args.crop,
                year=year,
                counties=counties,
                points=points(crop=args.crop, year=year, fips=counties)
                if args.target
                else None,
                landsat7_last=args.landsat7_last,
            )
            (out / f"scenes_mpc_{args.crop}_{year}.txt").write_text("\n".join(ids))
            if args.target:
                pixels.to_parquet(pixels_path, index=False)
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
