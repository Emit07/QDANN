"""Production and harvested area on the lab explorer's global 5' frame (Stage A).

The gridded-land-sector-emissions repo reads SPAM-style grids of 2160 x 4320 cells,
-180..180 / -90..90 (`griddedlm.io.read_grid`). This writes the same frame per crop-year:

  qdann_<source>_<year>_P_<SPAM>.tif    production, t
  qdann_<source>_<year>_H_<SPAM>.tif    harvested area, ha
  qdann_<source>_<year>_cov_<SPAM>.tif  share of the cell's CDL crop pixels that are covered

Area basis: NASS harvested area of grain (silage corn excluded), placed on CDL crop pixels.
`--source nass` is the statistics-only baseline. Each county's NASS area and production are
spread evenly over its CDL pixels. A withheld county pools with the others of its
agricultural district under the district's OTHER (COMBINED) COUNTIES row. A county with
neither stays uncovered and only lowers `cov`.

  uv run python -m qdann.export --crop maize --years 2018 --state 19
"""

import argparse
import logging
import pathlib

import numpy
import pandas
import pyproj
import rasterio
import rasterio.transform
import shapely

from qdann import gee, mpc, nass

logger = logging.getLogger(__name__)

CELLS_PER_DEGREE = 12
FRAME = (180 * CELLS_PER_DEGREE, 360 * CELLS_PER_DEGREE)
FRAME_TRANSFORM = rasterio.transform.from_origin(
    -180.0, 90.0, 1 / CELLS_PER_DEGREE, 1 / CELLS_PER_DEGREE
)
DISTRICT_YEARS = range(2008, 2024)
PIXEL_HA = mpc.PIXEL_M**2 / 10_000
CROP_TO_SPAM = {"maize": "MAIZ", "soybean": "SOYB", "winter_wheat": "WHEA"}


def cell_counts(mask: numpy.ndarray, on: mpc.Grid) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The flat 5' cell index of each True pixel's centre, as (cells, pixel counts)."""
    rows, cols = numpy.nonzero(mask)
    xs, ys = on.transform @ (cols + 0.5, rows + 0.5)
    lon, lat = pyproj.Transformer.from_crs(on.crs, "EPSG:4326", always_xy=True).transform(
        xs, ys
    )
    row = numpy.floor((90.0 - lat) * CELLS_PER_DEGREE).astype(numpy.int64)
    col = numpy.floor((lon + 180.0) * CELLS_PER_DEGREE).astype(numpy.int64)
    return numpy.unique(row * FRAME[1] + col, return_counts=True)


def spread(
    pixels: dict[str, tuple[numpy.ndarray, numpy.ndarray]],
    stats: pandas.DataFrame,
    districts: dict[str, str],
) -> dict[str, numpy.ndarray]:
    """P, H and cov on the frame for one crop-year.

    `pixels` maps each county to its (cells, counts); `stats` is one year of `nass.totals`.
    Every published unit's totals land on its pixels in proportion to pixel count, so the
    frame sums back to exactly the published totals of every unit that has pixels.
    """
    production = numpy.zeros(FRAME[0] * FRAME[1])
    area = numpy.zeros_like(production)
    covered = numpy.zeros_like(production)
    crop_pixels = numpy.zeros_like(production)
    for cells, counts in pixels.values():
        numpy.add.at(crop_pixels, cells, counts)

    published = stats[stats["fips"].notna()].set_index("fips")
    combined = stats[stats["fips"].isna()].set_index("asd")
    pools: dict[str, list[str]] = {}
    for fips in pixels:
        if fips in published.index:
            pools[fips] = [fips]
        elif districts.get(fips) in combined.index:
            pools.setdefault(districts[fips], []).append(fips)

    for unit, members in pools.items():
        row = published.loc[unit] if unit in published.index else combined.loc[unit]
        cells = numpy.concatenate([pixels[f][0] for f in members])
        counts = numpy.concatenate([pixels[f][1] for f in members]).astype(numpy.float64)
        total = counts.sum()
        if total == 0:
            logger.warning("%s publishes %.0f t but has no CDL pixels", unit, row["production_t"])
            continue
        numpy.add.at(production, cells, row["production_t"] * counts / total)
        numpy.add.at(area, cells, row["area_ha"] * counts / total)
        numpy.add.at(covered, cells, counts)

    placed = set(pools)
    for unit, row in pandas.concat([published, combined]).iterrows():
        if unit not in placed:
            logger.warning("%s publishes %.0f t on no mapped pixel", unit, row["production_t"])
    with numpy.errstate(invalid="ignore", divide="ignore"):
        cov = numpy.where(crop_pixels > 0, covered / crop_pixels, 0.0)
    return {
        name: grid.reshape(FRAME)
        for name, grid in (("P", production), ("H", area), ("cov", cov))
    }


def write(grid: numpy.ndarray, path: pathlib.Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=FRAME[1],
        height=FRAME[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=FRAME_TRANSFORM,
        nodata=None,
        compress="deflate",
    ) as dst:
        dst.write(grid.astype(numpy.float32), 1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(CROP_TO_SPAM), required=True)
    parser.add_argument("--years", type=gee.year_range, required=True)
    parser.add_argument("--state", nargs="+", required=True, help="two-digit state FIPS")
    parser.add_argument("--out", type=pathlib.Path, default=gee.DATA_DIR / "export")
    args = parser.parse_args()

    # a withheld county has no record in the year it is withheld, so its district comes
    # from the years it is published: fetch the whole CDL era, one state per query to stay
    # under the record cap
    area_rows, production_rows = [], []
    for state in args.state:
        a, p = nass.fetch_rows(crop=args.crop, years=DISTRICT_YEARS, state_fips=(state,))
        area_rows += a
        production_rows += p
    stats = nass.totals(area_rows=area_rows, production_rows=production_rows, crop=args.crop)
    districts = stats.attrs["districts"]
    counties = {
        fips: polygon
        for state in args.state
        for fips, polygon in mpc.state_counties(state).items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(**mpc.GDAL_ENV):
        for year in args.years:
            tiles = mpc.cdl_tiles(year=year, region=shapely.union_all(list(counties.values())))
            pixels = {}
            for fips, polygon in counties.items():
                on = mpc.grid(polygon)
                pixels[fips] = cell_counts(
                    mpc.cropland(crop=args.crop, polygon=polygon, on=on, tiles=tiles), on
                )
            year_stats = stats[stats["year"] == year]
            grids = spread(pixels, year_stats, districts)
            cdl_ha = sum(c.sum() for _, c in pixels.values()) * PIXEL_HA
            # the lab's own CDL grids (build_cdl_5min.py) count these same pixels at 0.09 ha,
            # so H / CDL area is the calibration factor between the two area bases
            logger.info(
                "%d: %.0f t placed of %.0f t published; H %.0f ha vs CDL %.0f ha (x%.3f)",
                year,
                grids["P"].sum(),
                year_stats["production_t"].sum(),
                grids["H"].sum(),
                cdl_ha,
                grids["H"].sum() / cdl_ha,
            )
            for name, grid in grids.items():
                write(
                    grid,
                    args.out / f"qdann_nass_{year}_{name}_{CROP_TO_SPAM[args.crop]}.tif",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
