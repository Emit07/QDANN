"""Stage 1 of notes/yield_layer_plan.md: yield at every crop pixel of a county-year.

  uv run python -m qdann.predict --crop maize --year 2018 --county 19169 19015

One MPC pass reads every crop pixel's Landsat series (`mpc.county_means(full=True)`); Eq. 2
is fit per pixel above `MIN_OBSERVATIONS` clear dates (15, AMBIGUITIES #28); each pixel takes its gridMET cell's
twenty weather features (`weather.cell_means`); the saved Stage 0 model
(data/models/qdann_<crop>.pt) gives q50 and the q10/q90 interval. Writes under --out:

  <fips>/qdann_<crop>_<year>.tif   30 m, bands q50, lo, hi in t/ha; NaN off the crop
                                   and where the fit is under the floor
  qdann-raw_<year>_P_<SPAM>.tif    5' frame: H x the cell's mean predicted yield
  qdann-anchored_<year>_P_<SPAM>.tif  each county's NASS production placed in proportion
                                   to predicted yield, so it sums to NASS per county
  qdann_<year>_{H,cov,Ylo,Yhi}_<SPAM>.tif  area and coverage as `export`; the cell's mean
                                   pixel interval, t/ha
  predict_<crop>_<year>.json       per county: mean q50 vs NASS, coverage, time, memory, bytes
  <fips>/features_<crop>_<year>.npz  the 27 features per crop pixel and the grid
                                   (`save_features`), so other weights apply offline

lo/hi carry the county-scale conformal margin from `tune`; at a pixel they are wider in
truth, so Ylo/Yhi are a floor on the uncertainty, not a calibrated interval. A pixel with
no prediction stays NaN in the COG and takes the county-feature prediction in the 5'
products: the same model on the county's mean series, as the source table is built
(`county_inputs`).
"""

import argparse
import json
import logging
import pathlib
import resource
import time

import numpy
import pandas
import pyproj
import rasterio
import shapely
import torch
from rasterio.transform import Affine

from qdann import export, gee, harmonics, mpc, nass, weather
from qdann.model import QDANN

logger = logging.getLogger(__name__)

# pixels per Eq. 2 solve: bounds the (pixels, dates) float64 GCVI copy and the 7x7 stacks
CHUNK = 100_000
# a pixel's clear-date floor, lowered from the target's 20 on 2026-11-26 (AMBIGUITIES #28):
# 15-20 dates agree with Ma et al. nearly as well as 20+, below 12 the fit degrades fast
MIN_OBSERVATIONS = 15


def load(
    crop: str, path: pathlib.Path | None = None, check: bool = True
) -> tuple[QDANN, dict]:
    """The shipped data/models/qdann_<crop>.pt, or another `tune.save` file at `path`.

    `check` refuses a model trained on a non-MPC table (`check_reader`); pass False only
    to load a GEE-trained arm on purpose, for comparison."""
    path = weather.DATA_DIR / "models" / f"qdann_{crop}.pt" if path is None else path
    saved = torch.load(path, weights_only=False)
    model = QDANN(len(saved["features"]), dropout=saved["dropout"], intervals=True)
    model.load_state_dict(saved["state_dict"])
    logger.info("%s: %s", path.name, saved.get("provenance", "no provenance recorded"))
    if check:
        check_reader(saved)
    return model.eval(), saved


def check_reader(saved: dict) -> None:
    """The pixels here come from MPC, so the model's county table must too: a GEE-trained
    head on MPC pixels is silently flat (AMBIGUITIES #27). Checkpoints saved before
    `tune.save` recorded provenance pass with a warning."""
    inputs = saved.get("provenance", {}).get("inputs") or [None]
    if not isinstance(inputs[0], dict):
        logger.warning("checkpoint records no source table; cannot confirm it is MPC")
        return
    # ponytail: a filename tripwire, not identity; compare sha256 against the MPC table
    # if renamed copies ever appear
    name = pathlib.Path(inputs[0]["path"]).name
    if "mpc" not in name:
        raise SystemExit(f"model was trained on {name}, not an MPC source table (AMBIGUITIES #27)")


def pixel_harmonics(pixels: mpc.Pixels, crop: str, year: int) -> numpy.ndarray:
    """(crop pixels, 7) Eq. 2 coefficients, `gee.clean`'s masking applied per pixel."""
    t = gee.normalized_time(pandas.Series(pixels.dates), crop=crop, year=year).to_numpy()
    ret = numpy.empty((len(pixels.green), len(weather.HARMONIC_COLUMNS)))
    for i in range(0, len(ret), CHUNK):
        green = pixels.green[i : i + CHUNK].astype(float)
        nir = pixels.nir[i : i + CHUNK].astype(float)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            # Eq. 1, NaN where gee.clean would drop the date
            gcvi = numpy.where(green > gee.MIN_GREEN_REFLECTANCE, nir / green - 1, numpy.nan)
        ret[i : i + CHUNK] = harmonics.fit_many(
            t=t, y=gcvi, min_observations=MIN_OBSERVATIONS
        )
    return ret


def lonlat(pixels: mpc.Pixels) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Crop-pixel centres in degrees, in the order of `pixels.green`'s rows."""
    rows, cols = numpy.nonzero(pixels.crop)
    xs, ys = pixels.on.transform @ (cols + 0.5, rows + 0.5)
    return pyproj.Transformer.from_crs(
        pixels.on.crs, "EPSG:4326", always_xy=True
    ).transform(xs, ys)


def apply(
    model: QDANN, saved: dict, features: pandas.DataFrame
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """q50, lo, hi in t/ha per row; NaN where any feature is missing."""
    x = features[saved["features"]].to_numpy()
    ok = numpy.isfinite(x).all(axis=1)
    x = torch.tensor(
        (x[ok] - saved["feature_mean"]) / saved["feature_std"], dtype=torch.float32
    )
    with torch.no_grad():
        yhat, _, bounds = model(x)
    scale = lambda v: (v * saved["label_std"] + saved["label_mean"]).numpy()  # noqa: E731
    q50, q10, q90 = scale(yhat), scale(bounds[:, 0]), scale(bounds[:, 1])
    margin = saved["conformal_margin"]
    out = numpy.full((3, len(features)), numpy.nan)
    out[0, ok] = q50
    out[1, ok] = numpy.minimum(q10, q50) - margin
    out[2, ok] = numpy.maximum(q90, q50) + margin
    return out[0], out[1], out[2]


def county_inputs(long: pandas.DataFrame, crop: str) -> pandas.DataFrame:
    """Eq. 2 on each county's mean series, by fips, as `mpc.fit` builds the source table
    (sliver dates dropped, the rank floor)."""
    long = long[long["n_clear"] >= mpc.SLIVER * long["n_crop"]]
    fitted = gee.fit_table(long=gee.clean(long), crop=crop)
    return fitted.set_index("fips")[list(weather.HARMONIC_COLUMNS)]


def save_features(
    features: pandas.DataFrame, pixels: mpc.Pixels, path: pathlib.Path
) -> None:
    """~80 MB per county-year: float32 features in the order of `pixels.crop`'s True
    cells, the mask, and the grid it lies on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(
        path,
        features=features.to_numpy(numpy.float32),
        columns=numpy.array(features.columns, dtype=str),
        crop=pixels.crop,
        transform=numpy.array(tuple(pixels.on.transform)[:6]),
        crs=pixels.on.crs.to_wkt(),
    )


def load_features(path: pathlib.Path) -> tuple[pandas.DataFrame, numpy.ndarray, mpc.Grid]:
    """`save_features`'s features, crop mask and grid."""
    with numpy.load(path) as f:
        crop = f["crop"]
        grid = mpc.Grid(
            crs=pyproj.CRS.from_wkt(str(f["crs"])),
            transform=Affine(*f["transform"]),
            width=crop.shape[1],
            height=crop.shape[0],
        )
        return pandas.DataFrame(f["features"], columns=f["columns"].tolist()), crop, grid


def write_cog(bands: numpy.ndarray, pixels: mpc.Pixels, path: pathlib.Path) -> None:
    grid = numpy.full((len(bands), pixels.on.height, pixels.on.width), numpy.nan, "float32")
    grid[:, pixels.crop] = bands
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=pixels.on.width,
        height=pixels.on.height,
        count=len(bands),
        dtype="float32",
        crs=pixels.on.crs,
        transform=pixels.on.transform,
        nodata=numpy.nan,
        compress="deflate",
    ) as dst:
        dst.write(grid)
        dst.descriptions = ("q50", "lo", "hi")


def cell_sums(
    pixels: mpc.Pixels, *weights: numpy.ndarray
) -> tuple[numpy.ndarray, list[numpy.ndarray]]:
    """`export.cell_counts`'s cells, with each weight summed per cell."""
    cells, inverse = numpy.unique(export.cell_index(pixels.crop, pixels.on), return_inverse=True)
    return cells, [numpy.bincount(inverse, w, minlength=len(cells)) for w in weights]


def frames(
    county: dict[str, tuple[mpc.Pixels, numpy.ndarray, numpy.ndarray, numpy.ndarray]],
    stats: pandas.DataFrame,
    districts: dict[str, str],
) -> dict[str, numpy.ndarray]:
    """The 5' grids for one crop-year from each county's pixels and (q50, lo, hi)."""
    counts, yields, lo, hi = {}, {}, {}, {}
    for fips, (pixels, q50, low, high) in county.items():
        fill = lambda v: numpy.where(numpy.isfinite(v), v, numpy.nanmean(v))  # noqa: E731
        ones = numpy.ones(len(q50))
        cells, (n, y, l, h) = cell_sums(pixels, ones, fill(q50), fill(low), fill(high))
        counts[fips], yields[fips] = (cells, n), (cells, y)
        lo[fips], hi[fips] = (cells, l), (cells, h)
    base = export.spread(counts, stats, districts)
    anchored = export.spread(yields, stats, districts)
    size = export.FRAME[0] * export.FRAME[1]
    n, y, l, h = (numpy.zeros(size) for _ in range(4))
    for fips in counts:
        numpy.add.at(n, counts[fips][0], counts[fips][1])
        numpy.add.at(y, yields[fips][0], yields[fips][1])
        numpy.add.at(l, lo[fips][0], lo[fips][1])
        numpy.add.at(h, hi[fips][0], hi[fips][1])
    with numpy.errstate(invalid="ignore", divide="ignore"):
        mean = lambda v: numpy.where(n > 0, v / n, 0.0).reshape(export.FRAME)  # noqa: E731
        return {
            "raw_P": base["H"] * mean(y),
            "anchored_P": anchored["P"],
            "H": base["H"],
            "cov": base["cov"],
            "Ylo": mean(l),
            "Yhi": mean(h),
        }


def bytes_read() -> int:
    """Bytes received by the machine's network interfaces: /proc/self/io misses libcurl's
    socket reads, so this counts everything, and is only a measure on an otherwise idle
    connection."""
    lines = pathlib.Path("/proc/net/dev").read_text().splitlines()[2:]
    return sum(
        int(rest.split()[0]) for name, rest in (line.split(":", 1) for line in lines)
        if name.strip() != "lo"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(export.CROP_TO_SPAM), default="maize")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--county", nargs="+", required=True, help="five-digit FIPS")
    parser.add_argument("--out", type=pathlib.Path, default=gee.DATA_DIR / "predict")
    args = parser.parse_args()

    model, saved = load(args.crop)
    state = args.county[0][:2]
    assert all(f[:2] == state for f in args.county), "one state per run"
    counties = mpc.state_counties(state, tuple(args.county))
    started, read_before = time.monotonic(), bytes_read()
    with rasterio.Env(**mpc.GDAL_ENV):
        long, _, _, series = mpc.county_means(
            crop=args.crop, year=args.year, counties=counties, full=True
        )
    read_s, read_bytes = time.monotonic() - started, bytes_read() - read_before
    logger.info("read %d dates in %.0f s, %.2f GB", long["date"].nunique(), read_s, read_bytes / 1e9)

    region = shapely.union_all(list(counties.values()))
    cells = weather.cell_means(crop=args.crop, year=args.year, bounds=region.bounds)
    labels = weather.read(weather.DATA_DIR / f"nass_{args.crop}.parquet")
    labels = labels[labels["year"] == args.year].set_index("fips")[weather.LABEL_COLUMN]
    report, county = {}, {}
    county_harmonics = county_inputs(long, crop=args.crop)
    for fips, pixels in series.items():
        t0 = time.monotonic()
        features = pandas.DataFrame(
            pixel_harmonics(pixels, crop=args.crop, year=args.year),
            columns=list(weather.HARMONIC_COLUMNS),
        ).join(cells.at(*lonlat(pixels)))
        save_features(
            features[saved["features"]],
            pixels,
            args.out / fips / f"features_{args.crop}_{args.year}.npz",
        )
        q50, lo, hi = apply(model, saved, features)
        write_cog(
            numpy.stack([q50, lo, hi]),
            pixels,
            args.out / fips / f"qdann_{args.crop}_{args.year}.tif",
        )
        # the county row's weather is the mean over its crop pixels, against the source
        # table's mean over gridMET cells (weather.county_means): near equal in crop counties
        row = county_harmonics.loc[[fips]].reset_index(drop=True).join(
            features[list(weather.weather_columns(args.crop))].mean().to_frame().T
        )
        fill = [v[0] for v in apply(model, saved, row)]
        if numpy.isnan(fill[0]):
            fill = [numpy.nanmean(v) for v in (q50, lo, hi)]
        county[fips] = (
            pixels,
            *(numpy.where(numpy.isfinite(v), v, f) for v, f in zip((q50, lo, hi), fill)),
        )
        report[fips] = {
            "crop_pixels": len(q50),
            "predicted_share": float(numpy.isfinite(q50).mean()),
            "mean_q50": float(numpy.nanmean(q50)),
            "county_q50": float(fill[0]),
            "filled_mean_q50": float(county[fips][1].mean()),
            "nass_yield": float(labels.get(fips, numpy.nan)),
            "pixel_sd_q50": float(numpy.nanstd(q50)),
            "mean_interval_width": float(numpy.nanmean(hi - lo)),
            "fit_and_predict_s": time.monotonic() - t0,
        }
        logger.info("%s: %s", fips, report[fips])

    area_rows, production_rows = nass.fetch_rows(
        crop=args.crop, years=export.DISTRICT_YEARS, state_fips=(state,)
    )
    stats = nass.totals(area_rows=area_rows, production_rows=production_rows, crop=args.crop)
    districts = stats.attrs["districts"]
    stats = stats[(stats["year"] == args.year) & stats["fips"].isin(list(counties))]
    spam = export.CROP_TO_SPAM[args.crop]
    for name, grid in frames(county, stats, districts).items():
        prefix = {"raw_P": "qdann-raw", "anchored_P": "qdann-anchored"}.get(name, "qdann")
        export.write(grid, args.out / f"{prefix}_{args.year}_{name.split('_')[-1]}_{spam}.tif")

    summary = {
        "counties": report,
        "dates": int(long["date"].nunique()),
        "read_s": read_s,
        # machine-wide, see bytes_read
        "received_gb": read_bytes / 1e9,
        "total_s": time.monotonic() - started,
        # kilobytes on Linux
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    path = args.out / f"predict_{args.crop}_{args.year}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
