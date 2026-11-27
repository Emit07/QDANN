"""Overnight 2026-11-25: save every crop pixel's raw green/NIR series per county-year, so
the Stage 1 gate and the observation-floor study run offline at any floor.

  uv run python data/gate/read.py --year 2018 --county 19169 19015 19153

Writes data/gate/series/<fips>_<year>.npz: green, nir (crop pixels, dates) float16, NaN
where masked; dates; the crop mask and its grid. Skips county-years already on disk.
"""
import argparse, json, pathlib, time

import numpy, rasterio

from qdann import mpc, predict

OUT = pathlib.Path(__file__).parent / "series"


def path(fips: str, year: int) -> pathlib.Path:
    return OUT / f"{fips}_{year}.npz"


def load(fips: str, year: int) -> mpc.Pixels:
    with numpy.load(path(fips, year)) as f:
        crop = f["crop"]
        grid = mpc.Grid(
            crs=mpc.pyproj.CRS.from_wkt(str(f["crs"])),
            transform=rasterio.Affine(*f["transform"]),
            width=crop.shape[1],
            height=crop.shape[0],
        )
        return mpc.Pixels(grid, crop, f["dates"], f["green"].astype(numpy.float32),
                          f["nir"].astype(numpy.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--county", nargs="+", required=True)
    # the 2008-2023 table leaves Landsat 7 out after 2021 (#30): match its reader (#27)
    parser.add_argument("--landsat7-last", type=int)
    args = parser.parse_args()
    todo = tuple(f for f in args.county if not path(f, args.year).exists())
    if not todo:
        print("skip", args.year, args.county)
        return 0
    counties = mpc.state_counties(todo[0][:2], todo)
    t0, b0 = time.monotonic(), predict.bytes_read()
    with rasterio.Env(**mpc.GDAL_ENV):
        long, _, _, series = mpc.county_means(crop="maize", year=args.year,
                                              counties=counties, full=True,
                                              landsat7_last=args.landsat7_last)
    OUT.mkdir(parents=True, exist_ok=True)
    for fips, px in series.items():
        part = path(fips, args.year).with_suffix(".part.npz")
        numpy.savez(part, green=px.green.astype(numpy.float16), nir=px.nir.astype(numpy.float16),
                    dates=px.dates, crop=px.crop,
                    transform=numpy.array(tuple(px.on.transform)[:6]), crs=px.on.crs.to_wkt())
        part.rename(path(fips, args.year))
    log = {"year": args.year, "counties": list(todo), "landsat7_last": args.landsat7_last, "dates": int(long["date"].nunique()),
           "read_s": time.monotonic() - t0, "received_gb": (predict.bytes_read() - b0) / 1e9}
    with open(OUT / "reads.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    print(log, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
