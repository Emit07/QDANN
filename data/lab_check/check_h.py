"""Stage A cross-check: our CDL maize area per 5' cell vs the lab's build_cdl_5min.py grids.

  uv run python data/lab_check/check_h.py 2008 2013 2018

Per year: download NASS's national CDL zip, run the lab's script (copied in pipeline/, strip
size 1024 rows instead of 4096 so it fits in 7 GB; results identical), then count our CDL
maize pixels per cell exactly as export.py does (MPC CDL tiles, each county on its UTM
Landsat grid, pixel centre binning) and compare on Iowa's cells.
"""
import json, pathlib, subprocess, sys, time, urllib.request
import numpy, shapely
from qdann import export, mpc

HERE = pathlib.Path(__file__).parent
URL = "https://www.nass.usda.gov/sites/default/files/Research_and_Science/Cropland/Release/datasets/{y}_30m_cdls.zip"
OUT = HERE / "check_h.json"
results = json.loads(OUT.read_text()) if OUT.exists() else {}
for year in map(int, sys.argv[1:]):
    if str(year) in results:
        continue
    t0 = time.time()
    cdl = HERE / "data/raw/crop_masks/cdl"
    zipped = cdl / f"{year}_30m_cdls.zip"
    if not zipped.exists() and not (cdl / f"{year}_30m_cdls.tif").exists():
        # NASS's server answers 500 now and then
        for attempt in range(6):
            try:
                urllib.request.urlretrieve(URL.format(y=year), zipped.with_suffix(".part"))
                break
            except OSError as error:
                if attempt == 5:
                    raise
                print(f"{year}: {error}; retrying in {60 * 2**attempt} s", flush=True)
                time.sleep(60 * 2**attempt)
        zipped.with_suffix(".part").rename(zipped)
    lab_path = HERE / f"data/processed/cdl_{year}_5min_ha.npz"
    if not lab_path.exists():
        subprocess.run([sys.executable, str(HERE / "pipeline/build_cdl_5min.py"), str(year)], check=True)
    lab = numpy.load(lab_path)["MAIZ"].astype(float).ravel()

    counties = mpc.state_counties("19")
    iowa = shapely.union_all(list(counties.values()))
    ours = numpy.zeros(export.FRAME[0] * export.FRAME[1])
    with __import__("rasterio").Env(**mpc.GDAL_ENV):
        tiles = mpc.cdl_tiles(year=year, counties=counties)
        for fips, polygon in counties.items():
            on = mpc.grid(polygon)
            cells, counts = export.cell_counts(mpc.cropland("maize", polygon, on, tiles), on)
            numpy.add.at(ours, cells, counts * export.PIXEL_HA)
    # cells wholly inside Iowa: the lab's border cells also hold the neighbour states' maize
    touched = numpy.flatnonzero((ours > 0) | (lab > 0))
    rows, cols = numpy.divmod(touched, export.FRAME[1])
    d = 1 / export.CELLS_PER_DEGREE
    boxes = shapely.box(cols * d - 180, 90 - (rows + 1) * d, (cols + 1) * d - 180, 90 - rows * d)
    interior = touched[shapely.within(boxes, iowa)]
    a, b = ours[interior], lab[interior]
    big = b > 100
    results[str(year)] = {
        "interior_cells": int(len(interior)),
        "ours_mha": a.sum() / 1e6, "lab_mha": b.sum() / 1e6, "ratio_ours_over_lab": a.sum() / b.sum(),
        "cell_corr": float(numpy.corrcoef(a, b)[0, 1]),
        "median_abs_rel_diff_cells_over_100ha": float(numpy.median(numpy.abs(a[big] - b[big]) / b[big])),
        "p95_abs_rel_diff_cells_over_100ha": float(numpy.quantile(numpy.abs(a[big] - b[big]) / b[big], 0.95)),
        "minutes": (time.time() - t0) / 60,
    }
    print(year, json.dumps(results[str(year)]), flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    # the 1.8 GB zip and the extracted tif are only inputs
    for f in (zipped, cdl / f"{year}_30m_cdls.tif"):
        f.unlink(missing_ok=True)
