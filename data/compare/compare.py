"""Step 5: score each arm's pixel map against SCYM (10 km) and Ma et al. QDANN (30 m).

  uv run python data/compare/compare.py [--seeds 0 1 2]

Both references are used to score only: nothing here feeds training, features or a
shipped layer. Inputs:
  data/compare/predict/g*/<fips>/features_maize_<year>.npz   (run_predict.sh)
  data/models/qdann_maize.pt (no_target), qdann_maize_<arm>_s<seed>.pt   (arms.py)
  data/scym/<year>_scym2020_20201107_10000m.tif   (Zenodo 4267589, CC-BY-4.0)
  data/compare/ma/<fips>_<year>.npy   Ma et al. b1 on our grid, fetched once from EE
                                      (CC-BY-NC-SA, non-commercial research use)
Writes data/compare/compare.json.

Every score is within county-year: each county-year's mean is removed from both maps
before r, SD ratio (ours / reference) and slope (ours on reference) are taken, so the
shared between-county and between-year signal cannot inflate the agreement.
"""
import argparse, json, pathlib, statistics
import numpy, pyproj, rasterio, shapely, shapely.ops
from qdann import mpc, predict

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent
MA = HERE / "ma"
MA_COLLECTION = "projects/sat-io/open-datasets/lobell-lab/VAE_QDANN_YIELD_MAP/CORN_SOY_MAP"
MIN_PREDICTED = 0.5   # share of a SCYM cell's crop pixels we predict
MIN_INSIDE = 0.9      # share of a SCYM cell's area inside one run county
BLOCK = 10            # 30 m pixels per side of the 300 m block score against Ma
SAMPLE = 0.1          # share of matched 30 m pixels kept for the pixel score


def models(seeds):
    """Every saved arm: <arm>_s<seed> on all years, <arm>_no<year>_s<seed> without one."""
    yield "no_target_s0", predict.load("maize")
    for path in sorted((DATA / "models").glob("qdann_maize_*_s*.pt")):
        name = path.stem.removeprefix("qdann_maize_")
        if int(name.rsplit("_s", 1)[1]) in seeds:
            yield name, predict.load("maize", path)


def caches():
    """{(fips, year): path} for every cached county-year."""
    return {
        (p.parent.name, int(p.stem.rsplit("_", 1)[1])): p
        for p in sorted(HERE.glob("predict/g*/*/features_maize_*.npz"))
    }


def centres(crop, grid, crs):
    rows, cols = numpy.nonzero(crop)
    xs, ys = grid.transform @ (cols + 0.5, rows + 0.5)
    return pyproj.Transformer.from_crs(grid.crs, crs, always_xy=True).transform(xs, ys)


def demeaned(groups, ours, ref):
    """r, SD ratio, slope after removing each group's mean from both; and raw r."""
    ours, ref, groups = map(numpy.asarray, (ours, ref, groups))
    a, b = ours.copy(), ref.copy()
    for g in numpy.unique(groups):
        m = groups == g
        a[m] -= a[m].mean()
        b[m] -= b[m].mean()
    r = float(numpy.corrcoef(a, b)[0, 1])
    return {
        "r": r,
        "sd_ratio": float(a.std() / b.std()),
        "slope": float((a * b).sum() / (b * b).sum()),
        "raw_r": float(numpy.corrcoef(ours, ref)[0, 1]),
        "mean_diff": float(ours.mean() - ref.mean()),
        "n": int(len(a)),
    }


def scym_cells(key, crop, grid, q50s, county):
    """Per kept SCYM cell: (cell id, SCYM t/ha, {arm: our mean q50})."""
    fips, year = key
    with rasterio.open(DATA / "scym" / f"{year}_scym2020_20201107_10000m.tif") as src:
        scym, transform, crs = src.read(1), src.transform, src.crs
    x, y = centres(crop, grid, crs)
    col, row = (~transform * (x, y))
    flat = numpy.floor(row).astype(int) * scym.shape[1] + numpy.floor(col).astype(int)
    to = pyproj.Transformer.from_crs("EPSG:4269", crs, always_xy=True).transform
    shape = shapely.ops.transform(to, county)
    out = []
    for cell in numpy.unique(flat):
        r, c = divmod(int(cell), scym.shape[1])
        if scym[r, c] <= 0:  # -999 nodata
            continue
        x0, y0 = transform * (c, r)
        x1, y1 = transform * (c + 1, r + 1)
        box = shapely.box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if shape.intersection(box).area < MIN_INSIDE * box.area:
            continue
        here = flat == cell
        means = {}
        for arm, q50 in q50s.items():
            v = q50[here]
            if numpy.isfinite(v).mean() < MIN_PREDICTED:
                break
            means[arm] = float(numpy.nanmean(v))
        else:
            out.append((int(cell), float(scym[r, c]), means))
    return out


def ma_pixels(key, crop, grid, q50s):
    """Pixels with both maps; and the same on BLOCK x BLOCK means."""
    ma = numpy.load(MA / f"{key[0]}_{key[1]}.npy")[crop] / 1000.0  # kg/ha -> t/ha
    ok = numpy.isfinite(ma) & (ma > 0)
    for q50 in q50s.values():
        ok &= numpy.isfinite(q50)
    rows, cols = numpy.nonzero(crop)
    block = (rows // BLOCK) * (grid.width // BLOCK + 1) + cols // BLOCK
    return ok, ma, block


def fetch_ma(key, crop, grid):
    """Ma et al. b1 for one county-year, resampled (nearest) onto our 30 m grid."""
    path = MA / f"{key[0]}_{key[1]}.npy"
    if path.exists():
        return
    import ee
    from qdann import env
    ee.Initialize(project=env.get("EE_PROJECT"))
    # native grid: EPSG:4326, 0.000269 deg (30 m N-S, 22 m E-W here), int kg/ha
    image = ee.Image(f"{MA_COLLECTION}/IA_corn_soy_map_{key[1]}").select("b1").toFloat()
    t = grid.transform
    array = mpc.retry(lambda: ee.data.computePixels({
        "expression": image.unmask(-1),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": grid.width, "height": grid.height},
            "affineTransform": {"scaleX": t.a, "shearX": t.b, "translateX": t.c,
                                "shearY": t.d, "scaleY": t.e, "translateY": t.f},
            "crsCode": f"EPSG:{grid.crs.to_epsg()}",
        },
    }))
    array = array["b1"].astype("float32")
    array[array < 0] = numpy.nan
    MA.mkdir(exist_ok=True)
    numpy.save(path, array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--skip-ma", action="store_true")
    args = parser.parse_args()
    loaded = dict(models(args.seeds))
    found = caches()
    counties = mpc.state_counties("19", tuple(sorted({f for f, _ in found})))
    scym_rows, ma_rows, ma_blocks = [], [], []  # (group, ours by arm, ref)
    per_county = {}
    for key, path in found.items():
        features, crop, grid = predict.load_features(path)
        q50s = {arm: predict.apply(m, s, features)[0] for arm, (m, s) in loaded.items()}
        group = f"{key[0]}_{key[1]}"
        cells = scym_cells(key, crop, grid, q50s, counties[key[0]])
        scym_rows += [(group, means, ref) for _, ref, means in cells]
        row = {"scym_cells": len(cells),
               "pixel_sd": {a: float(numpy.nanstd(q)) for a, q in q50s.items()}}
        if not args.skip_ma:
            fetch_ma(key, crop, grid)
            ok, ma, block = ma_pixels(key, crop, grid, q50s)
            row["ma_pixels"] = int(ok.sum())
            row["ma_r"] = {a: float(numpy.corrcoef(q[ok], ma[ok])[0, 1]) for a, q in q50s.items()}
            row["ma_sd"] = float(ma[ok].std())
            # block means now, and a fixed 10% of pixels: keeping every pixel for 24
            # models ran the laptop out of memory
            _, inv, n = numpy.unique(block[ok], return_inverse=True, return_counts=True)
            full = n >= BLOCK * BLOCK // 4
            means = lambda v: (numpy.bincount(inv, v) / n)[full]  # noqa: E731
            ma_blocks.append((group, {a: means(q[ok]) for a, q in q50s.items()}, means(ma[ok])))
            sample = numpy.random.default_rng(0).random(int(ok.sum())) < SAMPLE
            ma_rows.append((group, {a: q[ok][sample] for a, q in q50s.items()}, ma[ok][sample]))
        per_county[group] = row
        print(group, json.dumps(row), flush=True)

    summary = {"per_county_year": per_county, "scym": {}, "ma_30m": {}, "ma_300m": {}}
    years = sorted({k[1] for k in found})
    # the _no<year> arms never saw that year's labels: read them on that year's row
    for period in ["all", *map(str, years)]:
        pick = lambda g: period == "all" or g.endswith(period)  # noqa: E731
        srows = [r for r in scym_rows if pick(r[0])]
        mrows = [r for r in ma_rows if pick(r[0])]
        brows = [r for r in ma_blocks if pick(r[0])]
        for ref in ("scym", "ma_30m", "ma_300m"):
            summary[ref][period] = {}
        for arm in loaded:
            if srows:
                summary["scym"][period][arm] = demeaned(
                    [g for g, _, _ in srows], [m[arm] for _, m, _ in srows],
                    [r for _, _, r in srows])
            if not mrows:
                continue
            for ref, rows in (("ma_30m", mrows), ("ma_300m", brows)):
                summary[ref][period][arm] = demeaned(
                    numpy.repeat(numpy.arange(len(rows)), [len(r) for _, _, r in rows]),
                    numpy.concatenate([o[arm] for _, o, _ in rows]),
                    numpy.concatenate([r for _, _, r in rows]))
    # each arm over its seeds: mean, and min-max as the seed spread
    summary["by_arm"] = {}
    for ref in ("scym", "ma_30m", "ma_300m"):
        for period, arms in summary[ref].items():
            by = {}
            for name, v in arms.items():
                by.setdefault(name.rsplit("_s", 1)[0], []).append(v)
            summary["by_arm"].setdefault(ref, {})[period] = {
                arm: {m: [statistics.mean(v[m] for v in vs), min(v[m] for v in vs),
                          max(v[m] for v in vs)] for m in ("r", "sd_ratio", "mean_diff")}
                | {"seeds": len(vs), "n": vs[0]["n"]}
                for arm, vs in sorted(by.items())
            }
    out = HERE / "compare.json"
    out.write_text(json.dumps(summary, indent=2))
    for ref, periods in summary["by_arm"].items():
        for period, arms in periods.items():
            for arm, v in arms.items():
                (r, rlo, rhi), (sd, sdlo, sdhi), (d, _, _) = v["r"], v["sd_ratio"], v["mean_diff"]
                print(f"{ref:8s} {period:4s} {arm:20s} r {r:6.3f} [{rlo:.3f}-{rhi:.3f}]  "
                      f"sd ratio {sd:4.2f} [{sdlo:.2f}-{sdhi:.2f}]  diff {d:+5.2f}  "
                      f"seeds {v['seeds']}  n {v['n']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
