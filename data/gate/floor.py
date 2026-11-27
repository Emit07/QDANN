"""Overnight 2026-11-25: what does lowering the 20-date pixel floor (AMBIGUITIES #18) cost?

  uv run python data/gate/floor.py      # -> data/gate/floor.json

Core 2018 counties with a raw series and Ma et al.'s 30 m map; the shipped head.
1. Coverage: share of crop pixels predicted at floors 7-25.
2. Real sparse pixels: by n_obs bucket, within-county r and SD ratio against Ma, and the
   level offset from Ma relative to the county's well-observed (>= 20) pixels.
3. Mask transplant: dense pixels (>= 30 obs) given a real sparse pixel's observation mask
   from the same county, refit and predicted. Error against their own full-series q50 is
   pure fit noise from having fewer and gappier dates, with the pixel held fixed.
"""
import json, pathlib, sys

import numpy, pandas

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gate, read  # noqa: E402
from qdann import gee, harmonics, predict, weather  # noqa: E402

HERE = pathlib.Path(__file__).parent
MA = weather.DATA_DIR / "compare" / "ma"
YEAR = 2018
FLOORS = (7, 8, 10, 12, 15, 20, 25)
BUCKETS = [(7, 10), (10, 12), (12, 15), (15, 20), (20, 25), (25, 30), (30, 999)]
SAMPLE = 40_000
rng = numpy.random.default_rng(0)
model, saved = predict.load("maize")


def q50(f: pandas.DataFrame) -> numpy.ndarray:
    return predict.apply(model, saved, f)[0]


def stats(ours, ma):
    ok = numpy.isfinite(ours) & numpy.isfinite(ma)
    if ok.sum() < 100:
        return None
    a, b = ours[ok] - ours[ok].mean(), ma[ok] - ma[ok].mean()
    return {"n": int(ok.sum()), "r": float(numpy.corrcoef(a, b)[0, 1]),
            "sd_ratio": float(a.std() / b.std())}


def main():
    out = {"coverage": {}, "real": {}, "transplant": {}}
    pooled_real = {b: ([], []) for b in BUCKETS}
    pooled_tp = {b: [] for b in BUCKETS}
    for fips in gate.counties():
        if not (read.path(fips, YEAR).exists() and (MA / f"{fips}_{YEAR}.npy").exists()):
            continue
        f = gate.all_features(fips, YEAR)
        px = read.load(fips, YEAR)
        ma = numpy.load(MA / f"{fips}_{YEAR}.npy")[px.crop] / 1000.0
        ma[~(ma > 0)] = numpy.nan
        n = f["n_obs"].to_numpy()
        y = q50(f)
        out["coverage"][fips] = {fl: float(((n >= fl) & numpy.isfinite(y)).mean()) for fl in FLOORS}
        good = (n >= 20) & numpy.isfinite(y) & numpy.isfinite(ma)
        base_offset = float(numpy.mean(y[good] - ma[good])) if good.sum() > 100 else numpy.nan
        rows = {}
        for lo, hi in BUCKETS:
            m = (n >= lo) & (n < hi)
            s = stats(y[m], ma[m])
            if s:
                k = m & numpy.isfinite(y) & numpy.isfinite(ma)
                s["offset_vs_dense"] = float(numpy.mean(y[k] - ma[k]) - base_offset)
                s["share"] = float(m.mean())
                pooled_real[(lo, hi)][0].append(y[k] - y[k].mean())
                pooled_real[(lo, hi)][1].append(ma[k] - ma[k].mean())
            rows[f"{lo}-{hi}"] = s
        # the real spread of q50 within the county, to read the transplant RMSE against
        rows["dense_q50_sd"] = float(numpy.nanstd(y[n >= 30]))
        out["real"][fips] = rows

        # 3. transplant sparse masks onto dense pixels
        t = gee.normalized_time(pandas.Series(px.dates), crop="maize", year=YEAR).to_numpy()
        dense = numpy.flatnonzero((n >= 30) & numpy.isfinite(y))
        sparse = numpy.flatnonzero((n >= 7) & (n < 20))
        if len(dense) < 1000 or len(sparse) < 1000:
            continue
        i = rng.choice(dense, min(SAMPLE, len(dense)), replace=False)
        j = rng.choice(sparse, len(i))
        gi = gate.gcvi(px.green[i], px.nir[i])
        gj = gate.gcvi(px.green[j], px.nir[j])
        thin = numpy.where(numpy.isfinite(gj), gi, numpy.nan)
        coef = harmonics.fit_many(t=t, y=thin, min_observations=7)
        ft = f.iloc[i].copy()
        ft[list(weather.HARMONIC_COLUMNS)] = coef
        yt = q50(ft)
        kept = numpy.isfinite(thin).sum(axis=1)
        tp = {}
        for lo, hi in BUCKETS[:4]:
            m = (kept >= lo) & (kept < hi) & numpy.isfinite(yt)
            if m.sum() < 100:
                continue
            e = yt[m] - y[i][m]
            tp[f"{lo}-{hi}"] = {"n": int(m.sum()), "rmse_vs_full": float(numpy.sqrt((e ** 2).mean())),
                                "bias_vs_full": float(e.mean()),
                                "r_vs_full": float(numpy.corrcoef(yt[m], y[i][m])[0, 1]),
                                "unfit": float(numpy.isnan(yt[(kept >= lo) & (kept < hi)]).mean())}
            pooled_tp[(lo, hi)].append(e)
        out["transplant"][fips] = tp
        print(fips, json.dumps(out["coverage"][fips]), flush=True)

    out["pooled_real"] = {}
    for (lo, hi), (a, b) in pooled_real.items():
        if a:
            a, b = numpy.concatenate(a), numpy.concatenate(b)
            out["pooled_real"][f"{lo}-{hi}"] = {"n": len(a), "r": float(numpy.corrcoef(a, b)[0, 1]),
                                                "sd_ratio": float(a.std() / b.std())}
    out["pooled_transplant"] = {
        f"{lo}-{hi}": {"n": int(sum(len(e) for e in es)),
                       "rmse_vs_full": float(numpy.sqrt(numpy.mean(numpy.concatenate(es) ** 2))),
                       "bias_vs_full": float(numpy.mean(numpy.concatenate(es)))}
        for (lo, hi), es in pooled_tp.items() if es}
    (HERE / "floor.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("pooled_real", "pooled_transplant")}, indent=2))


if __name__ == "__main__":
    main()
