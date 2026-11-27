"""Stage 1 gate (notes/yield_layer_plan.md): does the 5' raw layer place a held-out county's
production better than the Stage A fallback a withheld county gets today?

  uv run python data/gate/gate.py train            # leave-one-county-out heads
  uv run python data/gate/gate.py score            # -> data/gate/gate.json

GATE_TABLE=<name> runs the same gate on data/source_mpc_maize_<name>.parquet, with that
table's tune_maize_<name>.json and conformal margin; heads are qdann_maize_gate_<name>_no*
and results gate_f<floor>_<name>.json. Years past the table's last year count as unseen.

Models: the shipped recipe (plain head, MPC source table, DA off, Stage 0 dropout/epochs and
conformal margin), refit without one county, seeds 0-2, as data/models/qdann_maize_gate_no<fips>_s<seed>.pt.

Per county-year: raw yield = mean over crop pixels of q50, unpredicted pixels taking the
county's mean prediction, which is what the raw P sums to over the county (H is NASS area
spread over CDL pixels, so sum(P)/NASS area = pixel mean). Baselines: the district's
yield without the county (Stage A's fallback for a withheld county), and the state's.
Years in the source table (2008-2018) test a new county; 2019/2021 also test a new year,
scored with the LOCO heads (county and year unseen) and with the shipped head (year unseen).
"""
import concurrent.futures, json, os, pathlib, sys

import numpy, pandas, torch

from qdann import export, gee, harmonics, mpc, nass, predict, tune, weather

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import read  # noqa: E402

D = weather.DATA_DIR
HERE = pathlib.Path(__file__).parent
SEEDS = (0, 1, 2)
NAME = os.environ.get("GATE_TABLE")
TAG = f"_{NAME}" if NAME else ""
SOURCE = D / f"source_mpc_maize{TAG}.parquet"
TUNED = json.loads((D / f"tune_maize{TAG}.json").read_text())
MARGIN = (TUNED["with_intervals"] if NAME else
          json.loads((D / "stage1" / "ship_mpc.json").read_text()))["conformal_margin"]
LAST_SEEN = 2018 if not NAME else int(pandas.read_parquet(SOURCE, columns=["year"])["year"].max())


def model_path(fips: str, seed: int) -> pathlib.Path:
    return D / "models" / f"qdann_maize_gate{TAG}_no{fips}_s{seed}.pt"


def fit(job):
    fips, seed = job
    if model_path(fips, seed).exists():
        return f"skip {fips} s{seed}"
    torch.set_num_threads(1)
    table = weather.read(SOURCE)
    tune.save(table[table["fips"] != fips], "maize", TUNED["chosen"]["dropout"],
              TUNED["chosen"]["epochs"], MARGIN, seed=seed, name=f"gate{TAG}_no{fips}_s{seed}")
    return f"{fips} s{seed}"


def counties() -> list[str]:
    """Every county with a series on disk or a Step 5 feature cache."""
    have = {p.name.split("_")[0] for p in (HERE / "series").glob("*_*.npz") if ".part" not in p.name}
    have |= {p.parent.name for p in (D / "compare" / "predict").glob("g*/*/features_maize_*.npz")}
    return sorted(have)


def features(fips: str, year: int, floor: int = 20) -> pandas.DataFrame | None:
    """(crop pixels, 27) features plus n_obs, from the raw series at `floor`, else Step 5's
    cache (floor 20 only). Each pixel's fit doesn't depend on the floor, so the series is fit
    once at the rank floor (7) and a higher floor is a mask on n_obs."""
    raw = all_features(fips, year)
    if raw is not None:
        out = raw.copy()
        out.loc[out["n_obs"] < floor, list(weather.HARMONIC_COLUMNS)] = numpy.nan
        return out
    hits = list((D / "compare" / "predict").glob(f"g*/{fips}/features_maize_{year}.npz"))
    return predict.load_features(hits[0])[0] if floor == 20 and hits else None


def all_features(fips: str, year: int) -> pandas.DataFrame | None:
    cache = HERE / "features" / f"{fips}_{year}.parquet"
    if cache.exists():
        return pandas.read_parquet(cache)
    if read.path(fips, year).exists():
        px = read.load(fips, year)
        t = gee.normalized_time(pandas.Series(px.dates), crop="maize", year=year).to_numpy()
        coef = numpy.empty((len(px.green), 7))
        n_obs = numpy.empty(len(px.green), int)
        for i in range(0, len(coef), predict.CHUNK):
            y = gcvi(px.green[i : i + predict.CHUNK], px.nir[i : i + predict.CHUNK])
            coef[i : i + predict.CHUNK] = harmonics.fit_many(t=t, y=y, min_observations=7)
            n_obs[i : i + predict.CHUNK] = numpy.isfinite(y).sum(axis=1)
        lon, lat = predict.lonlat(px)
        bounds = (lon.min(), lat.min(), lon.max(), lat.max())
        cells = weather.cell_means(crop="maize", year=year, bounds=bounds)
        out = pandas.DataFrame(coef, columns=list(weather.HARMONIC_COLUMNS)).join(cells.at(lon, lat))
        out["n_obs"] = n_obs
    else:
        return None
    cache.parent.mkdir(exist_ok=True)
    out.to_parquet(cache)
    return out


def gcvi(green: numpy.ndarray, nir: numpy.ndarray) -> numpy.ndarray:
    """Eq. 1, NaN where gee.clean would drop the date (as predict.pixel_harmonics)."""
    g, n = green.astype(float), nir.astype(float)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return numpy.where(g > gee.MIN_GREEN_REFLECTANCE, n / g - 1, numpy.nan)


def nass_totals() -> pandas.DataFrame:
    path = HERE / "nass_ia.parquet"
    if not path.exists():
        a, p = nass.fetch_rows(crop="maize", years=export.DISTRICT_YEARS, state_fips=("19",))
        t = nass.totals(area_rows=a, production_rows=p, crop="maize")
        dist = t.attrs["districts"]
        t["asd"] = [a if a else dist.get(f) for f, a in zip(t["fips"], t["asd"])]
        t.to_parquet(path)
    return pandas.read_parquet(path)


def baselines(t: pandas.DataFrame, fips: str, year: int) -> dict:
    yr = t[t["year"] == year]
    me = yr[yr["fips"] == fips]
    if me.empty:
        return {}
    others = yr[yr["fips"] != fips]
    district = others[others["asd"] == me["asd"].iloc[0]]
    y = lambda d: float(d["production_t"].sum() / d["area_ha"].sum())  # noqa: E731
    return {"nass": y(me), "area_ha": float(me["area_ha"].iloc[0]), "asd": me["asd"].iloc[0],
            "district": y(district), "state": y(others)}


def county_yield(path: pathlib.Path, f: pandas.DataFrame) -> tuple[float, float]:
    """Filled pixel mean of q50 and the predicted share."""
    model, saved = predict.load("maize", path)
    q50 = predict.apply(model, saved, f)[0]
    return float(numpy.nanmean(q50)), float(numpy.isfinite(q50).mean())


def score(floor: int = 20) -> list[dict]:
    t = nass_totals()
    table = weather.read(SOURCE).set_index(["fips", "year"])
    rows = []
    for fips in counties():
        for year in range(2016, 2024):
            f = features(fips, year, floor)
            b = baselines(t, fips, year)
            if f is None or not b:
                continue
            loco = [county_yield(model_path(fips, s), f) for s in SEEDS if model_path(fips, s).exists()]
            if not loco:
                continue
            row = {"fips": fips, "year": year, **b, "predicted_share": loco[0][1],
                   "raw": float(numpy.mean([v for v, _ in loco])),
                   "raw_seeds": [v for v, _ in loco]}
            if numpy.isnan(row["raw"]):
                row["raw"] = None
            if year > 2018:  # the 2008-2018 shipped head: year unseen
                row["shipped"] = county_yield(D / "models" / "qdann_maize.pt", f)[0]
            if (fips, year) in table.index:
                # the same LOCO heads on the county's mean features: pixel vs county scale
                one = table.loc[[(fips, year)]].reset_index()
                row["county_features"] = float(numpy.mean(
                    [predict.apply(*predict.load("maize", model_path(fips, s)), one)[0][0] for s in SEEDS]))
                # unpredicted pixels take the county-feature prediction instead of the
                # predicted pixels' mean
                share = row["predicted_share"]
                row["raw_fill"] = (row["county_features"] if row["raw"] is None else
                                   share * row["raw"] + (1 - share) * row["county_features"])
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "raw_seeds"}), flush=True)
    return rows


def summarize(rows: list[dict], only_year: int | None = None) -> dict:
    df = pandas.DataFrame(rows)
    df = df[df["raw"].notna()]
    if only_year:
        df = df[df["year"] == only_year]
    out = {}
    for name, part in (("seen_years", df[df["year"] <= LAST_SEEN]), ("unseen_years", df[df["year"] > LAST_SEEN])):
        if part.empty:
            continue
        s = {"n": len(part), "counties": int(part["fips"].nunique())}
        for col in ("raw", "raw_fill", "shipped", "county_features", "district", "state"):
            if col not in part or part[col].isna().all():
                continue
            p = part[part[col].notna()]
            err = p[col] - p["nass"]
            # within-year: each year's mean removed from both, so level is not scored
            dm = lambda v: v - p.groupby("year")[v.name].transform("mean")  # noqa: E731
            r = float(numpy.corrcoef(dm(p[col]), dm(p["nass"]))[0, 1]) if col not in ("state",) else None
            # lab Phase 0 style: share of the test counties' production placed differently,
            # per year, with every method using NASS area
            placed = []
            for _, g in p.groupby("year"):
                truth = g["nass"] * g["area_ha"]
                est = g[col] * g["area_ha"]
                placed.append(0.5 * float((est / est.sum() - truth / truth.sum()).abs().sum()))
            s[col] = {"rmse": float(numpy.sqrt((err ** 2).mean())), "bias": float(err.mean()),
                      "mae": float(err.abs().mean()), "within_year_r": r,
                      "misplaced_share": float(numpy.mean(placed)), "n": len(p)}
        out[name] = s
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "train":
        jobs = [(f, s) for s in SEEDS for f in (sys.argv[2:] or counties())]
        with concurrent.futures.ProcessPoolExecutor(2) as pool:
            for line in pool.map(fit, jobs):
                print(line, flush=True)
    elif what == "score":
        floor = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        rows = score(floor)
        years = sorted({r["year"] for r in rows})
        result = {"floor": floor, "source": str(SOURCE), "summary": summarize(rows),
                  "by_year": {y: summarize(rows, y) for y in years}, "rows": rows}
        (HERE / f"gate_f{floor}{TAG}.json").write_text(json.dumps(result, indent=2, default=float))
        print(json.dumps(result["summary"], indent=2, default=float))
