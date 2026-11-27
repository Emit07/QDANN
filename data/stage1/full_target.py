"""Stage 1 step 4 on the laptop: the Fig. 16 ablation and leave-one-year-out with the full
Iowa target (28,388 pixel-years from data/mpc/19), 1000 epochs, dropout 0.1, 5 seeds.

  uv run python data/stage1/full_target.py

Every finished fit is appended to full_target.jsonl and skipped on a rerun.
"""
import concurrent.futures, json, logging, os, pathlib, statistics, time
from qdann import ablation, train, weather

HERE = pathlib.Path(__file__).parent
LOG = HERE / "full_target.jsonl"
EPOCHS, SEEDS = 1000, range(5)
ARMS = ("none", *ablation.COMPONENTS, "source_only", "no_target")
d = weather.DATA_DIR
source = weather.read(d / "source_maize.parquet")
target = weather.read(d / "target_mpc_maize.parquet")
years = sorted(source["year"].unique().tolist())


def run(job):
    kind, arg, seed = job
    t0 = time.time()
    # "source_only": no adversarial branch and no VAE filter, the plain transfer baseline
    # "loyo" is the full model; "loyo:<arm>" drops <arm> as the ablation does
    arm = arg if kind == "ablation" else kind.partition(":")[2] or "none"
    # "no_target": the Stage 0 head, trained without the target and scored on its pixels
    both = ("adversarial", "vae_filter")
    off = {"none": (), "source_only": both, "no_target": both}.get(arm, (arm,))
    switches = dict.fromkeys(ablation.COMPONENTS, True) | dict.fromkeys(off, False)
    year = None if kind == "ablation" else arg
    r = train.evaluate(source, EPOCHS, seed=seed, target=target, crop="maize",
                       holdout_year=year, mix_target=arm != "no_target", **switches)
    return {"job": job, "result": r, "seconds": time.time() - t0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    done = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            row = json.loads(line)
            done[tuple(row["job"])] = row
    jobs = [("ablation", arm, s) for arm in ARMS for s in SEEDS]
    LOYO = ("loyo", "loyo:source_only", "loyo:vae_filter")
    jobs += [(k, y, s) for k in LOYO for y in years for s in SEEDS]
    todo = [j for j in jobs if j not in done]
    print(f"{len(todo)} of {len(jobs)} fits to run", flush=True)
    with concurrent.futures.ProcessPoolExecutor(os.process_cpu_count() or 1) as pool:
        for row in pool.map(run, todo):
            done[tuple(row["job"])] = row
            with LOG.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("done", row["job"], f"{row['seconds']:.0f} s", flush=True)

    summary = {"ablation": {}}
    metrics = ("r2", "within_year_r2", "pixel_county_rmse", "domain_accuracy")
    for arm in ARMS:
        runs = [done[("ablation", arm, s)]["result"] for s in SEEDS]
        summary["ablation"][arm] = {m: statistics.mean(r[m] for r in runs) for m in metrics if m in runs[0]}
    for k in LOYO:
        pooled, within = [], []
        for s in SEEDS:
            held = train.split(source, seed=s)[1][weather.LABEL_COLUMN].to_numpy()
            runs = [done[(k, y, s)]["result"] for y in years]
            sse = sum(r["rmse"] ** 2 * r["n_test"] for r in runs)
            pooled.append(1 - sse / float(((held - held.mean()) ** 2).sum()))
            within.append(statistics.mean(r["r2"] for r in runs))
        summary[k] = {
            "r2": statistics.mean(pooled), "within_year_r2": statistics.mean(within),
            "per_year_r2": {y: statistics.mean(done[(k, y, s)]["result"]["r2"] for s in SEEDS) for y in years},
        }
    summary["mean_fit_seconds"] = statistics.mean(r["seconds"] for r in done.values())
    (HERE / "full_target.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
