"""Step 5 arms, each trained on every county and saved as data/models/qdann_maize_<arm>_s<seed>.pt.

  uv run python data/compare/arms.py [--holdout-year 2016] [--seeds 0 1 2]

The no-target head is the shipped data/models/qdann_maize.pt (tune.save, seed 0).
--holdout-year drops that year's county labels (not its target pixels, as in step 4's
leave-one-year-out) and also trains the no-target head, as <arm>_no<year>_s<seed>.
Settings and the conformal margin are Stage 0's, from data/tune_maize.json.
"""
import argparse, concurrent.futures, json, time
from qdann import tune, weather

d = weather.DATA_DIR
ARMS = {  # name: (adversarial, vae_filter); no_target trains without the target
    "no_target": (False, False),
    "source_only": (False, False),
    "full": (True, True),
    "no_vae": (True, False),
}
tuned = json.loads((d / "tune_maize.json").read_text())
source = weather.read(d / "source_maize.parquet")
target = weather.read(d / "target_mpc_maize.parquet")


def run(job):
    arm, seed, year = job
    name = f"{arm}_s{seed}" if year is None else f"{arm}_no{year}_s{seed}"
    path = d / "models" / f"qdann_maize_{name}.pt"
    if path.exists():
        return f"skip {path.name}"
    t0 = time.time()
    adversarial, vae_filter = ARMS[arm]
    tune.save(source if year is None else source[source["year"] != year], "maize",
              tuned["chosen"]["dropout"], tuned["chosen"]["epochs"],
              tuned["with_intervals"]["conformal_margin"],
              target=None if arm == "no_target" else target,
              adversarial=adversarial, vae_filter=vae_filter, seed=seed, name=name)
    return f"{path.name} in {time.time() - t0:.0f} s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--holdout-year", type=int)
    args = parser.parse_args()
    # the all-years seed-0 no-target head is the shipped qdann_maize.pt
    jobs = [(a, s, args.holdout_year) for s in args.seeds for a in ARMS
            if (a, s, args.holdout_year) != ("no_target", 0, None)]
    with concurrent.futures.ProcessPoolExecutor(3) as pool:
        for line in pool.map(run, jobs):
            print(line, flush=True)
