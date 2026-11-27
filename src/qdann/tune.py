"""Stage 0 of notes/yield_layer_plan.md: pick the head's dropout and epochs, then score it.

  uv run python -m qdann.tune --crop maize --source data/source_mpc_maize.parquet

1. Grid over dropout x epochs on an inner split of each fold's training counties, so the
   held-out counties play no part in the choice. (0.5, 1000) is today's setting, for
   reference.
2. The chosen setting on the held-out counties, with and without the q10/q90 outputs, the
   interval calibrated by `train.conformal`.
3. Leave-one-year-out with the chosen setting: train without year Y, score held-out
   counties in year Y.
4. Ridge, random forest and the MSE DNN on the same folds.
5. One model on every county, saved with its scalers for Stage 1.

Each finished job is appended to `data/tune_<crop>.jsonl` and skipped on a rerun, so an
interrupted run resumes.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import statistics
import subprocess

import torch

from qdann import baselines, train, weather


SEEDS = range(5)
GRID = [(0.5, 1000)] + [(d, e) for d in (0.1, 0.2, 0.3) for e in (1000, 3000)]


def _job(table, kind: str, seed: int, dropout: float, epochs: int, extra: dict) -> dict:
    if kind == "baselines":
        return baselines.evaluate(table, epochs=epochs, seed=seed, dropout=dropout)
    return train.evaluate(table, epochs=epochs, seed=seed, dropout=dropout, **extra)


def _run(table, jobs: list[dict], done: dict, log: pathlib.Path) -> None:
    todo = [job for job in jobs if _key(job) not in done]
    print(f"{len(todo):d} of {len(jobs):d} jobs to run", flush=True)
    with concurrent.futures.ProcessPoolExecutor(os.process_cpu_count() or 1) as pool:
        futures = {
            pool.submit(
                _job,
                table,
                job["kind"],
                job["seed"],
                job["dropout"],
                job["epochs"],
                job.get("extra", {}),
            ): job
            for job in todo
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future] | {"result": future.result()}
            done[_key(job)] = job
            with log.open("a") as f:
                f.write(json.dumps(job) + "\n")
            print(f"done {_key(job)}", flush=True)


def _key(job: dict) -> str:
    # the source table's sha256 is part of the key: a cached result for another table
    # must not be reused (lines written before it was keyed have none, and never match)
    keys = ("kind", "seed", "dropout", "epochs", "extra", "source")
    return json.dumps({k: job.get(k) for k in keys})


def _jobs(table, kind: str, dropout: float, epochs: int, **extra) -> list[dict]:
    source = table.attrs["source"]["sha256"]
    return [
        {"kind": kind, "seed": s, "dropout": dropout, "epochs": epochs, "extra": extra,
         "source": source}
        for s in SEEDS
    ]


def provenance(*tables) -> dict:
    """The code version and the input files behind an artifact: git commit, whether the
    tree had uncommitted changes, and each table's `weather.read` path and sha256."""
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], capture_output=True, text=True, check=True,
        cwd=pathlib.Path(__file__).parent,
    ).stdout.strip()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "inputs": [t.attrs.get("source", "unknown: not read with weather.read") for t in tables],
    }


def _mean(done: dict, jobs: list[dict], metric: str) -> float:
    return statistics.mean(done[_key(job)]["result"][metric] for job in jobs)


def save(
    table,
    crop: str,
    dropout: float,
    epochs: int,
    margin: float,
    target=None,
    adversarial: bool = False,
    vae_filter: bool = False,
    seed: int | tuple[int, ...] = 0,
    name: str | None = None,
    match_years: bool = False,
    overwrite: bool = False,
) -> pathlib.Path:
    """Every county, with the scalers `predict` will need to apply it to pixels.

    With `target`, its pixels are mixed into training and `adversarial` / `vae_filter`
    pick the arm, as in `train.evaluate`; `name` then keeps the arm's weights apart from
    the shipped `qdann_<crop>.pt`. `match_years` pairs each county-year with target pixels
    of its own year (`train.fit`).

    A tuple of seeds trains one head per seed and saves all their weights in one file;
    `predict.load` then averages their outputs (AMBIGUITIES #32).

    Refuses to replace an existing checkpoint unless `overwrite`: the default name is the
    shipped model. Writes `provenance` into the checkpoint and a `.json` beside it."""
    path = weather.DATA_DIR / "models" / (
        f"qdann_{crop}.pt" if name is None else f"qdann_{crop}_{name}.pt"
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    # before training: edits made while the heads train must not mark this code dirty
    started = provenance(table, *([] if target is None else [target]))
    features = train.feature_columns(table)
    x, *rest = train.standardize(
        table, *([] if target is None else [target]), columns=features
    )
    y = torch.tensor(table[weather.LABEL_COLUMN].to_numpy(), dtype=torch.float32)
    mean, std = float(y.mean()), float(y.std())
    y = (y - mean) / std
    years = torch.tensor(table["year"].to_numpy())
    state_dicts = []
    for s in (seed,) if isinstance(seed, int) else seed:
        xs, ys, yrs, weights = x, y, years, None
        if vae_filter:
            keep, weights = train.vae_weights(x, rest[0], epochs=epochs, seed=s, crop=crop)
            xs, ys, yrs = x[keep], y[keep], years[keep]
        model = train.fit(
            source_x=xs,
            source_y=ys,
            target_x=rest[0] if rest else None,
            sample_weights=weights,
            adversarial=adversarial,
            source_years=yrs if match_years else None,
            target_years=torch.tensor(target["year"].to_numpy()) if match_years else None,
            epochs=epochs,
            seed=s,
            dropout=dropout,
            interval_weight=1.0,
        )
        state_dicts.append(model.state_dict())
    path.parent.mkdir(exist_ok=True)
    meta = {
        "provenance": started,
        "arm": {"adversarial": adversarial, "vae_filter": vae_filter,
                "seed": seed if isinstance(seed, int) else list(seed),
                "match_years": match_years},
    }
    torch.save(
        {
            **({"state_dict": state_dicts[0]} if isinstance(seed, int) else {"state_dicts": state_dicts}),
            "features": features,
            "feature_mean": table[features].mean().to_numpy(),
            "feature_std": table[features].std().to_numpy(),
            "label_mean": mean,
            "label_std": std,
            "dropout": dropout,
            "epochs": epochs,
            # t/ha added to each side of [min(q10, point), max(q90, point)], the mean of
            # the held-out folds' conformal margins: county-scale, 80% nominal
            "conformal_margin": margin,
            **meta,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop", choices=sorted(weather.CROP_TO_MONTHS), default="maize"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="multiply every epoch count (smoke test)"
    )
    # no default: source_<crop>.parquet is the Earth Engine table, and the shipped model
    # is trained on the MPC one (AMBIGUITIES #27)
    parser.add_argument("--source", type=pathlib.Path, required=True, help="source table")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace data/models/qdann_<crop>.pt"
    )
    parser.add_argument(
        "--name", help="save as qdann_<crop>_<name>.pt and tune_<crop>_<name>.json instead"
    )
    args = parser.parse_args()
    table = weather.read(args.source)
    # first, not after hours of jobs: fails fast where git is missing (a container)
    started = provenance(table)
    suffix = "" if args.name is None else f"_{args.name}"
    shipped = weather.DATA_DIR / "models" / f"qdann_{args.crop}{suffix}.pt"
    if shipped.exists() and not args.overwrite:
        parser.error(f"{shipped} exists; pass --overwrite to replace it at the end")
    log = weather.DATA_DIR / f"tune_{args.crop}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            job = json.loads(line)
            done[_key(job)] = job
    grid = [(d, max(2, round(e * args.scale))) for d, e in GRID]

    print("\n== 1. grid on the inner split", flush=True)
    inner = {g: _jobs(table, "qdann", *g, inner=True) for g in grid}
    _run(table, [j for jobs in inner.values() for j in jobs], done, log)
    rows = {g: {m: _mean(done, jobs, m) for m in ("r2", "within_year_r2")} for g, jobs in inner.items()}
    for (d, e), row in rows.items():
        print(f"dropout {d:.1f} epochs {e:5d}  inner R2 {row['r2']:6.3f}  within-year {row['within_year_r2']:6.3f}")
    dropout, epochs = max(rows, key=lambda g: rows[g]["r2"])
    print(f"chosen: dropout {dropout} epochs {epochs}", flush=True)

    print("\n== 2-4. held-out folds, leave-one-year-out, baselines", flush=True)
    final = _jobs(table, "qdann", dropout, epochs, interval_weight=1.0)
    point = _jobs(table, "qdann", dropout, epochs)
    years = sorted(table["year"].unique().tolist())
    loyo = {y: _jobs(table, "qdann", dropout, epochs, holdout_year=y) for y in years}
    base = _jobs(table, "baselines", dropout, epochs)
    _run(table, final + point + [j for js in loyo.values() for j in js] + base, done, log)

    summary = {
        "inner_grid": {f"{d}_{e}": row for (d, e), row in rows.items()},
        "chosen": {"dropout": dropout, "epochs": epochs},
    }
    for name, jobs in (("with_intervals", final), ("point_only", point)):
        summary[name] = {
            m: _mean(done, jobs, m)
            for m in done[_key(jobs[0])]["result"]
            if not m.startswith("per_year_mean_nrmse")
        }
    # pooled over each fold's years, from per-year SSE, then averaged over folds
    pooled, within = [], []
    for s in SEEDS:
        held = train.split(table, seed=s)[1][weather.LABEL_COLUMN].to_numpy()
        runs = [done[_key(loyo[y][s])]["result"] for y in years]
        sse = sum(r["rmse"] ** 2 * r["n_test"] for r in runs)
        pooled.append(1 - sse / float(((held - held.mean()) ** 2).sum()))
        within.append(statistics.mean(r["r2"] for r in runs))
    summary["leave_one_year_out"] = {
        "r2": statistics.mean(pooled),
        "within_year_r2": statistics.mean(within),
    }
    summary["baselines"] = {
        name: {
            m: statistics.mean(done[_key(j)]["result"][name][m] for j in base)
            for m in ("r2", "within_year_r2", "rmse")
        }
        for name in ("ridge", "random_forest", "dnn")
    }
    margin = summary["with_intervals"]["conformal_margin"]
    summary["provenance"] = started
    summary["model_path"] = str(
        # three seeds, outputs averaged: the gate only supports the mean (#32, Ale 2026-11-27)
        save(table, args.crop, dropout, epochs, margin, seed=(0, 1, 2), name=args.name,
             overwrite=args.overwrite)
    )
    out = weather.DATA_DIR / f"tune_{args.crop}{suffix}.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
