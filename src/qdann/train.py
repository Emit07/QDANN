"""Train the yield head on the county-level source table and score it against baselines.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.1. The training loop
itself is shared with `synth.py`, which exercises it on a synthetic domain shift where the
target labels are known.

  uv run python -m qdann.train --crop maize --epochs 1000

  uv run python -m qdann.train --crop maize --epochs 1000 --target

  uv run python -m qdann.train --crop maize --target --jdluc data/qdann_yields_maize.parquet

Without `--target` this runs with the adversarial branch off (lambda = 0) and measures one
thing: whether the 27 features predict a held-out county's yield better than that year's
state mean does. With it, the unlabelled pixels of `target_<crop>.parquet` are mixed into
every batch and both arms of the ablation are run and reported. `--jdluc` also writes the
QDANN arm's target pixels, averaged per held-out county-year, in the yield schema jdluc's
`trace.derive_jurisdictional_production_kg` reads.
"""

import argparse
import logging
import pathlib
from collections.abc import Iterable

import numpy
import pandas
import pyogrio.raw
import torch

from qdann import mpc, vae, weather
from qdann.losses import (
    QUANTILES,
    domain_loss,
    mse_loss,
    quantile_loss,
    update_quantile_weights,
)
from qdann.model import QDANN

logger = logging.getLogger(__name__)

HOLDOUT = 0.2
# ponytail: Iowa only, which is all the MPC read covers; a state table when it grows
STATE_NAMES = {"19": "Iowa"}
# jdluc's usda_nass_quickstats crop names
JDLUC_CROP_NAMES = {"maize": "CORN"}


def _show(weights: dict[float, float]) -> str:
    return ", ".join(f"w_{q}={w:.3f}" for q, w in weights.items())


def fit(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    adversarial: bool = False,
    quantile: bool = True,
    epochs: int = 400,
    weight_update_at: int | None = None,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> QDANN:
    """Train QDANN, or the same model with the adversarial branch switched off.

    Both arms see the same mixed source-and-target batches, so BatchNorm statistics are
    identical and the only difference between them is the domain loss and the reversed
    gradient it sends back through G_f. With no target domain there is nothing to mix in,
    and `target_x` is None.

    `sample_weights` carries Eq. 16's 1/L_i, one per source row, in `source_x`'s order.
    `quantile=False` swaps Eq. 7 for plain MSE: the Section 4.2 DNN baseline, and the arm of
    Fig. 16 that removes the quantile loss. Eqs. 8-11 rebalance the three quantiles against
    each other, so they are skipped along with it.
    """
    torch.manual_seed(seed)
    # Section 3.1 puts the Eq. 8-11 update partway through training, not at a fixed epoch
    weight_update_at = epochs // 2 if weight_update_at is None else weight_update_at
    model = QDANN(n_features=source_x.shape[1], lambda_=1.0 if adversarial else 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = dict.fromkeys(QUANTILES, 1.0)
    logger.info(f"quantile weights before epoch {weight_update_at:d}: {_show(weights)}")
    for epoch in range(epochs):
        if quantile and epoch == weight_update_at:
            model.eval()
            with torch.no_grad():
                weights = update_quantile_weights(y=source_y, yhat=model(source_x)[0])
            model.train()
            logger.info(f"quantile weights after Eqs. 8-11: {_show(weights)}")
        batches = torch.randperm(len(source_x)).split(batch_size)
        # BatchNorm1d raises on a batch of one, and a county holdout leaves an arbitrary
        # row count behind
        if len(batches[-1]) == 1:
            batches = batches[:-1]
        for batch in batches:
            x, n = source_x[batch], len(batch)
            if target_x is not None:
                # one mixed batch, so the discriminator cannot separate the domains on
                # per-domain normalization statistics (AMBIGUITIES.md #6)
                x = torch.cat([x, target_x[torch.randint(len(target_x), (n,))]])
            yhat, logit = model(x)
            batch_weights = None if sample_weights is None else sample_weights[batch]
            loss = (
                quantile_loss(
                    y=source_y[batch],
                    yhat=yhat[:n],
                    weights=weights,
                    sample_weights=batch_weights,
                )
                if quantile
                else mse_loss(
                    y=source_y[batch], yhat=yhat[:n], sample_weights=batch_weights
                )
            )
            if adversarial:
                loss = loss + domain_loss(
                    logit=logit, d=torch.cat([torch.ones(n), torch.zeros(n)])
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def r_squared(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """Eq. 17."""
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rmse(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float(((y - yhat) ** 2).mean().sqrt())


def nrmse(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """Eq. 17: RMSE normalized by the observed mean, so crops at different yield scales
    compare on one number. The paper doesn't fix a normalization convention; this is the
    one standard in the crop-yield literature."""
    return rmse(y=y, yhat=yhat) / float(y.mean())


def feature_columns(table: pandas.DataFrame) -> list[str]:
    """The 27 feature columns of a source table: everything but the keys and the label."""
    ret = [
        column
        for column in table.columns
        if column not in (*weather.KEY_COLUMNS, weather.LABEL_COLUMN)
    ]
    assert len(ret) == weather.N_FEATURES, f"{len(ret):d} feature columns"
    return ret


def scores(y: torch.Tensor, yhat: torch.Tensor) -> dict[str, float]:
    return {
        "r2": r_squared(y=y, yhat=yhat),
        "rmse": rmse(y=y, yhat=yhat),
        "nrmse": nrmse(y=y, yhat=yhat),
    }


def split(
    table: pandas.DataFrame, holdout: float = HOLDOUT, seed: int = 0
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Hold out whole counties: one county across adjacent years is nearly one sample."""
    counties = numpy.sort(table["fips"].unique())
    held = numpy.random.default_rng(seed).choice(
        counties, size=round(holdout * len(counties)), replace=False
    )
    mask = table["fips"].isin(held)
    return table[~mask], table[mask]


def standardize(
    train: pandas.DataFrame, *frames: pandas.DataFrame, columns: list[str]
) -> tuple[torch.Tensor, ...]:
    """Every frame on the training counties' mean and standard deviation.

    The target domain is scaled on the source's statistics rather than its own: pooled
    scaling would partly align the two domains before G_f ever saw them, and the
    adversarial branch would get credit for work the scaler had already done.
    AMBIGUITIES.md #4.

    `synth.make_domains` standardizes over the whole source set, which is right there --
    the shift has to survive into the model -- and leaks the holdout here.
    """
    mean, std = train[columns].mean(), train[columns].std()
    return tuple(
        torch.tensor(((frame[columns] - mean) / std).to_numpy(), dtype=torch.float32)
        for frame in (train, *frames)
    )


def domain_accuracy(
    model: QDANN, source_x: torch.Tensor, target_x: torch.Tensor, seed: int = 0
) -> float:
    """How often G_d names the right domain, over an equal number of rows from each.

    0.5 is the number the adversarial branch is trying to reach: G_f has made the two
    domains indistinguishable. 1.0 that will not come down is AMBIGUITIES.md #6's symptom
    and the suspect is BatchNorm, not the loss; 0.5 from the first epoch means the domains
    were never separable, which is a fault in the feature tables rather than in training.
    """
    n = min(len(source_x), len(target_x))
    rows = torch.randperm(len(target_x), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logit = model.eval()(torch.cat([source_x[:n], target_x[rows[:n]]]))[1]
    d = torch.cat([torch.ones(n), torch.zeros(n)])
    return float(((logit > 0).float() == d).float().mean())


def per_year_mean(train: pandas.DataFrame, test: pandas.DataFrame) -> torch.Tensor:
    """The baseline that matters: the training counties' mean yield in the test row's year.

    The weather features carry the year, so a model can score well on the pooled variance
    while knowing nothing county-to-county. Within one state, county yields in a given
    year are alike, so this is a strong baseline and the margin over it will look small.
    """
    means = train.groupby("year")[weather.LABEL_COLUMN].mean()
    predicted = test["year"].map(means).fillna(train[weather.LABEL_COLUMN].mean())
    return torch.tensor(predicted.to_numpy(), dtype=torch.float32)


def pixel_counties(
    predicted: numpy.ndarray, target: pandas.DataFrame, test: pandas.DataFrame
) -> pandas.DataFrame:
    """The target pixels' predictions (t/ha, in `target`'s row order) averaged per
    county-year, for the held-out county-years only, next to their NASS label: the model
    is trained on counties and applied to pixels, and this is where the two meet."""
    keys = list(weather.KEY_COLUMNS)
    return (
        target[keys]
        .assign(predicted=predicted)
        .groupby(keys, as_index=False)
        .agg(predicted=("predicted", "mean"), n_pixels=("predicted", "size"))
        .merge(test[[*keys, weather.LABEL_COLUMN]], on=keys)
    )


def jdluc_yields(
    counties: pandas.DataFrame, crop: str, county_names: dict[str, str]
) -> pandas.DataFrame:
    """`pixel_counties` in jdluc's `usda_nass_quickstats` schema, at a tier of its own."""
    return pandas.DataFrame(
        {
            "admin_level": "DISTRICT",
            "admin_id": "USA" + counties["fips"],
            "jurisdiction_name": [
                f"{STATE_NAMES[f[:2]]} | {county_names[f]}" for f in counties["fips"]
            ],
            "crop_name": JDLUC_CROP_NAMES[crop],
            "year": counties["year"],
            # both repos convert bushels at 56 lb, so t/ha to kg/ha is all that's left
            "yield_kg_per_ha": counties["predicted"] * 1000,
            "yield_tier": "QDANN",
        }
    ).set_index(["admin_level", "admin_id", "jurisdiction_name", "crop_name", "year"])


def evaluate(
    table: pandas.DataFrame,
    epochs: int,
    seed: int = 0,
    target: pandas.DataFrame | None = None,
    adversarial: bool = False,
    quantile: bool = True,
    vae_filter: bool = False,
    crop: str | None = None,
    counties: list[pandas.DataFrame] | None = None,
) -> dict[str, float]:
    """Fit one arm and score it on the held-out counties.

    Both arms are run with the same `target`: they then share their BatchNorm statistics
    and differ only in the domain loss and the reversed gradient, which is the difference
    the ablation is meant to measure. The target pixels come from every county, held-out
    ones included -- UDA is transductive and the paper maps the region it trains on. Only
    the *labels* are held out, and the target has none.

    `vae_filter` adds Section 3.2's step between the two: a VAE fitted on the target tensor
    drops the training counties it cannot reconstruct and weights the survivors by 1/L_i.
    It needs the target domain to train on and the crop to pick a threshold.

    `quantile=False` is Fig. 16's remaining arm; see `fit`.

    With a target, the pixels are also scored as county means (`pixel_counties`), which
    is appended to `counties` when one is passed.
    """
    features = feature_columns(table)
    train, test = split(table, seed=seed)
    logger.info(
        f"{len(train):d} rows / {train['fips'].nunique():d} counties train, "
        f"{len(test):d} rows / {test['fips'].nunique():d} counties held out"
    )
    frames = (train, test) if target is None else (train, test, target)
    train_x, test_x, *rest = standardize(*frames, columns=features)
    # the pinball gradient is bounded and Adam normalizes it, so the yield head's output
    # moves about one learning rate per step -- reaching an 11 t/ha intercept from zero
    # would take ~10k steps. The loss is scale-equivariant, so standardizing the label on
    # the training counties changes nothing but that walk. AMBIGUITIES.md #17
    mean, std = train[weather.LABEL_COLUMN].mean(), train[weather.LABEL_COLUMN].std()
    train_y, test_y = (
        torch.tensor(frame[weather.LABEL_COLUMN].to_numpy(), dtype=torch.float32)
        for frame in (train, test)
    )
    train_y_std = (train_y - mean) / std

    sample_weights = None
    if vae_filter:
        assert rest and crop is not None, (
            "the VAE filter needs a target domain and a crop"
        )
        errors = vae.reconstruction_error(
            vae.fit(rest[0], epochs=epochs, seed=seed), train_x
        )
        keep = errors <= vae.filter_threshold(errors, crop)
        logger.info(
            f"VAE filter keeps {int(keep.sum()):d} of {len(keep):d} source rows"
        )
        # `fit` shuffles the tensors it is handed, so filtering all three together here
        # keeps the weights aligned with the rows they belong to
        train_x, train_y_std = train_x[keep], train_y_std[keep]
        sample_weights = 1.0 / errors[keep]

    model = fit(
        source_x=train_x,
        source_y=train_y_std,
        target_x=rest[0] if rest else None,
        sample_weights=sample_weights,
        adversarial=adversarial,
        quantile=quantile,
        epochs=epochs,
        seed=seed,
    )
    with torch.no_grad():
        yhat = model(test_x)[0] * std + mean
    baseline = per_year_mean(train, test)
    ret = scores(y=test_y, yhat=yhat) | {
        f"per_year_mean_{name}": value
        for name, value in scores(y=test_y, yhat=baseline).items()
    }
    if rest:
        ret["domain_accuracy"] = domain_accuracy(
            model, source_x=train_x, target_x=rest[0], seed=seed
        )
        with torch.no_grad():
            predicted = (model(rest[0])[0] * std + mean).numpy()
        pixels = pixel_counties(predicted, target=target, test=test)
        ret["pixel_county_rmse"] = rmse(
            y=torch.tensor(pixels[weather.LABEL_COLUMN].to_numpy()),
            yhat=torch.tensor(pixels["predicted"].to_numpy()),
        )
        ret["pixel_county_n"] = float(len(pixels))
        if counties is not None:
            counties.append(pixels)
    return ret


def evaluate_seeds(
    table: pandas.DataFrame,
    epochs: int,
    seeds: Iterable[int] = range(5),
    **kwargs,
) -> dict[str, list[float]]:
    """`evaluate` once per seed, each metric gathered across the runs.

    The seed draws the held-out counties as well as the initialization, so the spread here
    is a spread over folds: a single `evaluate` says nearly as much about which counties it
    drew as about the model, which is why Fig. 16's deltas are read off means.
    """
    runs = [evaluate(table, epochs=epochs, seed=seed, **kwargs) for seed in seeds]
    return {name: [run[name] for run in runs] for name in runs[0]}


def report(name: str, scores: dict[str, float]) -> None:
    accuracy = scores.get("domain_accuracy")
    print(
        f"{name:>14s}  R2 {scores['r2']:6.3f}  RMSE {scores['rmse']:5.3f} t/ha"
        f"  NRMSE {scores['nrmse']:5.3f}"
        f"  margin {scores['r2'] - scores['per_year_mean_r2']:+6.3f}"
        + ("" if accuracy is None else f"  domain acc {accuracy:.3f}")
        + (
            f"  pixel RMSE {scores['pixel_county_rmse']:5.3f} t/ha"
            f" (n={scores['pixel_county_n']:.0f})"
            if "pixel_county_rmse" in scores
            else ""
        )
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop", choices=sorted(weather.CROP_TO_MONTHS), default="maize"
    )
    # Eq. 7 sums three pinball losses, whose gradient magnitude is constant, and Adam
    # normalizes it: the head moves about one learning rate per step regardless of how
    # wrong it is, so what training needs is steps, and a county table is small
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target",
        action="store_true",
        help="mix in the unlabelled pixels and run both arms of the ablation",
    )
    parser.add_argument(
        "--jdluc",
        type=pathlib.Path,
        help="write the QDANN arm's held-out county yields here, in jdluc's schema",
    )
    args = parser.parse_args()
    if args.jdluc and not args.target:
        parser.error("--jdluc needs --target")

    table = weather.read(weather.DATA_DIR / f"source_{args.crop}.parquet")
    target = (
        weather.read(weather.DATA_DIR / f"target_{args.crop}.parquet")
        if args.target
        else None
    )
    counties: list[pandas.DataFrame] = []
    print()
    for name, adversarial in (
        (("source only", False), ("qdann", True))
        if args.target
        else (("model", False),)
    ):
        scores = evaluate(
            table,
            epochs=args.epochs,
            seed=args.seed,
            target=target,
            adversarial=adversarial,
            counties=counties,
        )
        report(name, scores)
    print(
        f"{'per-year mean':>14s}  R2 {scores['per_year_mean_r2']:6.3f}  "
        f"RMSE {scores['per_year_mean_rmse']:5.3f} t/ha  "
        f"NRMSE {scores['per_year_mean_nrmse']:5.3f}"
    )
    if args.jdluc:
        # the last arm run is QDANN's
        fips, names = pyogrio.raw.read(
            mpc.TIGER_PATH, columns=["GEOID", "NAME"], read_geometry=False
        )[3]
        jdluc_yields(
            counties[-1],
            crop=args.crop,
            county_names=dict(zip(fips, names, strict=True)),
        ).to_parquet(args.jdluc)
        print(f"{len(counties[-1]):d} county yields -> {args.jdluc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
