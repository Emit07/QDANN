"""Train the yield head on the county-level source table and score it against baselines.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.1. The training loop
itself is shared with `synth.py`, which exercises it on a synthetic domain shift where the
target labels are known.

  uv run python -m qdann.train --crop maize --epochs 1000

  uv run python -m qdann.train --crop maize --epochs 1000 --target

Without `--target` this runs with the adversarial branch off (lambda = 0) and measures one
thing: whether the 27 features predict a held-out county's yield better than that year's
state mean does. With it, the unlabelled pixels of `target_<crop>.parquet` are mixed into
every batch and both arms of the ablation are run and reported.
"""

import argparse
import logging

import numpy
import pandas
import torch

from qdann import weather
from qdann.losses import QUANTILES, domain_loss, quantile_loss, update_quantile_weights
from qdann.model import QDANN

logger = logging.getLogger(__name__)

HOLDOUT = 0.2


def _show(weights: dict[float, float]) -> str:
    return ", ".join(f"w_{q}={w:.3f}" for q, w in weights.items())


def fit(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor | None = None,
    adversarial: bool = False,
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
    """
    torch.manual_seed(seed)
    # Section 3.1 puts the Eq. 8-11 update partway through training, not at a fixed epoch
    weight_update_at = epochs // 2 if weight_update_at is None else weight_update_at
    model = QDANN(n_features=source_x.shape[1], lambda_=1.0 if adversarial else 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = dict.fromkeys(QUANTILES, 1.0)
    logger.info(f"quantile weights before epoch {weight_update_at:d}: {_show(weights)}")
    for epoch in range(epochs):
        if epoch == weight_update_at:
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
            loss = quantile_loss(y=source_y[batch], yhat=yhat[:n], weights=weights)
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


def evaluate(
    table: pandas.DataFrame,
    epochs: int,
    seed: int = 0,
    target: pandas.DataFrame | None = None,
    adversarial: bool = False,
) -> dict[str, float]:
    """Fit one arm and score it on the held-out counties.

    Both arms are run with the same `target`: they then share their BatchNorm statistics
    and differ only in the domain loss and the reversed gradient, which is the difference
    the ablation is meant to measure. The target pixels come from every county, held-out
    ones included -- UDA is transductive and the paper maps the region it trains on. Only
    the *labels* are held out, and the target has none.
    """
    features = [
        column
        for column in table.columns
        if column not in (*weather.KEY_COLUMNS, weather.LABEL_COLUMN)
    ]
    assert len(features) == weather.N_FEATURES, f"{len(features):d} feature columns"

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

    model = fit(
        source_x=train_x,
        source_y=(train_y - mean) / std,
        target_x=rest[0] if rest else None,
        adversarial=adversarial,
        epochs=epochs,
        seed=seed,
    )
    with torch.no_grad():
        yhat = model(test_x)[0] * std + mean
    baseline = per_year_mean(train, test)
    scores = {
        "r2": r_squared(y=test_y, yhat=yhat),
        "rmse": rmse(y=test_y, yhat=yhat),
        "per_year_mean_r2": r_squared(y=test_y, yhat=baseline),
        "per_year_mean_rmse": rmse(y=test_y, yhat=baseline),
    }
    if rest:
        scores["domain_accuracy"] = domain_accuracy(
            model, source_x=train_x, target_x=rest[0], seed=seed
        )
    return scores


def report(name: str, scores: dict[str, float]) -> None:
    accuracy = scores.get("domain_accuracy")
    print(
        f"{name:>14s}  R2 {scores['r2']:6.3f}  RMSE {scores['rmse']:5.3f} t/ha"
        f"  margin {scores['r2'] - scores['per_year_mean_r2']:+6.3f}"
        + ("" if accuracy is None else f"  domain acc {accuracy:.3f}")
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
    args = parser.parse_args()

    table = weather.read(weather.DATA_DIR / f"source_{args.crop}.parquet")
    target = (
        weather.read(weather.DATA_DIR / f"target_{args.crop}.parquet")
        if args.target
        else None
    )
    print()
    for name, adversarial in (
        (("source only", False), ("qdann", True)) if args.target else (("model", False),)
    ):
        scores = evaluate(
            table,
            epochs=args.epochs,
            seed=args.seed,
            target=target,
            adversarial=adversarial,
        )
        report(name, scores)
    print(
        f"{'per-year mean':>14s}  R2 {scores['per_year_mean_r2']:6.3f}  "
        f"RMSE {scores['per_year_mean_rmse']:5.3f} t/ha"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
