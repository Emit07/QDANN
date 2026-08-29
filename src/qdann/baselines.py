"""The comparison models of Section 4.2: ridge regression, random forest, and a plain DNN.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 4.2. See SPEC.md
section 7. All three read the same 27 features as QDANN and none of them sees the target
domain: what they measure is how much of QDANN's accuracy comes from the features alone.

  uv run python -m qdann.baselines --crop maize --epochs 1000

SCYM, the fourth comparison in the paper, is a published third-party product rather than a
model this repo trains -- AMBIGUITIES.md #22.
"""

import argparse
import logging

import numpy
import pandas
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from qdann import train, weather

logger = logging.getLogger(__name__)


def ridge(
    train_x: numpy.ndarray,
    train_y: numpy.ndarray,
    test_x: numpy.ndarray,
    alpha: float = 0.1,
) -> numpy.ndarray:
    return Ridge(alpha=alpha).fit(train_x, train_y).predict(test_x)


def random_forest(
    train_x: numpy.ndarray,
    train_y: numpy.ndarray,
    test_x: numpy.ndarray,
    seed: int = 0,
    n_estimators: int = 200,
) -> numpy.ndarray:
    return (
        RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        .fit(train_x, train_y)
        .predict(test_x)
    )


def evaluate(
    table: pandas.DataFrame, epochs: int, seed: int = 0
) -> dict[str, dict[str, float]]:
    """All three baselines on the fold `train.evaluate` uses, so the numbers compare.

    The split, the feature columns and the standardization are `train`'s own, not a second
    set: a baseline scored on a different fold would look like a fair comparison and not be
    one. Ridge and the forest fit the raw label -- AMBIGUITIES.md #17's standardization is
    about the pinball gradient's walk, and sklearn does not take that walk.
    """
    features = train.feature_columns(table)
    trained, held = train.split(table, seed=seed)
    train_x, test_x = train.standardize(trained, held, columns=features)
    train_y, test_y = (
        torch.tensor(frame[weather.LABEL_COLUMN].to_numpy(), dtype=torch.float32)
        for frame in (trained, held)
    )
    mean, std = float(train_y.mean()), float(train_y.std())

    model = train.fit(
        source_x=train_x,
        source_y=(train_y - mean) / std,
        quantile=False,
        epochs=epochs,
        seed=seed,
    )
    with torch.no_grad():
        dnn = model(test_x)[0] * std + mean
    predicted = {
        "ridge": ridge(train_x.numpy(), train_y.numpy(), test_x.numpy()),
        "random_forest": random_forest(
            train_x.numpy(), train_y.numpy(), test_x.numpy(), seed=seed
        ),
        "dnn": dnn.numpy(),
    }
    return {
        name: train.scores(y=test_y, yhat=torch.tensor(yhat, dtype=torch.float32))
        for name, yhat in predicted.items()
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop", choices=sorted(weather.CROP_TO_MONTHS), default="maize"
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = weather.read(weather.DATA_DIR / f"source_{args.crop}.parquet")
    print()
    for name, scores in evaluate(table, epochs=args.epochs, seed=args.seed).items():
        print(
            f"{name:>14s}  R2 {scores['r2']:6.3f}  RMSE {scores['rmse']:5.3f} t/ha"
            f"  NRMSE {scores['nrmse']:5.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
