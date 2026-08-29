"""The component ablation of Fig. 16: what each of QDANN's three parts is worth.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 4.4 and Fig. 16. See
SPEC.md section 8.

  uv run python -c "from qdann import ablation, weather; \
d = weather.DATA_DIR; \
print(ablation.ablation(weather.read(d / 'source_maize.parquet'), \
weather.read(d / 'target_maize.parquet'), 'maize', epochs=1000))"

Each row removes exactly one component and leaves the other two on, so the numbers are
attributable: the quantile-loss row still runs the VAE filter and the adversarial branch.
"""

import statistics
from collections.abc import Iterable

import pandas

from qdann import train

COMPONENTS = ("adversarial", "vae_filter", "quantile")


def ablation(
    table: pandas.DataFrame,
    target: pandas.DataFrame,
    crop: str,
    epochs: int,
    seeds: Iterable[int] = range(5),
) -> pandas.DataFrame:
    """Mean held-out R2 for the full model and for each single-component-removed variant.

    `delta_r2` is the ablated mean minus the full model's, so a component that helps shows
    up as a negative number: removing it cost that much R2. Every row is averaged over the
    same seeds, which is the same set of county folds -- a single fold's noise is larger
    than the deltas Fig. 16 reports.
    """
    seeds = list(seeds)
    rows = []
    for removed in (None, *COMPONENTS):
        switches = dict.fromkeys(COMPONENTS, True) | (
            {} if removed is None else {removed: False}
        )
        runs = train.evaluate_seeds(
            table,
            epochs=epochs,
            seeds=seeds,
            target=target,
            crop=crop,
            **switches,
        )
        rows.append(
            {"removed": removed or "none", "r2_mean": statistics.mean(runs["r2"])}
        )
    ret = pandas.DataFrame(rows)
    ret["delta_r2"] = ret["r2_mean"] - ret["r2_mean"].iloc[0]
    return ret
