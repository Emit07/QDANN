# QDANN (reimplementation)

A clean-room reimplementation of the Quantile-loss Domain Adversarial Neural Network from:

> Ma, Y., Liang, S.-Z., Myers, D.B., Swatantran, A., Lobell, D.B. (2024).
> Subfield-level crop yield mapping without ground truth data: A scale transfer framework.
> *Remote Sensing of Environment* **315**, 114427. https://doi.org/10.1016/j.rse.2024.114427

**Not affiliated with or endorsed by the authors.** The published model code was never released;
this is written from the paper alone. The authors' yield maps are distributed separately under
CC-BY-NC-SA 4.0 and are **not** used as an input here — this implementation trains only on
public-domain data (USDA NASS Quick Stats, Landsat, gridMET, USDA CDL), so its outputs carry no
non-commercial restriction.

## Status

Model implementation. No feature pipeline, no trained weights, no yield maps yet.

## Layout

| Path | Contents |
|---|---|
| `SPEC.md` | Every equation and architectural claim in the paper, transcribed to pseudocode with section references |
| `AMBIGUITIES.md` | Details the paper leaves unspecified, what was assumed, and expected impact |
| `config.toml` | Hyperparameters, each tagged `paper_specified` / `reasonable_assumption` / `implementation_choice` |
| `src/qdann/` | Model, losses, synthetic-data check |
| `tests/` | Shape, gradient-sign, and loss-asymmetry checks |

## Development

```bash
uv sync
uv run pytest -v
```

Torch is pinned to the CPU build (`torch==2.9.1+cpu`) via an explicit index — the model is ~50k
parameters and never needs a GPU.
