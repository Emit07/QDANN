# QDANN (reimplementation)

A clean-room reimplementation of the Quantile-loss Domain Adversarial Neural Network from:

> Ma, Y., Liang, S.-Z., Myers, D.B., Swatantran, A., Lobell, D.B. (2024).
> Subfield-level crop yield mapping without ground truth data: A scale transfer framework.
> *Remote Sensing of Environment* **315**, 114427. https://doi.org/10.1016/j.rse.2024.114427

**Not affiliated with or endorsed by the authors.** The published model code was never released;
this is written from the paper alone. The authors' yield maps are distributed separately under
CC-BY-NC-SA 4.0 and are **not** used as an input here, this implementation trains only on
public-domain data (USDA NASS Quick Stats, Landsat, gridMET, USDA CDL), so its outputs carry no
non-commercial restriction.

## Status

Model and pipeline implemented; the source domain is built and scored, the target domain is
not finished.

- **Source table**: `data/source_maize.parquet`, 99 Iowa counties, 2008-2018, 27 features.
- **County holdout**: mean R2 0.640 over five seeds, +0.172 over a per-year mean. Numbers and
  their caveats in [`notes/county_holdout_scores.md`](notes/county_holdout_scores.md).
- **Adversarial branch**: verified end to end on a synthetic domain shift with known target
  labels; on real pixels it awaits the target feature table (`data/gridmet_target_maize.parquet`
  is the weather leg only, the GCVI leg is an unrun Earth Engine job).
- **Not yet**: trained weights, subfield yield maps, crops other than maize.

## Layout

| Path | Contents |
|---|---|
| `SPEC.md` | Every equation and architectural claim in the paper, transcribed to pseudocode with section references |
| `AMBIGUITIES.md` | Details the paper leaves unspecified, what was assumed, and expected impact |
| `config.toml` | Hyperparameters, each tagged `paper_specified` / `reasonable_assumption` / `implementation_choice` |
| `notes/` | Measured results, with what each table does and does not support |
| `src/qdann/nass.py` | County-year yields from USDA NASS Quick Stats: the source labels |
| `src/qdann/gee.py` | County GCVI composites from Landsat Collection 2, via Earth Engine |
| `src/qdann/harmonics.py` | Harmonic regression over a GCVI series (Eq. 2) |
| `src/qdann/weather.py` | Monthly gridMET weather, and the joined source table |
| `src/qdann/target.py` | The unlabeled target domain: CDL-masked crop pixels |
| `src/qdann/model.py` | Feature extractor, yield predictor, domain discriminator, gradient reversal |
| `src/qdann/losses.py` | Quantile and domain losses, and the quantile reweighting (Eqs. 5-11, 16) |
| `src/qdann/vae.py` | The VAE data filter (Eqs. 12-16) |
| `src/qdann/train.py` | Training loop, county holdout, scoring, CLI |
| `src/qdann/baselines.py` | Ridge, random forest, and plain DNN comparisons (Section 4.2) |
| `src/qdann/ablation.py` | The component ablation of Fig. 16 |
| `src/qdann/synth.py` | Synthetic domain shift with known target labels, to check QDANN end to end |
| `tests/` | Shapes, gradient signs, loss asymmetry, holdout leakage, and pipeline wiring |

## Development

```bash
uv sync
uv run pytest -v
```

Torch is pinned to the CPU build (`torch==2.9.1+cpu`) via an explicit index — the model is ~50k
parameters and never needs a GPU.
