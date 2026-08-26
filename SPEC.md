# SPEC — QDANN, transcribed from the paper

Source: Ma, Y., Liang, S.-Z., Myers, D.B., Swatantran, A., Lobell, D.B. (2024).
"Subfield-level crop yield mapping without ground truth data: A scale transfer framework."
*Remote Sensing of Environment* 315, 114427. DOI `10.1016/j.rse.2024.114427`.

The paper is not vendored here (Elsevier copyright). Every claim below cites its equation or
section number so it can be checked against the publication.

Two places where the code deliberately departs from the printed equations are marked
**DEPARTURE** and cross-referenced to `AMBIGUITIES.md`.

---

## 1. Features (27 per sample)

### 2.1 — study area and window

The paper states these in prose; they size every export and set what `t` is normalized over.

| | States | Years | Landsat window |
|---|---|---|---|
| Maize, soybean | IA, IL, IN, OH, MN, MO, SD, WI | 2008-2018 | Jan 1 - Dec 31 of the study year |
| Winter wheat | CO, KS, NE, ND, OK, TX, MI, MO, IN, IL, OH, SD, WI | 2008-2022 | Sep 1 (prior year) - Aug 31 |

Nebraska is absent from the maize/soybean list, which is what §2.1 says. The 2008 start is
set by CDL, which only reaches national coverage that year, so no earlier yield can be
paired with a crop mask.

For winter wheat the window straddles two calendar years and **the CDL year is the harvest
year**: CDL for year Y labels wheat planted in Y-1, so the Sep-Dec half of the window is
masked by the following year's CDL.

### Eq. 1 — vegetation index

```
GCVI = NIR / Green - 1
```

§2.2: chosen over NDVI/EVI for sensitivity to LAI in dense canopy without early saturation.

### Eq. 2 — harmonic regression over the GCVI time series

```
f(t) = c + sum_{k=1..n} [ a_k * cos(2*pi*w*k*t) + b_k * sin(2*pi*w*k*t) ]
```

- `t` = observation date, **normalized to [0, 1]** across the cropping year.
- `w = 1.5` (§2.2, following Deines et al. 2021).
- `n = 3`.
- Fitted per point-year or county-year sample.

Yields **7 coefficients** = `c` + `a_1..a_3` + `b_1..b_3`. These are the satellite predictors.

### Weather (§2.3) — 20 features

Five gridMET variables at 4 km, resampled to 30 m: `ppt`, `srad`, `tmmx`, `tmmn`, `vpd`. The
gridMET band for precipitation is `pr`; `ppt` is the PRISM spelling.

The 30 m resampling is what the **subfield** samples need -- each pixel takes the value of the
4 km cell containing it. The county leg does not do it: a county mean taken at gridMET's native
~4638 m (`projection().nominalScale()`) is the same quantity without the upsampling. The two
paragraphs are not in conflict; the resampling only ever mattered on the target side.

Monthly means over the growing season, which is crop-dependent:

| Crop | Months | Count |
|---|---|---|
| Maize, soybean | May–August | 5 x 4 = 20 |
| Winter wheat | March–June | 5 x 4 = 20 |

§2.3 is explicit that no feature selection is performed: "we did not try to optimize the choices
of weather variables... Careful feature selection would lower the model generalizability."

**Feature vector = 7 harmonic + 20 weather = 27.**

---

## 2. Domains

§3.1. Source `D_s = {(x_i, y_i)}_{i=1..Ns}` is county-level (USDA NASS). Target
`D_t = {(x_i)}_{i=1..Nt}` is subfield-level and **unlabeled**.

The paper's justification for domain adaptation as the right tool: `p(x,y) = p(x)p(y|x)`, and
since both domains come from the same region, `p_s(y|x) ~= p_t(y|x)` while
`p_s(x) != p_t(x)` due to spatial aggregation. So **the marginal shift is the whole problem**.
This is what `synth.py` reproduces synthetically: same `y|x`, different `p(x)`.

Domain label: `d = 1` if `x` is from the source domain, `0` otherwise (§3.1).

Sample counts (Table 1): county maize 6154 / soybean 6217 / winter wheat 3572.
Subfield (Corteva AgriScience, **not publicly available**) maize 495,627 / soybean 400,857 /
winter wheat 36,841 — used for *evaluation*, plus the winter-wheat VAE target set.

---

## 3. DANN objective

### Eq. 3

```
L = L_y - lambda * L_d
```

`lambda = 1` (§3.1: "the magnitudes of yield prediction loss and domain loss are similar.
Therefore, we set lambda to a value of 1").

**DEPARTURE (implementation, not a paper error).** The minus sign describes `G_f`'s objective:
minimize yield loss while *maximizing* domain loss. `G_d` still minimizes `L_d` normally. In
code this is `loss = L_y + L_d` with a gradient reversal layer between `G_f` and `G_d` carrying
the sign. Implementing the minus literally makes `G_d` maximize its own loss too, which trains
and converges to garbage. See `AMBIGUITIES.md` #10.

### Eq. 4 — yield loss (the MSE baseline QDANN replaces)

```
L_y = (1/Ns) * sum_{i=1..Ns} (y_i - yhat_i)^2
```

### Eq. 5 — domain loss

```
L_d = (1/(Ns+Nt)) * sum_{i=1..Ns+Nt} [ d_i*log(dhat_i) + (1-d_i)*log(1-dhat_i) ]
```

Binary cross-entropy over both domains. Note the paper prints it without the leading minus that
a loss requires; implemented as standard BCE. `G_d` emits a logit and this is
`BCEWithLogitsLoss` for numerical stability.

---

## 4. Quantile loss

### Eq. 6 — pinball loss (Koenker & Bassett 1978)

**As printed on p. 6:**

```
L_q(y, yhat) = { q       * |y - yhat|   if y - yhat < 0
               { (1 - q) * |y - yhat|   otherwise
```

**DEPARTURE — the branches are swapped in the publication. Implement them reversed:**

```
L_q(y, yhat) = { (1 - q) * |y - yhat|   if y - yhat < 0   (overestimation)
               { q       * |y - yhat|   otherwise         (underestimation)
```

`y - yhat < 0` means `yhat > y`, i.e. overestimation, which the printed Eq. 6 costs `q*|err|` —
at `q = 0.1` that is the *cheap* branch. Three independent statements in the same paper say the
opposite:

- §3.1: "when q is less than 0.5, the model will get a higher penalty if it overestimates the
  yield (Fig. 6(a))"
- p. 7: "L_q=0.1 has a higher penalty on overestimation while L_q=0.9 has a higher penalty on
  underestimation."
- Fig. 6(a), `q=0.1`, `y=10 t/ha`: loss ~1 at prediction 0, rising to ~9 at prediction 20.

Text, figure, and standard Koenker-Bassett all agree; Eq. 6 alone disagrees.

This matters twice over because **Eqs. 8-11 depend on it**: the weight update raises `w_0.1`
when the model overestimates, which is only corrective if `L_q=0.1` actually punishes
overestimation. Transcribing Eq. 6 literally makes the bias correction push the wrong way —
and it still trains, still converges, and still produces plausible maps. Do not "fix" the code
back to the printed form. See `AMBIGUITIES.md` #1; guarded by `tests/test_model.py::test_pinball_asymmetry`.

### Eq. 7 — three-quantile combination

```
L_y(y, yhat) = w_0.1 * L_q=0.1(y, yhat)
             + w_0.5 * L_q=0.5(y, yhat)
             + w_0.9 * L_q=0.9(y, yhat)
```

§3.1 motivation: "It is intractable to find an optimal q that works in all experiment years."

Printed per-sample; reduced over the batch by mean (Eq. 4 establishes the convention).

### Eqs. 8-9 — bias ratios, computed once on the source set

```
r_o = sum_{i=1..Ns} I(yhat_i - y_i) / Ns      # overestimation fraction
r_u = sum_{i=1..Ns} I(y_i - yhat_i) / Ns      # underestimation fraction
```

`I` is "an indicator function that returns 1 when the number inside is positive, and 0
otherwise" (§3.1, verbatim).

Because `I` is *strictly* positive, exact ties (`yhat == y`) count in neither, so
`r_o + r_u <= 1` rather than `== 1`. Assert with a tolerance, never equality.

### Eqs. 10-11 — weight update

```
w_0.1 = 2 * exp(r_o^2) / (exp(r_o^2) + exp(r_u^2))
w_0.9 = 2 * exp(r_u^2) / (exp(r_o^2) + exp(r_u^2))
w_0.5 = 1                                          # fixed; "equal penalization on over- and
                                                   # under-estimation"
```

By construction `w_0.1 + w_0.9 = 2` exactly.

Schedule (§3.1): all three weights initialize to 1. Train `M = 200` epochs. At epoch `M`, run
the source set forward, compute `r_o`/`r_u`, update the weights **once**, then continue to 400
epochs total.

The justification for using source bias as a proxy for target bias (§3.1): "if systematic biases
are observed in the county-level yield mapping, the same biases would likely happen at the
subfield level."

---

## 5. Architecture (§3.1)

Verbatim: "Our QDANN model has a total depth of ten. Its feature extractor G_f has one input
layer, four hidden layers, and one output layer. Both the yield predictor G_y and the domain
discriminator G_d have one input layer, two hidden layers, and one output layer. Each hidden
layer has 64 neurons. The output layer of G_f has a size of 32. The batch normalization layer
and dropout layer (with a ratio of 0.5) are used between hidden layers to account for
overfitting."

Counting `Linear` modules:

| Module | Linear layers | Shape |
|---|---|---|
| `G_f` | 6 | 27->64, 64->64, 64->64, 64->64, 64->64, 64->32 |
| `G_y` | 4 | 32->64, 64->64, 64->64, 64->1 |
| `G_d` | 4 | 32->64, 64->64, 64->64, 64->1 (logit) |

`G_f -> G_y` is 6 + 4 = **10**, matching the stated "total depth of ten". This is the only
reading of "input layer" that gives exactly ten; see `AMBIGUITIES.md` #9.

BatchNorm1d + ReLU + Dropout(0.5) between hidden layers; not after the final output layer of
either head. Whether `G_f`'s 32-wide output is itself activated is not stated; implemented with
a ReLU and no BatchNorm or dropout. See `AMBIGUITIES.md` #12.

### Hyperparameters (§3.1)

| Name | Value |
|---|---|
| `lambda` | 1 |
| epochs | 400 |
| `M` (weight update epoch) | 200 |
| dropout | 0.5 |
| hidden width | 64 |
| `G_f` output width | 32 |

Optimizer, learning rate, batch size, and weight initialization are **never stated**. See
`AMBIGUITIES.md` #2.

### Winter wheat target augmentation (§3.1)

Wheat has too few subfield samples in early years to represent the study area, so 2000 pixels
labeled "winter wheat" in CDL are sampled per mapping year and added to the target domain.
They carry no yield label and are dropped at evaluation.

---

## 6. VAE data filter (§3.2)

Purpose: county-year samples from very different regions/years cause **negative transfer**
(Wang et al. 2019). The VAE filters them out.

Trained on **target-domain** feature vectors `x in D_t`, so it learns the target feature
distribution.

### Eqs. 12-14

```
L_VAE   = L_recon + L_latent
L_recon = (1/nt) * sum_{i=1..nt} ||xhat_i - x_i||^2
L_latent = KL( N(mu, sigma^2) || N(0, 1) )
```

### Eq. 15 — per-sample reconstruction error

At inference each **source** vector `x_i in D_s` is passed through the trained VAE:

```
L_i = ||xhat_i - x_i||^2
```

### Filtering threshold

| Crop | Threshold |
|---|---|
| Maize, soybean | mean reconstruction error over all county-year samples |
| Winter wheat | 80th percentile of all reconstruction errors |

Source samples with `L_i > threshold` are dropped.

### Eq. 16 — final weighted yield loss

As printed:

```
L_y(y, yhat) = sum_{i=1..Ns} (1/L_i) * [ w_0.1*L_q=0.1(y_i,yhat_i)
                                       + w_0.5*L_q=0.5(y_i,yhat_i)
                                       + w_0.9*L_q=0.9(y_i,yhat_i) ]
```

Survivors are weighted by `1/L_i` — smaller reconstruction error means more relevant to the
target, so more weight.

**DEPARTURE.** Eq. 16 is printed as an unnormalized sum while Eqs. 4 and 5 both carry an
explicit `1/N`. As printed, `L_y` scales with batch size and with the arbitrary magnitude of
`1/L_i`, which destroys the `lambda = 1` balance against the normalized `L_d` of Eq. 5.
Implemented as a `1/L_i`-weighted mean (divide by `sum_i 1/L_i`). See `AMBIGUITIES.md` #11.

VAE latent dimension and layer widths are never stated. See `AMBIGUITIES.md` #5.

---

## 7. Evaluation (§3.3)

### Eq. 17

```
R^2 = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - ybar)^2
```

Plus RMSE and NRMSE.

Baselines: Ridge (alpha 0.1), Random Forest (200 trees), DNN (same depth and widths as QDANN),
and SCYM (Lobell et al. 2015; Deines et al. 2021 — maize and soybean only).

Every experiment repeated with **five random seeds** to account for initialization variance.

---

## 8. Ablation (Fig. 16) — delta R^2 when each component is removed

| Removed | Maize | Soybean | Winter wheat |
|---|---|---|---|
| Quantile loss | -0.09 | -0.03 | -0.09 |
| UDA (adversarial branch) | -0.04 | -0.03 | -0.05 |
| VAE filter | -0.05 | -0.02 | -0.03 |

The quantile loss is the largest single contributor. `DANN+VAE` *underperforms* plain `DANN`
for soybean and winter wheat, which is why the VAE is the component to drop if time is short.
