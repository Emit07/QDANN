# AMBIGUITIES

Every point where this implementation had to decide something the paper does not settle, plus
three places where the paper's printed equations contradict its own text.

Each entry gives: what the paper says, what we did, and the expected direction of impact if the
choice is wrong. When the reproduction lands short of the published R^2, this file is the
diagnosis list.

Tags match the `src` field in `config.toml`.

---

## 1. Eq. 6 is published with its branches swapped

**Paper.** Eq. 6 (p. 6) assigns `q*|err|` when `y - yhat < 0`, i.e. when the model
*over*estimates. At `q = 0.1` that is the cheap branch.

**Contradicted by,** in the same paper: §3.1 ("when q is less than 0.5, the model will get a
higher penalty if it overestimates the yield"), p. 7 ("L_q=0.1 has a higher penalty on
overestimation"), and Fig. 6(a), which plots loss rising from ~1 to ~9 as the prediction goes
from 0 to 20 against `y = 10`. Standard Koenker-Bassett pinball loss agrees with the text, not
the equation.

**Choice.** Implement the branches reversed relative to the printed Eq. 6: overestimation costs
`(1-q)*|err|`, underestimation costs `q*|err|`.

**Impact if wrong.** Severe and silent. Eqs. 8-11 raise `w_0.1` when the model overestimates,
which only corrects bias if `L_q=0.1` punishes overestimation. Transcribing Eq. 6 literally
makes the bias correction push the wrong way while still training, still converging, and still
producing plausible maps. Guarded by `tests/test_model.py::test_pinball_asymmetry`.

`implementation_choice` — the highest-value entry in this file.

---

## 2. Optimizer, learning rate, batch size, weight initialization

**Paper.** Never stated. §3.1 gives depth, widths, dropout, `lambda`, `M`, and epoch count, and
stops there.

**Choice.** Adam, `lr = 1e-3`, `batch_size = 256`, PyTorch default initialization.

**Impact if wrong.** Moderate. 400 epochs on ~6000 county samples is a small problem; Adam at
1e-3 is unlikely to fail outright. Batch size interacts with the shared-BatchNorm choice (#6) —
a small batch makes domain statistics noisier. First knob to sweep if convergence looks wrong.

`reasonable_assumption`

---

## 3. Constant vs. ramped `lambda`

**Paper.** §3.1 sets `lambda = 1` flat, justified by the two loss magnitudes being similar.

**Contrast.** The original DANN (Ganin et al. 2016) ramps `lambda` from 0 to 1 over training,
specifically because a strong adversarial signal early on destabilizes the feature extractor
before it has learned anything task-relevant.

**Choice.** Constant 1, following the paper.

**Impact if wrong.** Moderate. If early training is unstable or the domain discriminator
collapses to chance immediately, a ramp is the first thing to try. Cheap to add later.

`implementation_choice`

---

## 4. Feature standardization scheme

**Paper.** Never mentions scaling. The 27 features are on wildly different scales — harmonic
coefficients are order 1, `srad` is order 100s, `ppt` order 10s.

**Options.** Fit the scaler on source only, on source+target pooled, or per-year.

**Choice.** Standardize on source statistics, apply to both domains.

**Impact if wrong.** Potentially large, and it interacts directly with the adversarial branch.
Pooled scaling partially aligns the domains before `G_f` ever sees them, which would make the
UDA ablation look weaker than it is. Source-only scaling is the honest choice: it keeps the
domain shift intact and lets `G_f` do the aligning. Per-year would leak year information.

`implementation_choice`

---

## 5. VAE latent dimension and layer widths

**Paper.** §3.2 and Eqs. 12-15 describe the VAE's loss and role completely, and its architecture
not at all. Fig. 7 is schematic.

**Choice.** `latent_dim = 8`, encoder hidden `[64, 32]`, decoder mirrored.

**Impact if wrong.** Small. The VAE is only a filter and a weight source, and the paper's own
ablation (Fig. 16) shows it is the weakest component — `DANN+VAE` *underperforms* plain `DANN`
for soybean and winter wheat. A latent that is too wide reconstructs everything and filters
nothing; too narrow and it filters indiscriminately. The threshold rule (mean error for maize
and soybean, 80th percentile for wheat) is stated, so the filter rate is bounded either way.

`reasonable_assumption`

---

## 6. Shared vs. per-domain BatchNorm

**Paper.** §3.1 says BatchNorm is used between hidden layers. It does not say how batches are
composed across the two domains.

**Choice.** Concatenate source and target into one batch per forward pass, so BatchNorm sees a
mixed population. The yield loss then masks to the source rows.

**Rationale.** Running separate passes gives each domain its own batch statistics, and the
discriminator can then separate domains on normalization artifacts rather than on features —
which makes the adversarial game trivially winnable and the alignment meaningless.

**Impact if wrong.** Moderate. Per-domain BN (as in AdaBN and several DANN variants) is a
defensible alternative and sometimes stronger. If the domain discriminator's accuracy sits at
100% and refuses to fall, suspect this.

`implementation_choice`

---

## 7. Pixel vs. field as the target sample unit

**Paper.** Target domain samples are Corteva AgriScience field-level yield-monitor records,
aggregated within field boundaries. Table 1: 495,627 maize field-years.

**Problem.** The Corteva data are listed in the paper's Data Availability as not available, and
there is no public equivalent of the field boundaries.

**Choice.** Sample CDL-masked 30 m pixels as the target unit. The paper itself does this for
winter wheat (§3.1: 2000 CDL "winter wheat" pixels added to the target domain per year), so it
is the paper's own fallback, not an invention.

**Impact if wrong.** Real but bounded, and it affects *evaluation* more than training. Pixels
are noisier than field means, so the target feature distribution is wider than the paper's. This
should if anything make the domain gap larger and the adversarial branch more necessary, not
less. It does mean subfield accuracy cannot be measured directly without Corteva access — see
the county-holdout validation plan.

`implementation_choice`

---

## 8. Landsat Collection 1 `pixel_qa` vs. Collection 2 `QA_PIXEL`

**Paper.** §2.2 describes cloud masking without naming the QA band or collection. The work
predates the Collection 1 retirement.

**Problem.** Collection 1 is decommissioned. Collection 2 L2 renamed the band `pixel_qa` ->
`QA_PIXEL` with different bit positions, and surface reflectance now needs the scale factor
`SR = DN * 0.0000275 - 0.2`, which Collection 1 did not.

**Choice.** Collection 2 L2 throughout, with the C2 scale factor and C2 QA bit positions.

**Impact if wrong.** Large and easy to miss. Skipping the scale factor leaves reflectance in raw
DN, which makes GCVI wrong by a constant-ish but not constant factor, and the harmonic
coefficients meaningless. Reading C1 bit positions against a C2 band silently masks the wrong
pixels. Verify GCVI lands in a plausible range (roughly 0-8 over growing crops) before trusting
any harmonic fit.

`implementation_choice` — logged now, bites in the feature pipeline.

---

## 9. What "one input layer" counts as

**Paper.** §3.1: "a total depth of ten. Its feature extractor G_f has one input layer, four
hidden layers, and one output layer. Both the yield predictor G_y and the domain discriminator
G_d have one input layer, two hidden layers, and one output layer."

**Problem.** "Input layer" could mean a `Linear` module or just the input vector.

**Choice.** A `Linear`. Then `G_f` = 1+4+1 = 6, `G_y` = 1+2+1 = 4, and the `G_f -> G_y` path is
10 — exactly the stated total depth. Reading it as the input vector gives 5+3 = 8, which
contradicts "a total depth of ten".

**Impact if wrong.** Small. Two extra layers on a 27-feature problem changes capacity
marginally.

`implementation_choice` — resolved by arithmetic, recorded because the phrasing is ambiguous.

---

## 10. Eq. 3's minus sign is not implemented literally

**Paper.** Eq. 3: `L = L_y - lambda*L_d`.

**Reading.** This describes `G_f`'s objective — minimize yield loss while *maximizing* domain
loss. `G_d` still minimizes `L_d` in the ordinary way. §3.1 says as much: "G_f and G_d are
trained adversarially", connected by a gradient reversal layer.

**Choice.** Compute `loss = L_y + L_d` and let the GRL carry the sign into `G_f` alone.

**Impact if wrong.** Severe. Backpropagating `L_y - L_d` directly makes `G_d` maximize its own
loss too, so the discriminator degrades instead of competing and the adversarial signal is
noise. This is the most common DANN implementation bug; guarded by
`tests/test_model.py::test_grl_sign`.

`implementation_choice`

---

## 11. Eq. 16 is printed as an unnormalized sum

**Paper.** Eq. 16: `L_y = sum_{i=1..Ns} (1/L_i) * [w_0.1*L_q=0.1 + w_0.5*L_q=0.5 + w_0.9*L_q=0.9]`.

**Problem.** Eqs. 4 and 5 both carry an explicit `1/N`; Eq. 16 does not. As printed, `L_y`
scales with the number of source samples in the batch *and* with the arbitrary magnitude of
`1/L_i` — reconstruction errors have no natural scale. That breaks the `lambda = 1` balance
against `L_d`, which Eq. 5 normalizes, and §3.1 justifies `lambda = 1` precisely by the two
losses having similar magnitudes.

**Choice.** Implement as a `1/L_i`-weighted mean: divide by `sum_i 1/L_i`.

**Impact if wrong.** Large but obvious. Left as a raw sum, the yield term dominates the domain
term by orders of magnitude and the adversarial branch stops mattering — the model quietly
degrades to source-only training. Only reachable once the VAE exists.

`implementation_choice`

---

## 12. Whether G_f's output is activated

**Paper.** §3.1: "The batch normalization layer and dropout layer (with a ratio of 0.5) are used
between hidden layers." G_f's 64->32 output layer is not a hidden layer, so nothing is said
about what follows it.

**Problem.** Taken literally, G_f's final Linear is followed by nothing, and each head begins
with a Linear — so `64->32` and `32->64` compose into a single linear map and the depth-ten
network has one fewer nonlinearity than its layer count suggests.

**Choice.** ReLU on the 32-d bottleneck; no BatchNorm and no dropout there, since §3.1 confines
those to hidden layers. DANN implementations conventionally expose a post-activation feature
representation to the discriminator.

**Impact if wrong.** Small. It costs one nonlinearity on a 27-feature problem. Worth knowing
about because it slightly changes what "domain-invariant features" means: the discriminator
sees a non-negative representation either way under this choice, and an unconstrained one
without it.

`implementation_choice`
