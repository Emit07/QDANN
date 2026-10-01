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

**Paper.** §2.2 names both, one paragraph apart: "The Collection 2 data was used", and two
sentences earlier the band `pixel_qa`, which is the Collection 1 name. The paragraph
contradicts itself rather than being silent, which is the stronger evidence of the two.

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

---

## 13. Missing county yields are dropped, not backfilled from a coarser tier

**Paper.** §2.1 says the source labels are county-level NASS yields and says nothing about
counties NASS does not publish. Table 1 gives 6154 maize / 6217 soybean / 3572 winter wheat
county-year samples, which is well short of every county in the study area for every year.

**Problem.** NASS suppresses a county-year whose value would identify an operation (Title 7
U.S.C., CIPSEA), returning `(D)`, `(Z)`, `(S)` or `(NA)`. Something has to fill or drop those.

**Choice.** Drop them. A county-year keeps a label only where both its own AREA HARVESTED and
its own PRODUCTION survived suppression. No fallback to the agricultural district or the state,
and no averaging across years.

**Rationale.** A district- or state-tier fallback would inject exactly the spatial aggregation
the model is trained to undo: the label would carry a coarser support than the features, and the
scale transfer would be learning to reproduce an aggregation artifact. Multi-year averaging would
break the pairing with a specific year's harmonics and weather.

**Also recorded here.** Yield is reconstructed as PRODUCTION / AREA HARVESTED rather than read
from the YIELD statistic, because below the state level wheat is published only per class and the
ALL CLASSES yield does not exist. It covers more counties than YIELD does, and gives all three
crops one code path.

**Impact if wrong.** Moderate, and it biases the sample rather than corrupting it. Dropping
suppressed cells skews the source set toward counties with enough operations to escape
suppression — larger, more intensively cropped counties. Those are also the counties where the
target domain has the most subfield pixels, so the shift is partly self-correcting, but the model
sees fewer marginal-cropland county-years than the region actually contains.

`implementation_choice`

---

## 14. Mean reflectance first, GCVI second

**Paper.** §2.2 is explicit about the order: the CDL mask is applied, "the remaining data were
then aggregated to the county level by calculating the mean reflectance value for each band.
After that", GCVI is computed.

**Problem.** GCVI is a ratio, so `mean(NIR) / mean(green) - 1` is not `mean(NIR / green - 1)`.
The two differ by Jensen's inequality and the gap grows with the variance of green within the
county -- exactly what a county of mixed soil and canopy has. The paper's order is followed
here, but it forces a second decision: a ratio of means cannot be computed per pixel and then
averaged, so the harmonic fit has to happen client-side on an extracted series rather than
server-side per pixel.

**Choice.** Reduce green and NIR separately per (county, date), divide afterwards, fit Eq. 2 on
the resulting series. The target domain runs the identical code path over a single pixel.

**Impact if wrong.** Moderate on its own, severe if the two domains disagree. A source-target
difference in this order would manufacture a marginal shift of exactly the kind the adversarial
branch exists to remove, so the model would spend its capacity undoing a processing artefact.

`implementation_choice`

---

## 15. Which CDL codes count as the crop

**Paper.** §2.2 says only that CDL was used to mask "irrelevant land covers". It does not list
codes.

**Problem.** CDL carries single-crop codes (1 maize, 5 soybeans, 24 winter wheat) and separate
double-crop codes where a winter crop and a summer crop share the year: 26 winter wheat /
soybeans, 225 winter wheat / corn, 236 winter wheat / sorghum, 238 winter wheat / cotton. A
double-cropped field's NASS production is reported under both commodities, so excluding those
codes drops area that the county yield denominator still contains.

**Choice.** Single-crop codes only.

**Impact if wrong.** Negligible for Corn Belt maize and soybean, where double cropping is rare.
Material for winter wheat in Kansas and Oklahoma, where wheat-then-soybean and wheat-then-sorghum
are common: those counties get a crop mask smaller than their reported wheat area, so their
county-mean GCVI is drawn from a non-representative subset of their wheat. Revisit before
trusting the winter wheat leg; it does not affect the maize result this repo validates first.

`implementation_choice` -- crop-dependent, harmless today, revisit for wheat.

---

## 16. "Monthly mean" precipitation over a daily total

**Paper.** §2.3 lists five gridMET variables and takes the "monthly mean" of each over the
growing season.

**Problem.** Four of the five are daily state variables whose monthly mean is the obvious
quantity: `srad`, `tmmn`, `tmmx`, `vpd`. `pr` is not -- it is a daily precipitation *total* in
mm, so its monthly mean is a mean daily rainfall rate, while the agronomically meaningful
figure is the monthly accumulation. The paper writes `ppt`, which is the PRISM spelling, and
never says which of the two it took.

**Choice.** The mean, uniformly, as written. Each month is its own feature column and the day
count is fixed within a column, so the mean is the sum times a constant: monotone in rainfall,
and identical to the sum once the features are standardized.

**Impact if wrong.** None for maize and soybean, whose May-August months are 31/30/31/31 days.
February's length varies across the study years, so for a crop window that included it the two
would differ slightly between leap and common years -- winter wheat starts in March, so this
does not arise here either. Recorded so the mean is not "fixed" to a sum, which would rescale
one feature column against the config's standardization without changing anything it predicts.

`implementation_choice` -- equivalent under standardization; do not change it silently.

---

## 17. Whether the yield is standardized before the quantile loss

**Paper.** §3.1 standardizes the input features and says nothing about the label. Eq. 7 is
written directly on yield, and Fig. 6 plots predictions in kg/ha.

**Problem.** Eq. 7 sums three pinball losses, each of whose gradient with respect to the
prediction is a constant of magnitude q or 1-q; Adam then divides by the gradient's own
running magnitude. The size of a parameter step is therefore set by the learning rate and not
by how wrong the prediction is, so a head initialized near zero walks toward an 11 t/ha
intercept at roughly one learning rate per step -- about ten thousand steps before the fit
begins, on a county table that supplies four steps per epoch. A squared loss has no such
problem, which is why this does not show up in the equations as printed.

**Choice.** Standardize the label on the training counties, train on the standardized target,
and invert before scoring, so R2 and RMSE stay in t/ha. The three pinball losses are
positively homogeneous and Eqs. 8-11 depend only on the sign of the residual, so both the
loss and the weight update are unchanged by the rescaling.

**Impact if wrong.** None on the fitted function, which is why this is a scaling decision and
not a modelling one. Undoing it does not make the model wrong, it makes it slow, and slow here
looks exactly like a model that cannot beat the per-year mean -- a false negative on the one
gate this stage exists to run.

`implementation_choice` -- affects optimization only; keep the inverse transform with it.

---

## 18. The observation floor Eq. 2 is fitted above, and why the target's is higher

**Paper.** §2.2 fits the harmonic to "the growing-season time series" and reports roughly 20
clear Landsat observations per season. It sets no minimum, and §3.1 does not say whether the
subfield leg uses a different one from the county leg.

**Problem.** Eq. 2 has 2n+1 = 7 coefficients. At exactly seven distinct dates the design
matrix is square and the fit interpolates with zero residual, so `harmonics.fit`'s rank floor
of 7 admits series that carry no information beyond the dates themselves. The two domains sit
on opposite sides of this. A county mean survives a date if *any* pixel in the county is
clear, which is why the committed county table has a median of 52 observations per county-year
against §2.2's ~20; a single pixel needs itself clear, on a window that includes SLC-off
Landsat 7 throughout. The county table also measures what the noise costs. Its 20
worst-observed county-years (15-22 observations) against its best-observed 500:

| coefficient | SD at ~17 obs | SD at ~55 obs |
|---|---|---|
| `c`  | 0.367 | 0.297 |
| `a1` | 0.755 | 0.460 |
| `a3` | 0.298 | 0.234 |
| `b3` | 0.482 | 0.217 |

The third harmonic is 2.2x noisier at 17 observations, in counties that do not behave that way
in their well-observed years -- so it is fit noise, not signal. Left in, it inflates the target
feature variance, and the marginal shift `G_d` sees would then be sampling noise rather than
the aggregation gap the branch exists to remove. Phase 4 would look like it was working.

**Choice.** `target.MIN_OBSERVATIONS = 20`, against the source leg's default of 7, and
`--from-csv` prints the retained fraction at 7/12/15/20/25/30 so the number is chosen against
the measured pixel histogram rather than against this file. Pixel-years below the floor get NaN
coefficients from `harmonics.fit` and are dropped in `target.join`. If a defensible floor
retains too few, `--per-county` goes up and the export is re-run; the floor does not come down
to keep rows.

**Impact if wrong.** Too low is the dangerous direction and it is invisible: the features are
all present, all finite, and wrong in variance rather than in level. Too high costs rows and
shows up immediately as thin counties in the Phase 5 aggregation.

`implementation_choice` -- measured, not assumed; re-measure when the window or the sensor list
changes.

---

## 19. How a target pixel gets chosen: 50 per county, on CDL's grid, unmasked

**Paper.** §3.1 trains on 495,627 field-year samples from Corteva, and augments wheat with
"2000 pixels ... sampled per mapping year". It does not say how the pixels were drawn, nor how
the CDL mask and the Landsat grid were reconciled.

**Problem.** Three decisions, one subject.

*How many.* This is set by the validation, not by the model: Phase 5's only available gate
averages pixel predictions inside a county-year, so `per_county` is that mean's sample size.
Fifty gives ~54k pixel-years over Iowa 2008-2018 -- 50x the source set, ample for a branch that
subsamples per batch anyway -- for an export in the hundreds of megabytes. Two hundred would
halve the gate's standard error for four times the export.

*How drawn.* `sample(numPixels=k)` draws before masking, so a county that is 55% maize returns
~0.55k points and the count varies county to county, which silently reweights the county
aggregation toward high-maize counties. `stratifiedSample` with `classValues=[0, 1],
classPoints=[0, k]` guarantees k crop pixels. The seed carries the year because CDL rotates: a
2018 maize pixel is 2019 soybean, so a panel fixed across years would be wrong.

*On which grid.* CDL is 30 m Albers, Landsat C2 L2 is 30 m UTM, so the two grids are offset by
up to half a pixel. `stratifiedSample` takes no projection argument and returns CDL pixel
centres. `sampleRegions` does **not** then read the Landsat pixel containing each, as this
entry first said: the date mosaic's default projection is EPSG:4326 at 1°, so Earth Engine
resamples Landsat onto a 4326 grid at `scale=30` first. Measured at the 50 Story 2018 points,
its values equal the containing native pixel's on only 66% of point-dates, and on one
single-scene L8 date 30 of 50 matched the containing pixel and the other 20 a neighbour.
No Earth Engine target GCVI table was ever built, so nothing depends on that path;
`mpc.sample` reads the containing native pixel by definition, and that is the target pixel
from here on.

**Choice.** 50 per county per year, `stratifiedSample`, seeded by `seed + year`, and the target
imagery is deliberately **not** `updateMask(cropland)`ed the way `gee.build` masks the source.
The sampling already is the mask. Re-applying it would make Earth Engine resample CDL onto the
Landsat grid and drop every point whose Landsat centre falls in the neighbouring CDL cell -- a
non-random thinning at exactly the field edges, where the pixels that differ most from the
county mean live. The domains stay equivalent either way: only crop locations contribute to
both, the source masking then averaging, the target selecting then reading.

**Impact if wrong.** The count is a precision knob and shows up in the Phase 5 spread. The
draw and the mask are correctness: either bug biases *which* pixels the target domain contains,
and a biased target is a biased alignment, with nothing downstream to reveal it.

`implementation_choice` -- the sub-pixel CDL/Landsat offset is what it is at 30 m; recorded,
not engineered around. Which Landsat pixel is read is not left to a resampler: the one
containing the point.

---

## 20. The county mean is taken over whatever stayed clear that day

**Paper.** §2.2 reduces the crop-masked county to a mean per date. It does not state a minimum
clear fraction.

**Problem.** The source leg accepts a county-date if *any* pixel in it survives the cloud
screen. On a mostly-cloudy date the "county mean" is a mean over whatever sliver stayed clear,
which is not the county mean and is not the same quantity as on a clear date. This is where the
county table's median of 52 observations comes from, against §2.2's ~20. The target has no
equivalent: a clouded pixel is simply absent from its series. So the two domains differ in
their processing and not only in their support -- exactly the kind of difference
`notes/source_feature_table.md` warns manufactures a false marginal shift, and the adversarial
branch cannot tell the two apart.

**Choice.** Recorded and not fixed in this session. The fix is a minimum clear fraction on the
county reduction -- `ee.Reducer.mean().combine(ee.Reducer.count())`, drop dates whose count
falls below a fraction of the county's masked pixel total -- and it requires re-exporting the
entire source leg, so it folds into the export that adds the other seven states.

**Impact if wrong.** It is already wrong, by a bounded amount: it perturbs the source `c` and
low-order coefficients on cloudy dates. It is the prime suspect if `target.compare` shows the
harmonic means displaced between the domains, and it must be ruled in or out there before any
displacement is blamed on the aggregation, the mask, or the model.

`known_gap` -- fix with the multi-state re-export; check `compare` before blaming anything else.

---

## 21. The VAE is trained on its own, before QDANN

**Paper.** §3.2 says the VAE is trained on the target-domain feature vectors and that its
per-sample reconstruction error filters and weights the source set (Eqs. 15-16). It does not
say whether it is trained jointly with QDANN or beforehand.

**Problem.** Trained jointly, the filter and the weights move every epoch: rows leave and
re-enter the source set while the yield head is fitting them, and `1/L_i` is a moving target
that the head can chase. Eqs. 15-16 read as a fixed preprocessing step -- `L_i` appears as a
constant in the yield loss, not as a term with a gradient -- and the reconstruction loss shares
no parameters with G_f, so joint training would buy nothing but the coupling.

**Choice.** Train the VAE to convergence on the target tensor first, take `L_i` over the source
rows once, then train QDANN on the filtered, weighted source set. `L_i` never changes during
QDANN's training and no gradient flows from the yield loss into the VAE. It gets the same epoch
budget and seed as the run it feeds.

**Impact if wrong.** Small, and in the same direction as #5: this is the weakest component and
Fig. 16 shows it costing accuracy on two of the three crops. A joint schedule would mostly add
variance across seeds.

`reasonable_assumption` -- see #5 for the architecture and #11 for how `1/L_i` is normalized.

---

## 22. SCYM is not reproduced

**Paper.** Section 4.2 compares QDANN against ridge regression, a random forest, a plain DNN
and SCYM (Lobell et al. 2015; Deines et al. 2021), the last of these being a published
30 m yield product rather than a model the paper trains.

**Problem.** The first three are models: given the same 27 features and the same fold they can
be fitted here and scored on the same held-out counties. SCYM is a scale-and-crop-specific
regression calibrated against a crop model, distributed as a raster; reproducing its numbers
means downloading that raster and reprojecting it onto this repo's counties, not writing a
comparison model.

**Choice.** Implement ridge, the random forest and the DNN in `baselines.py`; skip SCYM. The
baseline table is therefore three rows where the paper's is four.

**Impact if wrong.** None on QDANN's own numbers. It removes one external reference point:
Section 4.2 uses SCYM to argue QDANN beats the published state of the art, and that claim is
not re-tested here.

`known_gap` -- a data-availability question, not a modelling decision.
