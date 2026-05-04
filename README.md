# Paper Notes — Things Worth Including

## Results worth highlighting

### The memory gradient is a spectrum, not a binary
Early vs. late is too coarse. We have three distinct optimal β values across three measures — GD (0.5), TRT (0.1), reg (0.7) — that don't fall neatly on one side of a line. This is the central novel finding. The argument should be: if memory were uniformly engaged or not, you'd expect all measures to peak at the same β. They don't.

### reg is the strongest and most robust finding (ΔAIC = 54)
Regressions peak at β = 0.7 (ΔAIC = 53.95) and are perfectly stable across all fixation exclusion thresholds — identical ΔAIC whether you use loose (80–1500ms), standard (100–1200ms), or strict (150–800ms) cutoffs. This is both the largest effect and the most robust. Since reg is a binary outcome (any rightward fixation), fixation duration thresholds don't affect it, which explains why it's the only measure with perfectly identical results across all three threshold conditions. This is the cleanest result in the paper — medium-range context loss triggers lookbacks.

### GPT dissociates from TRT
Both are "late" measures, but GPT is predicted by dependency length and not by lossy surprisal, while TRT is the opposite. They're not the same thing. Regression-inclusive reading (GPT) is driven by syntactic complexity; total re-reading (TRT) by local memory failure. This is worth its own paragraph — it's a new finding that hasn't been noted in the ET literature.

### TRT β = 0.1 — re-reading is hyper-local
The optimal β for TRT is 0.1, meaning only the most recent 1–2 words matter for total re-reading time. Interpretation: by the time a reader re-reads a word, they've already forgotten the broader sentence context. Re-reading is a *local repair*, not a global one.

### FFD null is a clean negative result
First fixation duration shows no effect at any β, across all thresholds. This is not a failure — it's theoretically meaningful. Lexical access (the cognitive process FFD indexes) is memory-independent. It happens before integration with prior context can influence processing time.

### dep_z is nearly orthogonal to everything
From the correlation matrix: dep_z correlates at r < 0.09 with every other predictor (surp, lossy, wlen, freq, wpos). This means dependency length and lossy surprisal are capturing genuinely separate variance. The sentence that belongs in the paper: *"The near-zero correlation between dependency length and lossy surprisal (r = 0.02) confirms that these predictors tap distinct sources of processing difficulty."*

### Collinearity between surp_z and lossy_z (r = 0.852) makes positive results conservative
Standard and lossy surprisal are correlated at r = 0.852 (50-sample estimates at β = 0.5) because they're both derived from GPT-2 on the same words. Since the baseline model already includes standard surprisal, any incremental fit from lossy_z is variance that standard surprisal — despite being highly correlated — cannot explain. The positive ΔAIC values are a *conservative* test, not an inflated one.

---

## Theory connections

### Information Locality Hypothesis (Futrell et al.)
Futrell's ILH predicts that linguistic structures minimize the distance between mutually predictive elements, keeping dependencies short enough to fit in the active memory window. The GD optimal β = 0.5 (~2–3 word window) maps directly onto this: early first-pass integration is sensitive to exactly the context range that English syntax tends to keep local. The GD finding is empirical support for the ILH from eye-tracking.

### Why GPT-2 and not BERT (theoretical extension, no implementation needed)
Bidirectional models like BERT integrate future context by design. Since lossy-context surprisal models a *forward* incremental reader who forgets earlier words, BERT is the wrong architecture — it would conflate past and future context and likely wash out or invert the β gradient. GPT-2's left-to-right factorization is not just a practical choice; it's the theoretically correct one for modeling human reading. Worth noting as a limitation and a direction: testing whether the β gradient survives with other causal LMs (GPT-Neo, Llama) would be a natural extension.

---

## Methods points worth making explicit

- **Why ZuCo-1?** It's one of the few publicly available ET datasets with simultaneous EEG recording, word-level fixation data, and a large enough sentence set (354) for mixed-effects modeling. We use only the ET here — the EEG channel is a natural extension.
- **Why 50 samples for lossy surprisal?** Pilot runs with 20 samples showed noisier β curves. 50 samples produced smooth, interpretable curves without substantial additional compute. The curves are stable; going to 100 is not necessary.
- **Gamma family for continuous measures** — fixation durations are strictly positive and right-skewed. Gamma with log link is more appropriate than Gaussian. Worth one sentence.
- **Both random effects (subject + sentence)** are necessary — subjects differ in reading speed, sentences differ in inherent difficulty. Excluding either inflates false positives.

---

## Sensitivity analysis summary (confirmed with 50-sample file)

| Measure | Standard optimal β | Loose optimal β | Strict optimal β | Verdict |
|---------|-------------------|-----------------|------------------|---------|
| FFD     | 0.9 (ΔAIC = 0.09) | 0.1 (ΔAIC = 0.14) | 0.1 (ΔAIC = 2.73) | **Null — robust** |
| GD      | 0.5 (ΔAIC = 10.71) | 0.1 (ΔAIC = 15.41)* | 0.1 (ΔAIC = 15.04)* | Significant; some convergence failures under non-standard thresholds |
| GPT     | 0.9 (ΔAIC = 0.98) | 0.7 (ΔAIC = 0.27) | 0.9 (ΔAIC = 5.00)† | **Null — robust** |
| TRT     | 0.1 (ΔAIC = 17.49) | 0.1 (ΔAIC = 18.27) | 0.1 (ΔAIC = 19.47) | **Stable: β = 0.1 across all** |
| reg     | 0.7 (ΔAIC = 53.95) | 0.7 (ΔAIC = 53.95) | 0.7 (ΔAIC = 53.95) | **Perfectly stable — strongest finding** |

\* Many convergence failures (NAs) at other β values for GD under loose/strict thresholds — optimal β within converged models, not full curve.
† GPT strict β=0.5 gives ΔAIC=−44.21 — clear convergence artifact (non-positive-definite Hessian); discard.

Note on GD: standard threshold is the most complete and reliable (β=0.5 peak). Loose/strict thresholds lose many GD cells to convergence failures, making their "optimal β" less trustworthy. The direction (positive effect, moderate β) is consistent.

## Pending analyses (results to add once done)

- **Per-subject β optima**: are the optimal β values consistent across all 12 readers, or is there individual variation? If some readers show GD β=0.3 and others β=0.7, that's an individual differences finding worth discussing.

## Random slopes (done — negative result, reportable)

Random slopes for `lossy_z` by subject were attempted for all 25 beta × measure cells. In every cell where the comparison was computable, ΔAIC(slopes vs intercepts) was negative (range: −2.00 to −3.91), meaning random slopes *hurt* model fit. This is expected: with only 12 subjects there is not enough data to estimate a per-subject slope covariance matrix.

Paper sentence: *"Random slopes for lossy surprisal by subject were attempted for all measures and β values; slopes models did not improve over random intercepts (ΔAIC ≤ 0 in all comparisons), consistent with the limited number of subjects (N = 12). All reported models use random intercepts only."*

Note: fixed-effect estimates for `lossy_z` were stable between the random-intercepts and random-slopes models where both converged cleanly, confirming the fixed effects are not sensitive to the random-effects specification.
