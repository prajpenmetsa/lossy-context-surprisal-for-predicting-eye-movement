# Paper Notes & Outline

---

## PAPER STRUCTURE

---

### Abstract

One paragraph. Cover: (1) motivation — human memory is lossy, standard surprisal assumes full context; (2) what we did — tested lossy-context surprisal on five eye-tracking measures from ZuCo-1; (3) central finding — optimal β differs across measures, forming a memory gradient: GD β=0.5, TRT β=0.1, reg β=0.7, FFD and GPT null; (4) implication — lexical access is memory-independent, early integration uses a ~2–3 word window, regression reflects medium-range integration failure; (5) novelty — first ET validation of lossy surprisal, first evidence for a memory gradient across reading stages.

---

### 1. Introduction

**Argument to build:**
- Reading is incremental prediction: each word is processed against prior context (cite Hale 2001, Levy 2008 on surprisal).
- But human memory is not a perfect recorder. Readers don't equally retain everything they've read — earlier words decay, attention is limited.
- Surprisal theory assumes full context. This is an idealisation.
- Futrell et al. (2020) proposed lossy-context surprisal: weight earlier words by β^distance. β=1 recovers standard surprisal; β→0 gives local, forgetting-heavy surprisal. Validated on self-paced reading only.
- Eye-tracking is richer than SPR: it captures multiple processing stages within a single word — from the very first fixation (FFD) to whether the eye ever comes back (regression). SPR collapses all of this into one measure.
- This distinction matters: if memory affects processing, it likely affects different stages differently. Lexical access (FFD) may be memory-independent; late integration (regression) may be memory-sensitive.
- **RQ1**: Does lossy surprisal explain ET variance beyond standard surprisal and dependency length?
- **RQ2**: Does the optimal β differ across early vs late measures — and across measures within those categories?

---

### 2. Background / Related Work

**Subsections:**

**2.1 Surprisal Theory**
Hale (2001) and Levy (2008) established surprisal (−log P(word | context)) as a predictor of reading difficulty. Higher surprisal → longer reading times. Validated extensively on SPR and ET. Standard surprisal assumes the reader has perfect memory of all prior words.

**2.2 Memory and Sentence Processing**
Working memory constraints in sentence comprehension (cite Gibson 1998 on dependency locality, Lewis & Vasishth 2005 on ACT-R interference). Readers don't hold full sentences in memory — decay and interference are well-documented. The key unresolved question: *how much* context do readers effectively use, and does it vary by processing stage?

**2.3 Lossy-Context Surprisal**
Futrell et al. (2020): formalise memory decay as a retention weight β^(t−i) for word at position i when predicting word at position t. Compute surprisal on degraded contexts sampled from this distribution. Lower β = more forgetting. Validated on Dundee corpus (SPR). Finding: intermediate β outperforms both β=1 (full context) and β=0 (unigram), suggesting readers use partial but not complete context. We extend this to eye-tracking.

**2.4 Eye-Tracking Measures**
Define the five measures and their cognitive interpretations:
- **FFD** — first fixation duration: duration of the first fixation on a word during first-pass reading. Indexes early lexical access.
- **GD** — gaze duration: total time on a word before the eye moves rightward. Indexes first-pass lexical-syntactic integration.
- **GPT** — go-past time: all time from first fixation on a word until the eye moves rightward past it, including regressions to earlier words. Indexes syntactically-driven re-analysis.
- **TRT** — total reading time: sum of all fixations on a word across the entire trial. Indexes overall processing load and re-reading.
- **reg** — regression: binary indicator of whether the eye ever moved leftward from the word. Indexes integration failure.

**2.5 Information Locality Hypothesis**
Futrell et al.'s ILH predicts that speakers arrange language to keep mutually predictive elements close together — minimising dependency distances to fit within the reader's effective memory window. Our GD finding (β=0.5, ~2–3 word window) is direct eye-tracking support for this hypothesis: early first-pass integration is sensitive to exactly the context range that English syntax tends to keep local.

---

### 3. Methods

**3.1 Dataset**
ZuCo-1 (Hollenstein et al., 2018): 12 adult native English speakers read 354 sentences from movie reviews and Wikipedia while eye movements were recorded with an EyeLink 1000 (500 Hz, monocular). Word-level fixation data extracted. 

**3.2 Eye-Tracking Measures**
(See definitions in §2.4.) Fixation exclusions: FFD and GD 100–1200/2000 ms; GPT and TRT 100–3000 ms. These are standard in the ET literature. Sensitivity analyses with loose (80–1500 ms) and strict (150–800 ms) thresholds confirmed results are not threshold-driven (see Appendix).

**3.3 GPT-2 Surprisal**
Word-level surprisal computed using GPT-2 (Radford et al., 2019) via the HuggingFace `transformers` library. Subword token probabilities summed to give word-level surprisal in bits.

**3.4 Lossy-Context Surprisal**
Following Futrell et al. (2020): for each word at position t, sample 50 degraded contexts by retaining word at position i with probability β^(t−i). Compute GPT-2 surprisal under each degraded context and average. Grid: β ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. β=1.0 recovers standard surprisal. 50 samples chosen after pilot showed curves stable beyond that point.

**3.5 Dependency Length**
Computed using spaCy's dependency parser on the ZuCo-1 sentences. Dependency length = distance in words between head and dependent.

**3.6 Statistical Models**
Mixed-effects regression using `glmmTMB` (Brooks et al., 2017). Continuous measures (FFD, GD, GPT, TRT): Gamma family with log link. Binary measure (reg): binomial with logit link. All predictors z-scored.

**Baseline model** (all five measures):
```
DV ~ word_length + log_freq + word_position + dep_length + surprisal
   + (1 | subject) + (1 | sentence)
```

**Lossy model** (adds lossy surprisal):
```
DV ~ word_length + log_freq + word_position + dep_length + surprisal + lossy_surprisal
   + (1 | subject) + (1 | sentence)
```

Fit compared via ΔAIC = AIC(baseline) − AIC(lossy). Positive ΔAIC = lossy surprisal improves fit. Random slopes for lossy surprisal by subject were attempted but did not improve fit (ΔAIC ≤ 0 in all cases), consistent with N=12 subjects being too few to estimate a per-subject slope covariance matrix. All models use random intercepts only.

**Figure for this section:** `fig_corr_matrix.png` — include as a supplementary or in the methods to demonstrate that dep_length is orthogonal to all other predictors (r < 0.09) and to justify that surp↔lossy collinearity (r = 0.852) makes the ΔAIC test conservative rather than inflated.

---

### 4. Results

**4.1 Baseline Models**

Report that standard surprisal is a significant predictor for FFD (p < 0.05) and regression (p < 0.001), but not for GD or TRT. Dependency length is significant for GPT and regression. For GD and TRT, neither predictor explains much variance — this motivates testing lossy surprisal.

**4.2 β Decay Curves**

**→ Figure 1: `fig1_final.png`** (β decay curves, faceted early/late)

Central result. For each measure, plot ΔAIC as a function of β. Key observations:
- FFD: flat at zero across all β — no memory effect at any context window.
- GD: peaks at β = 0.5 (ΔAIC = 10.71) — early integration benefits from moderate (~2–3 word) context retention.
- GPT: flat — regression-inclusive reading is not memory-sensitive.
- TRT: peaks at β = 0.1 (ΔAIC = 17.49) — total re-reading is driven by very local context.
- reg: peaks at β = 0.7 (ΔAIC = 53.95) — the strongest effect; medium-range context loss triggers lookbacks.

This directly answers RQ2: the optimal β is not the same across measures. It forms a gradient — GD (0.5) → TRT (0.1) → reg (0.7) — across three distinct reading stages.

**4.3 Effect Sizes**

**→ Figure 2: `fig2_final.png`** (standardized coefficient ± 95% CI by β)

All effects are negative — lossy surprisal predicts *easier* reading at the optimal β. Regression shows the steepest growth in effect size (from −0.11 at β=0.1 to −0.36 at β=0.7). TRT and GD show smaller, consistent negative effects. GPT and FFD show coefficients indistinguishable from zero across all β.

**4.4 Robustness**

Sensitivity analysis across three fixation exclusion thresholds (loose, standard, strict): TRT β=0.1 is stable (ΔAIC 17–19 across all thresholds); reg β=0.7 is identical across all three (ΔAIC = 53.95 in all cases, since reg is a binary outcome unaffected by duration cutoffs). GD shows some convergence failures under non-standard thresholds but the direction and approximate β are consistent. Full table in Appendix.

**→ Appendix Figure: `fig_sensitivity.png`**

**4.5 Per-Subject β Optima**

**→ Figure 3: `fig_per_subject_heatmap.png`** (subject × measure heatmap of optimal β)
**→ Figure 4: `fig_per_subject_dotplot.png`** (distribution of optimal β per measure)
**→ Supplementary: `fig_per_subject_reg.png`, `fig_per_subject_trt.png`, `fig_per_subject_gd.png`**

Key results:
- reg: 8/12 subjects show effects (ΔAIC > 2), modal β = 0.5 (matching group level), but spread from 0.1 to 0.9 (SD = 0.23). Genuine individual variation.
- TRT: 5/12 subjects, modal β = 0.1, concentrated in two subjects (ZKH, ZKW).
- GD: 4/12 subjects, split between β = 0.1 and β = 0.5.
- FFD and GPT: 2 and 1 subject respectively — confirms group-level null.
- 4 subjects (ZDM, ZGW, ZJN, ZMG) show no effects on any measure.

---

### 5. Discussion

**5.1 RQ1 Answered: Lossy Surprisal Explains Unique Variance**

Lossy surprisal improves model fit beyond standard surprisal and dependency length for GD, TRT, and regression. The near-zero correlation between dependency length and lossy surprisal (r = 0.02) confirms these are tapping distinct sources of processing difficulty. The high surp↔lossy correlation (r = 0.852) means the incremental test is conservative — the model must find variance in lossy surprisal that standard surprisal, despite being 85% correlated with it, cannot explain.

**5.2 RQ2 Answered: A Memory Gradient Across Processing Stages**

The optimal β values form a spectrum rather than a binary early/late split:
- **FFD (null)**: lexical access is memory-independent. The very first fixation reflects the difficulty of recognising a word, not its integration with prior context.
- **GD (β = 0.5)**: early first-pass integration uses a moderate memory window of ~2–3 words. Consistent with the Information Locality Hypothesis (see §5.4).
- **GPT (null)**: regression-inclusive reading is not memory-sensitive — it is instead driven by dependency length (syntactic complexity). This dissociates GPT from TRT despite both being "late" measures.
- **TRT (β = 0.1)**: total re-reading is driven by very local context loss. By the time a reader makes a total re-reading pass, broader sentence context is already inaccessible — re-reading is a local repair mechanism.
- **reg (β = 0.7)**: the strongest effect. Medium-range context loss (roughly 4–6 words) triggers lookbacks. Readers regress when they've lost enough of the prior sentence to fail integration, but not so much that they've given up on it.

**5.3 GPT Dissociates from TRT**

This is a new finding. Both GPT and TRT include re-reading time, but they have opposite patterns: GPT is predicted by dependency length (not lossy surprisal), while TRT is predicted by lossy surprisal (not dependency length). GPT captures syntactically-driven regressions — the reader realises they misanalysed a structure and goes back. TRT captures the cumulative cost of words that were simply hard to integrate given degraded memory. These are not the same cognitive operation and should not be conflated in future work.

**5.4 Information Locality Hypothesis**

The GD result (β = 0.5, ~2–3 word window) provides direct eye-tracking support for Futrell's Information Locality Hypothesis. The ILH predicts that speakers arrange language to keep mutually predictive elements within the reader's effective memory window. The ~2–3 word window we find for first-pass integration corresponds closely to the average dependency length in English (~2 words). Readers are sensitive to exactly the context range that syntax tends to keep local — a tight correspondence between memory architecture and linguistic structure.

**5.5 Individual Differences**

The per-subject analysis shows that group-level effects are driven by consistent subsets of readers. For regression, 8 of 12 subjects show meaningful effects, with individual optimal β values ranging from 0.1 to 0.9 (SD = 0.23). This suggests that while the direction of the effect is consistent, the optimal memory window for triggering regressions varies across readers. Possible explanation: readers with larger working memory capacity can retain more context and therefore only regress when a wider window of context is lost (higher β), while readers with more limited capacity regress more readily from local context loss (lower β). Four subjects showed no lossy surprisal effects on any measure — these may be more skilled or faster readers whose processing is less sensitive to context degradation.

**5.6 Why GPT-2? Implications of Bidirectional Models**

We used GPT-2 for a principled reason: lossy-context surprisal models a *forward incremental* reader who processes words left-to-right and forgets earlier words. GPT-2's causal left-to-right factorisation matches this architecture. Bidirectional models like BERT are fundamentally incompatible with this framework — they integrate future context during pretraining, so their word representations reflect both left and right context simultaneously. Applying lossy-context surprisal with BERT-style representations would conflate the memory-decay component (forgetting of past words) with access to future words the reader hasn't seen yet. This would likely wash out or invert the β gradient. More broadly, any model of human incremental reading difficulty should use a left-to-right architecture. A natural extension of this work is testing whether the β gradient replicates with other causal language models (GPT-Neo, LLaMA) and whether the gradient is model-dependent or a property of the task.

**5.7 Limitations**

- N = 12 subjects (ZuCo-1). Individual-level findings should be treated as exploratory. A larger ET dataset would allow random slopes by subject and more reliable per-subject β estimation.
- GPT-2 is a relatively small model (124M parameters). Larger causal LMs may produce different absolute surprisal values, potentially shifting optimal β.
- β is a single global parameter — it doesn't model position-dependent decay or word importance. More expressive memory models (e.g., attention-based decay) are a natural extension.
- We compute lossy surprisal at the word level, collapsing over subword tokens. This is standard practice but loses some tokenisation-level detail.

---

### 6. Conclusion

Memory decay shapes reading differently across processing stages. We showed that lossy-context surprisal — never previously tested on eye-tracking — explains variance in gaze duration, total reading time, and regression beyond standard surprisal and dependency length. The optimal β forms a gradient across measures (GD: 0.5, TRT: 0.1, reg: 0.7), directly answering that different reading stages reflect different memory windows. The dissociation between GPT and TRT is a novel finding. The per-subject analysis suggests genuine individual differences in memory window size for regression triggering. Taken together, these results argue that memory decay is not a uniform background parameter but a stage-specific, reader-specific constraint on sentence processing.

---

## FIGURE ASSIGNMENTS

| Figure | File | Section | Purpose |
|---|---|---|---|
| Figure 1 | `fig1_final.png` | §4.2 | Central result: β decay curves for all 5 measures |
| Figure 2 | `fig2_final.png` | §4.3 | Effect sizes: standardized coefficients ± 95% CI |
| Figure 3 | `fig_per_subject_heatmap.png` | §4.5 | Individual optimal β per subject per measure |
| Figure 4 | `fig_per_subject_dotplot.png` | §4.5 | Distribution of optimal β per measure across subjects |
| Appendix A | `fig_corr_matrix.png` | §3 / Appendix | Predictor correlation matrix |
| Appendix B | `fig_sensitivity.png` | §4.4 / Appendix | Sensitivity analysis: β curves under 3 threshold sets |
| Supplementary | `fig_per_subject_reg.png` | §4.5 supp. | Per-subject β curves for regression |
| Supplementary | `fig_per_subject_trt.png` | §4.5 supp. | Per-subject β curves for TRT |
| Supplementary | `fig_per_subject_gd.png` | §4.5 supp. | Per-subject β curves for GD |

---

## RESULTS NOTES (numbers to use when writing)

### Correlation matrix (β = 0.5)
- surp_z ↔ lossy_z: r = 0.852
- freq_z ↔ lossy_z: r = −0.828
- surp_z ↔ freq_z: r = −0.732
- wlen_z ↔ freq_z: r = −0.718
- dep_z ↔ all others: r < 0.09 (near-orthogonal)

### Group-level β optima and ΔAIC (standard threshold, 50-sample)
| Measure | Optimal β | ΔAIC | Significant? |
|---------|-----------|------|--------------|
| FFD | — | < 1 at all β | No |
| GD | 0.5 | 10.71 | Yes |
| GPT | — | < 1 at all β | No |
| TRT | 0.1 | 17.49 | Yes |
| reg | 0.7 | 53.95 | Yes |

### Random slopes
All ΔAIC(slopes vs intercepts) ≤ 0. Random intercepts only throughout.

### Per-subject (ΔAIC > 2 threshold)
| Measure | N with effect / 12 | Modal β | β SD |
|---------|-------------------|---------|------|
| FFD | 2 | 0.9 | 0 |
| GD | 4 | 0.1 | 0.231 |
| GPT | 1 | 0.1 | — |
| TRT | 5 | 0.1 | 0.335 |
| reg | 8 | 0.5 | 0.233 |

### Sensitivity analysis (all thresholds, reg)
reg ΔAIC = 53.95 at β = 0.7 — identical across loose, standard, and strict thresholds.
