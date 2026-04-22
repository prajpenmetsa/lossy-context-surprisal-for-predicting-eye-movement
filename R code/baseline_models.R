# =============================================================================
# baseline_models.R
# Fit baseline glmmTMB models for ZuCo-1 SR eye-tracking measures.
# Run this NOW (before lossy_surprisal.csv is ready).
# =============================================================================
# Install once:
#   install.packages(c("glmmTMB", "MuMIn", "tidyverse", "lme4"))

library(glmmTMB)
library(MuMIn)
library(tidyverse)

# ── 1. Load & prepare data ────────────────────────────────────────────────────

df_raw <- read_csv("analysis_data.csv", show_col_types = FALSE)

cat("Raw rows:", nrow(df_raw), "\n")

df <- df_raw %>%
  # Keep only rows with valid surprisal and covariates
  filter(!is.na(surprisal), !is.na(word_length), !is.na(log_freq)) %>%
  # Exclude sentence-initial words (surprisal = NaN already filtered,
  # but word_position == 0 has no left context — keep them for TRT/reg
  # but they need careful handling; simpler to exclude for consistency)
  filter(word_position > 0) %>%
  # Rename for convenience
  rename(
    sent_id  = global_sent_idx,
    subj     = subject,
    wlen     = word_length,
    wpos     = word_position,
    freq     = log_freq,
    surp     = surprisal
  ) %>%
  # Scale all predictors (mean=0, sd=1) — critical for model convergence
  mutate(
    surp_z  = scale(surp)[,1],
    wlen_z  = scale(wlen)[,1],
    freq_z  = scale(freq)[,1],
    wpos_z  = scale(wpos)[,1]
  ) %>%
  # Convert subject and sent_id to factors for random effects
  mutate(
    subj    = factor(subj),
    sent_id = factor(sent_id)
  )

cat("Analysis rows (valid surprisal, word_pos > 0):", nrow(df), "\n")
cat("Subjects:", nlevels(df$subj), "\n")
cat("Sentences:", nlevels(df$sent_id), "\n\n")

# ── 2. Subsets for fixated words only (FFD, GD, TRT require fixation) ─────────

df_fix <- df %>% filter(!is.na(FFD), FFD > 0)
cat("Fixated word tokens (for FFD/GD/TRT):", nrow(df_fix), "\n\n")

# ── 3. Fit baseline models ────────────────────────────────────────────────────
# Family: Gamma(link="log") for durations (positive, right-skewed)
# Random effects: random intercept per subject + per sentence item
# We use simpler random effects to ensure convergence:
#   (1 | subj) + (1 | sent_id)
# Add (0 + surp_z | subj) for random slope if convergence allows.

fit_rt_model <- function(outcome, data, label) {
  cat("─── Fitting", label, "───\n")
  formula_baseline <- as.formula(
    paste(outcome, "~ wlen_z + freq_z + wpos_z + surp_z +
          (1 | subj) + (1 | sent_id)")
  )
  m <- tryCatch(
    glmmTMB(formula_baseline, data = data, family = Gamma(link = "log"),
            control = glmmTMBControl(optimizer = optim,
                                     optArgs = list(method = "BFGS"))),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(m)) {
    cat("  AIC:", AIC(m), "\n")
    cat("  Marginal R²:", round(r.squaredGLMM(m)[1], 4), "\n")
    print(summary(m)$coefficients$cond)
  }
  cat("\n")
  return(m)
}

# FFD — First Fixation Duration (early measure)
m_FFD_base <- fit_rt_model("FFD", df_fix, "FFD (baseline)")

# GD — Gaze Duration (early measure)
m_GD_base  <- fit_rt_model("GD",  df_fix, "GD (baseline)")

# TRT — Total Reading Time (late measure)
m_TRT_base <- fit_rt_model("TRT", df_fix, "TRT (baseline)")

# Regression — binary late measure (logistic)
cat("─── Fitting Regression (baseline) ───\n")
m_reg_base <- tryCatch(
  glmmTMB(reg ~ wlen_z + freq_z + wpos_z + surp_z +
            (1 | subj) + (1 | sent_id),
          data = df, family = binomial(link = "logit"),
          control = glmmTMBControl(optimizer = optim,
                                   optArgs = list(method = "BFGS"))),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)
if (!is.null(m_reg_base)) {
  cat("  AIC:", AIC(m_reg_base), "\n")
  print(summary(m_reg_base)$coefficients$cond)
}

# ── 4. Save models ────────────────────────────────────────────────────────────
saveRDS(list(FFD = m_FFD_base, GD = m_GD_base,
             TRT = m_TRT_base, reg = m_reg_base),
        "baseline_models.rds")
saveRDS(df,     "analysis_df.rds")
saveRDS(df_fix, "analysis_df_fix.rds")

cat("\n✓ Baseline models saved to baseline_models.rds\n")
cat("  Run lossy_models.R once lossy_surprisal.csv is ready.\n")

# ── Fix FFD convergence: simpler random effects ───────────────────────────────
cat("─── Refitting FFD (subject only) ───\n")
m_FFD_simple <- glmmTMB(
  FFD ~ wlen_z + freq_z + wpos_z + surp_z + (1 | subj),
  data = df_fix, family = Gamma(link = "log"),
  control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))
)
cat("  AIC:", AIC(m_FFD_simple), "\n")
print(summary(m_FFD_simple)$coefficients$cond)
saveRDS(m_FFD_simple, "m_FFD_simple.rds")
