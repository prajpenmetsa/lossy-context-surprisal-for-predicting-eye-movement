# =============================================================================
# exclusion_criteria.R
# Apply standard eye-tracking exclusion criteria:
#   1. Exclude fixations < 100ms (FFD, GD, TRT)
#   2. Exclude fixations > 1200ms (outlier cap)
# Then refit all baseline + lossy models on clean data.
# =============================================================================

library(glmmTMB)
library(MuMIn)
library(tidyverse)

# ── 1. Load data ──────────────────────────────────────────────────────────────

df_raw <- read_csv("analysis_data.csv", show_col_types = FALSE)
cat("Raw rows:", nrow(df_raw), "\n")

# ── 2. Apply exclusion criteria ───────────────────────────────────────────────

# Standard criteria from ZuCo benchmark paper (Hollenstein et al. 2022)
# and general eye-tracking literature (Sereno & Rayner 2003):
#   - FFD < 100ms: too short to be a real reading fixation
#   - FFD > 1200ms: outlier (likely mind-wandering or blink artifact)
# Apply the same bounds to GD and TRT for consistency.

FFD_MIN  <- 100
FFD_MAX  <- 1200
GD_MIN   <- 100
GD_MAX   <- 2000   # GD can be longer (summed first-pass)
TRT_MIN  <- 100
TRT_MAX  <- 3000   # TRT can be longer still (multiple passes)

df <- df_raw %>%
  filter(!is.na(surprisal), !is.na(word_length), !is.na(log_freq)) %>%
  filter(word_position > 0) %>%
  rename(
    sent_id = global_sent_idx,
    subj    = subject,
    wlen    = word_length,
    wpos    = word_position,
    freq    = log_freq,
    surp    = surprisal
  ) %>%
  mutate(
    surp_z = scale(surp)[,1],
    wlen_z = scale(wlen)[,1],
    freq_z = scale(freq)[,1],
    wpos_z = scale(wpos)[,1],
    subj    = factor(subj),
    sent_id = factor(sent_id)
  )

# Fixated subset with exclusions applied
df_fix_clean <- df %>%
  filter(!is.na(FFD)) %>%
  filter(FFD >= FFD_MIN, FFD <= FFD_MAX) %>%
  filter(GD  >= GD_MIN,  GD  <= GD_MAX)  %>%
  filter(TRT >= TRT_MIN, TRT <= TRT_MAX)

cat("\nAfter exclusions:\n")
cat("  Total rows (for reg model):", nrow(df), "\n")
cat("  Fixated rows (for FFD/GD/TRT):", nrow(df_fix_clean), "\n")
cat("  Removed by exclusion:", nrow(df %>% filter(!is.na(FFD))) - nrow(df_fix_clean), "\n\n")

# Report exclusion rate per measure
orig <- df %>% filter(!is.na(FFD))
cat(sprintf("  FFD < 100ms excluded: %d (%.1f%%)\n",
    sum(orig$FFD < FFD_MIN), 100*mean(orig$FFD < FFD_MIN)))
cat(sprintf("  FFD > 1200ms excluded: %d (%.1f%%)\n",
    sum(orig$FFD > FFD_MAX), 100*mean(orig$FFD > FFD_MAX)))

# ── 3. Refit baseline models on clean data ────────────────────────────────────

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

fit_model <- function(outcome, data, family, label) {
  cat("─── Fitting", label, "───\n")
  formula <- as.formula(paste(
    outcome, "~ wlen_z + freq_z + wpos_z + surp_z + (1 | subj) + (1 | sent_id)"
  ))
  m <- tryCatch(
    glmmTMB(formula, data = data, family = family, control = ctrl),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(m)) {
    cat("  AIC:", round(AIC(m), 1), "\n")
    cat("  Marginal R²:", round(r.squaredGLMM(m)[1], 4), "\n")
    coefs <- summary(m)$coefficients$cond
    cat("  surp_z: β=", round(coefs["surp_z","Estimate"],4),
        " p=", round(coefs["surp_z","Pr(>|z|)"],4), "\n\n")
  }
  m
}

m_FFD <- fit_model("FFD", df_fix_clean, Gamma(link="log"), "FFD (clean)")
m_GD  <- fit_model("GD",  df_fix_clean, Gamma(link="log"), "GD (clean)")
m_TRT <- fit_model("TRT", df_fix_clean, Gamma(link="log"), "TRT (clean)")

cat("─── Fitting Regression (clean) ───\n")
m_reg <- tryCatch(
  glmmTMB(reg ~ wlen_z + freq_z + wpos_z + surp_z + (1|subj) + (1|sent_id),
          data = df, family = binomial(link="logit"), control = ctrl),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)
if (!is.null(m_reg)) {
  cat("  AIC:", round(AIC(m_reg), 1), "\n")
  coefs <- summary(m_reg)$coefficients$cond
  cat("  surp_z: β=", round(coefs["surp_z","Estimate"],4),
      " p=", round(coefs["surp_z","Pr(>|z|)"],4), "\n\n")
}

# ── 4. Refit lossy models on clean data ───────────────────────────────────────

lossy_raw <- read_csv("lossy_surprisal.csv", show_col_types = FALSE)
betas <- sort(unique(lossy_raw$beta))
betas <- betas[betas < 1.0]   # exclude beta=1.0 (collinear)

cat("Fitting lossy models for betas:", betas, "\n\n")

results <- list()

for (b in betas) {
  cat("════ beta =", b, "════\n")

  lossy_b <- lossy_raw %>%
    filter(beta == b) %>%
    select(global_sent_idx, word_idx, lossy_surprisal)

  df_b <- df %>%
    mutate(sent_id_num = as.numeric(as.character(sent_id))) %>%
    left_join(lossy_b, by = c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  df_fix_b <- df_fix_clean %>%
    mutate(sent_id_num = as.numeric(as.character(sent_id))) %>%
    left_join(lossy_b, by = c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  beta_res <- list(beta = b)

  for (dv in c("FFD","GD","TRT","reg")) {
    is_bin   <- (dv == "reg")
    data_use <- if (is_bin) df_b else df_fix_b
    fam      <- if (is_bin) binomial(link="logit") else Gamma(link="log")

    f_base  <- as.formula(paste(dv,"~ wlen_z+freq_z+wpos_z+surp_z+(1|subj)+(1|sent_id)"))
    f_lossy <- as.formula(paste(dv,"~ wlen_z+freq_z+wpos_z+surp_z+lossy_z+(1|subj)+(1|sent_id)"))

    mb <- tryCatch(glmmTMB(f_base,  data=data_use, family=fam, control=ctrl), error=function(e) NULL)
    ml <- tryCatch(glmmTMB(f_lossy, data=data_use, family=fam, control=ctrl), error=function(e) NULL)

    if (is.null(mb) || is.null(ml)) { cat("  [FAIL]", dv, "\n"); next }

    delta  <- AIC(mb) - AIC(ml)
    lrt    <- tryCatch(anova(mb, ml), error=function(e) NULL)
    p_val  <- if (!is.null(lrt)) lrt$`Pr(>Chisq)`[2] else NA
    r2b    <- tryCatch(r.squaredGLMM(mb)[1], error=function(e) NA)
    r2l    <- tryCatch(r.squaredGLMM(ml)[1], error=function(e) NA)
    coef_l <- summary(ml)$coefficients$cond["lossy_z",]

    cat(sprintf("  %-4s | ΔAIC=%6.2f | p=%.4f | ΔR²=%.4f | β=%.4f (se=%.4f)\n",
                dv, delta, ifelse(is.na(p_val),NA,p_val),
                ifelse(is.na(r2l-r2b),NA,r2l-r2b),
                coef_l["Estimate"], coef_l["Std. Error"]))

    beta_res[[dv]] <- list(
      delta_aic=delta, p_lrt=p_val, delta_r2=r2l-r2b,
      lossy_estimate=coef_l["Estimate"], lossy_se=coef_l["Std. Error"]
    )
  }
  cat("\n")
  results[[as.character(b)]] <- beta_res
}

# ── 5. Build and save clean summary table ─────────────────────────────────────

rows <- list()
for (b_chr in names(results)) {
  res <- results[[b_chr]]
  for (dv in c("FFD","GD","TRT","reg")) {
    if (!is.null(res[[dv]])) {
      rows[[length(rows)+1]] <- tibble(
        beta=as.numeric(b_chr), measure=dv,
        measure_type=ifelse(dv %in% c("FFD","GD"),"early","late"),
        delta_aic=res[[dv]]$delta_aic, p_lrt=res[[dv]]$p_lrt,
        delta_r2=res[[dv]]$delta_r2,
        lossy_estimate=res[[dv]]$lossy_estimate, lossy_se=res[[dv]]$lossy_se
      )
    }
  }
}

tbl_clean <- bind_rows(rows)
write_csv(tbl_clean, "model_comparison_clean.csv")

# ── 6. Remake figures with clean data ─────────────────────────────────────────

p1 <- tbl_clean %>%
  filter(!is.na(delta_aic)) %>%
  ggplot(aes(x=beta, y=delta_aic, color=measure, group=measure)) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  geom_hline(yintercept=0, linetype="dashed", color="gray50") +
  facet_wrap(~measure_type, labeller=labeller(
    measure_type=c(early="Early measures (FFD, GD)", late="Late measures (TRT, Regression)"))) +
  scale_color_manual(values=c(FFD="#2196F3",GD="#03A9F4",TRT="#FF5722",reg="#E91E63"),
                     labels=c(FFD="FFD",GD="GD",TRT="TRT",reg="Regression")) +
  scale_x_continuous(breaks=c(0.1,0.3,0.5,0.7,0.9)) +
  labs(title="Lossy-Context Surprisal: Incremental Fit by Memory Decay (β)",
       subtitle="Fixations <100ms and >1200ms excluded  |  β=1.0 excluded (collinear)",
       x="Retention parameter β  (lower = more forgetting)",
       y="ΔAIC (higher = better fit)", color="Measure") +
  theme_bw(base_size=13) +
  theme(legend.position="bottom", strip.text=element_text(face="bold"))

p2 <- tbl_clean %>%
  ggplot(aes(x=beta, y=lossy_estimate,
             ymin=lossy_estimate-1.96*lossy_se,
             ymax=lossy_estimate+1.96*lossy_se,
             color=measure, fill=measure)) +
  geom_ribbon(alpha=0.15, color=NA) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  geom_hline(yintercept=0, linetype="dashed") +
  facet_wrap(~measure_type, scales="free_y", labeller=labeller(
    measure_type=c(early="Early measures (FFD, GD)", late="Late measures (TRT, Regression)"))) +
  scale_color_manual(values=c(FFD="#2196F3",GD="#03A9F4",TRT="#FF5722",reg="#E91E63"),
                     labels=c(FFD="FFD",GD="GD",TRT="TRT",reg="Regression")) +
  scale_fill_manual(values=c(FFD="#2196F3",GD="#03A9F4",TRT="#FF5722",reg="#E91E63"),
                    labels=c(FFD="FFD",GD="GD",TRT="TRT",reg="Regression")) +
  scale_x_continuous(breaks=c(0.1,0.3,0.5,0.7,0.9)) +
  labs(title="Effect of Lossy-Context Surprisal (clean data)",
       subtitle="Standardised coefficient ± 95% CI  |  fixation outliers removed",
       x="Retention parameter β", y="Standardised coefficient",
       color="Measure", fill="Measure") +
  theme_bw(base_size=13) +
  theme(legend.position="bottom", strip.text=element_text(face="bold"))

ggsave("fig1_clean.png", p1, width=11, height=5.5, dpi=150)
ggsave("fig2_clean.png", p2, width=11, height=5.5, dpi=150)

saveRDS(list(FFD=m_FFD, GD=m_GD, TRT=m_TRT, reg=m_reg), "baseline_models_clean.rds")
saveRDS(df_fix_clean, "analysis_df_fix_clean.rds")
saveRDS(df, "analysis_df_clean.rds")

cat("✓ Done. Files saved:\n")
cat("  model_comparison_clean.csv\n")
cat("  fig1_clean.png\n")
cat("  fig2_clean.png\n")
cat("  baseline_models_clean.rds\n")
