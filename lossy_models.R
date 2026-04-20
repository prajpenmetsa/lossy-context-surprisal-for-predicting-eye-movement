# =============================================================================
# lossy_models.R
# Merge lossy surprisal, fit augmented models, compare to baseline,
# and produce the beta decay curve plots for the mid-eval.
# Run AFTER compute_lossy_surprisal.py finishes.
# =============================================================================

library(glmmTMB)
library(MuMIn)
library(tidyverse)

# ── 1. Load baseline models and data ─────────────────────────────────────────

baseline <- readRDS("baseline_models.rds")
df       <- readRDS("analysis_df.rds")
df_fix   <- readRDS("analysis_df_fix.rds")

cat("Baseline models loaded.\n")

# ── 2. Load and merge lossy surprisal ────────────────────────────────────────

lossy_raw <- read_csv("lossy_surprisal.csv", show_col_types = FALSE)
cat("Lossy surprisal rows:", nrow(lossy_raw), "\n")
cat("Beta values:", sort(unique(lossy_raw$beta)), "\n\n")

# ── 3. Fit augmented models for each beta ────────────────────────────────────

betas   <- sort(unique(lossy_raw$beta))
results <- list()

for (b in betas) {
  cat("════ beta =", b, "════\n")

  # Get lossy surprisal for this beta value
  lossy_b <- lossy_raw %>%
    filter(beta == b) %>%
    select(global_sent_idx, word_idx, lossy_surprisal)

  # Merge with the full dataframes
  df_b <- df %>%
    left_join(lossy_b,
              by = c("sent_id" = "global_sent_idx",   # sent_id was factored; need to match
                     "word_idx" = "word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  df_fix_b <- df_fix %>%
    left_join(lossy_b,
              by = c("sent_id" = "global_sent_idx",
                     "word_idx" = "word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  beta_results <- list(beta = b)

  for (dv in c("FFD", "GD", "TRT", "reg")) {
    is_binary <- (dv == "reg")
    data_use  <- if (is_binary) df_b else df_fix_b

    if (nrow(data_use) < 100) {
      cat("  [SKIP]", dv, "— too few rows\n")
      next
    }

    # Baseline (without lossy_z) on same subset
    fam <- if (is_binary) binomial(link = "logit") else Gamma(link = "log")
    formula_base  <- as.formula(paste(dv, "~ wlen_z + freq_z + wpos_z + surp_z + (1|subj) + (1|sent_id)"))
    formula_lossy <- as.formula(paste(dv, "~ wlen_z + freq_z + wpos_z + surp_z + lossy_z + (1|subj) + (1|sent_id)"))

    ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

    m_base <- tryCatch(glmmTMB(formula_base,  data = data_use, family = fam, control = ctrl), error = function(e) NULL)
    m_aug  <- tryCatch(glmmTMB(formula_lossy, data = data_use, family = fam, control = ctrl), error = function(e) NULL)

    if (is.null(m_base) || is.null(m_aug)) {
      cat("  [FAIL]", dv, "\n")
      next
    }

    aic_base <- AIC(m_base)
    aic_aug  <- AIC(m_aug)
    delta    <- aic_base - aic_aug     # positive = augmented is better

    lrt <- tryCatch(anova(m_base, m_aug), error = function(e) NULL)
    p_val <- if (!is.null(lrt)) lrt$`Pr(>Chisq)`[2] else NA

    r2_base <- tryCatch(r.squaredGLMM(m_base)[1], error = function(e) NA)
    r2_aug  <- tryCatch(r.squaredGLMM(m_aug)[1],  error = function(e) NA)
    delta_r2 <- r2_aug - r2_base

    lossy_coef <- summary(m_aug)$coefficients$cond["lossy_z", ]

    cat(sprintf("  %-4s | ΔAIC=%.2f | p=%.4f | ΔR²=%.4f | β=%.4f (se=%.4f)\n",
                dv, delta, ifelse(is.na(p_val), NA, p_val),
                ifelse(is.na(delta_r2), NA, delta_r2),
                lossy_coef["Estimate"], lossy_coef["Std. Error"]))

    beta_results[[dv]] <- list(
      aic_base = aic_base, aic_aug = aic_aug, delta_aic = delta,
      p_lrt = p_val, r2_base = r2_base, r2_aug = r2_aug, delta_r2 = delta_r2,
      lossy_estimate = lossy_coef["Estimate"],
      lossy_se       = lossy_coef["Std. Error"],
      lossy_z_stat   = lossy_coef["z value"],
      lossy_p        = lossy_coef["Pr(>|z|)"],
      model_aug      = m_aug
    )
  }

  results[[as.character(b)]] <- beta_results
  cat("\n")
}

saveRDS(results, "lossy_model_results.rds")
cat("✓ Lossy model results saved to lossy_model_results.rds\n\n")

# ── 4. Build summary table ────────────────────────────────────────────────────

rows <- list()
for (b_chr in names(results)) {
  res <- results[[b_chr]]
  for (dv in c("FFD", "GD", "TRT", "reg")) {
    if (!is.null(res[[dv]])) {
      rows[[length(rows)+1]] <- tibble(
        beta          = as.numeric(b_chr),
        measure       = dv,
        measure_type  = ifelse(dv %in% c("FFD","GD"), "early", "late"),
        delta_aic     = res[[dv]]$delta_aic,
        p_lrt         = res[[dv]]$p_lrt,
        delta_r2      = res[[dv]]$delta_r2,
        lossy_estimate= res[[dv]]$lossy_estimate,
        lossy_se      = res[[dv]]$lossy_se,
        sig           = ifelse(!is.na(res[[dv]]$p_lrt) & res[[dv]]$p_lrt < .05, "*", "")
      )
    }
  }
}

summary_tbl <- bind_rows(rows)
write_csv(summary_tbl, "model_comparison_table.csv")
cat("Summary table:\n")
print(summary_tbl %>% arrange(measure, beta), n = 40)

# ── 5. Plot: beta decay curves (THE key figure for the paper) ─────────────────

# Figure 1: ΔAIC by beta, separately for early vs late measures
p1 <- summary_tbl %>%
  ggplot(aes(x = beta, y = delta_aic, color = measure, group = measure)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~ measure_type, labeller = labeller(
    measure_type = c(early = "Early measures (FFD, GD)",
                     late  = "Late measures (TRT, Regression)")
  )) +
  scale_color_manual(values = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63")) +
  labs(
    title    = "Lossy-Context Surprisal: Incremental Fit by Memory Decay (β)",
    subtitle = "ΔAIC = baseline AIC − augmented AIC; positive = lossy surprisal helps",
    x        = "Retention parameter β  (β=1.0 → standard surprisal)",
    y        = "ΔAIC (higher = better fit)",
    color    = "Measure"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("fig1_beta_decay_curves.pdf", p1, width = 10, height = 5)
ggsave("fig1_beta_decay_curves.png", p1, width = 10, height = 5, dpi = 150)
cat("✓ Figure 1 saved: fig1_beta_decay_curves.pdf/.png\n")

# Figure 2: Lossy surprisal coefficient by beta (effect size direction)
p2 <- summary_tbl %>%
  ggplot(aes(x = beta, y = lossy_estimate, ymin = lossy_estimate - 1.96*lossy_se,
             ymax = lossy_estimate + 1.96*lossy_se, color = measure)) +
  geom_ribbon(aes(fill = measure), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~ measure_type, scales = "free_y") +
  scale_color_manual(values = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63")) +
  scale_fill_manual(values  = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63")) +
  labs(
    title  = "Effect of Lossy-Context Surprisal on Eye-Tracking Measures",
    x      = "Retention parameter β",
    y      = "Standardised coefficient (± 95% CI)",
    color  = "Measure", fill = "Measure"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("fig2_lossy_coefficients.pdf", p2, width = 10, height = 5)
ggsave("fig2_lossy_coefficients.png", p2, width = 10, height = 5, dpi = 150)
cat("✓ Figure 2 saved: fig2_lossy_coefficients.pdf/.png\n")

# ── 6. Print AIC comparison table (formatted for slides) ──────────────────────

cat("\n═══ AIC Comparison Table (for slides/paper) ═══\n")
aic_tbl <- summary_tbl %>%
  select(beta, measure, delta_aic, p_lrt, sig) %>%
  mutate(
    delta_aic = round(delta_aic, 2),
    p_lrt     = round(p_lrt, 4)
  ) %>%
  pivot_wider(names_from = measure,
              values_from = c(delta_aic, p_lrt, sig),
              names_glue  = "{measure}_{.value}")

print(aic_tbl, n = 20)
write_csv(aic_tbl, "aic_comparison_wide.csv")
cat("✓ AIC table saved: aic_comparison_wide.csv\n")
