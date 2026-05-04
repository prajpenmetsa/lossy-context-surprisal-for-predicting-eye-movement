# =============================================================================
# sensitivity_corr.R
# 1. Correlation matrix of all predictors (collinearity check)
# 2. Sensitivity analyses: β sweep under alternative fixation exclusion thresholds
#
# Loads from saved outputs of final_models.R — run that first.
# =============================================================================

library(glmmTMB)
library(tidyverse)

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

# ── Load saved data ────────────────────────────────────────────────────────────

saved   <- readRDS("csv/final_dfs.rds")
df      <- saved$df        # pre-exclusion, all words (used for reg)
df_fix  <- saved$df_fix    # standard exclusions (used for FFD/GD/GPT/TRT)

lossy_file <- if (file.exists("csv/lossy_surprisal_50.csv")) {
  "csv/lossy_surprisal_50.csv"
} else {
  "csv/lossy_surprisal.csv"
}
lossy_raw <- read_csv(lossy_file, show_col_types = FALSE)
betas     <- sort(unique(lossy_raw$beta))
betas     <- betas[betas < 1.0]
cat("Betas:", betas, "\n")


# =============================================================================
# 1. CORRELATION MATRIX
# =============================================================================

cat("\n═══ Correlation Matrix ═══\n\n")

# Join in one representative lossy beta (0.5) for the correlation matrix
lossy_05 <- lossy_raw %>%
  filter(beta == 0.5) %>%
  select(global_sent_idx, word_idx, lossy_surprisal)

df_cor <- df_fix %>%
  left_join(lossy_05, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
  filter(!is.na(lossy_surprisal)) %>%
  mutate(lossy_z = scale(lossy_surprisal)[,1]) %>%
  select(surp_z, dep_z, wlen_z, freq_z, wpos_z, lossy_z) %>%
  drop_na()

cor_mat <- cor(df_cor, method = "pearson")

cat("Correlation matrix (β = 0.5 lossy surprisal):\n")
print(round(cor_mat, 3))

# Flag high correlations
high_cor <- which(abs(cor_mat) > 0.7 & cor_mat != 1, arr.ind = TRUE)
if (nrow(high_cor) > 0) {
  cat("\n[WARN] Correlations > |0.7|:\n")
  for (i in seq_len(nrow(high_cor))) {
    r <- high_cor[i, 1]; c <- high_cor[i, 2]
    if (r < c) cat(sprintf("  %s — %s : r = %.3f\n",
                            rownames(cor_mat)[r], colnames(cor_mat)[c], cor_mat[r, c]))
  }
} else {
  cat("\nNo correlations exceed |0.7| — no collinearity concern.\n")
}

# ── Heatmap ───────────────────────────────────────────────────────────────────

pred_labels <- c(
  surp_z  = "Std. Surprisal",
  dep_z   = "Dep. Length",
  wlen_z  = "Word Length",
  freq_z  = "Log Freq.",
  wpos_z  = "Word Position",
  lossy_z = "Lossy Surp. (β=0.5)"
)

cor_long <- as.data.frame(cor_mat) %>%
  rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "r") %>%
  mutate(
    var1 = factor(pred_labels[var1], levels = pred_labels),
    var2 = factor(pred_labels[var2], levels = pred_labels)
  )

p_cor <- ggplot(cor_long, aes(x = var1, y = var2, fill = r)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3.5) +
  scale_fill_gradient2(
    low = "#2196F3", mid = "white", high = "#E91E63",
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  labs(
    title    = "Predictor Correlation Matrix",
    subtitle = "Fixated words only (standard exclusions) | lossy surprisal at β = 0.5",
    x = NULL, y = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    panel.grid   = element_blank()
  )

ggsave("fig_corr_matrix.png", p_cor, width = 7, height = 6, dpi = 150)
ggsave("fig_corr_matrix.pdf", p_cor, width = 7, height = 6)
cat("\n✓ Saved fig_corr_matrix.png\n")

write_csv(as.data.frame(round(cor_mat, 4)) %>% rownames_to_column("predictor"),
          "csv/predictor_correlations.csv")


# =============================================================================
# 2. SENSITIVITY ANALYSES — ALTERNATIVE FIXATION EXCLUSION THRESHOLDS
# =============================================================================
# Three threshold sets:
#   standard : FFD/GD 100–1200/2000ms, GPT/TRT 100–3000ms  (current)
#   loose    : FFD/GD  80–1500/2500ms, GPT/TRT  80–3500ms
#   strict   : FFD/GD 150– 800/1200ms, GPT/TRT 150–2000ms

cat("\n═══ Sensitivity Analyses ═══\n\n")

thresholds <- list(
  standard = list(ffd_lo=100, ffd_hi=1200, gd_lo=100, gd_hi=2000,
                  gpt_lo=100, gpt_hi=3000, trt_lo=100, trt_hi=3000),
  loose    = list(ffd_lo= 80, ffd_hi=1500, gd_lo= 80, gd_hi=2500,
                  gpt_lo= 80, gpt_hi=3500, trt_lo= 80, trt_hi=3500),
  strict   = list(ffd_lo=150, ffd_hi= 800, gd_lo=150, gd_hi=1200,
                  gpt_lo=150, gpt_hi=2000, trt_lo=150, trt_hi=2000)
)

make_df_fix <- function(df_base, th) {
  df_base %>%
    filter(!is.na(FFD)) %>%
    filter(FFD >= th$ffd_lo, FFD <= th$ffd_hi) %>%
    filter(GD  >= th$gd_lo,  GD  <= th$gd_hi)  %>%
    filter(GPT >= th$gpt_lo, GPT <= th$gpt_hi) %>%
    filter(TRT >= th$trt_lo, TRT <= th$trt_hi)
}

sens_rows <- list()

for (thresh_name in names(thresholds)) {
  th      <- thresholds[[thresh_name]]
  df_fix_t <- make_df_fix(df, th)
  cat(sprintf("Threshold: %-8s | n = %d\n", thresh_name, nrow(df_fix_t)))

  for (b in betas) {
    lossy_b <- lossy_raw %>%
      filter(beta == b) %>%
      select(global_sent_idx, word_idx, lossy_surprisal)

    df_b <- df %>%
      left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
      filter(!is.na(lossy_surprisal)) %>%
      mutate(lossy_z = scale(lossy_surprisal)[,1])

    df_fix_b <- df_fix_t %>%
      left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
      filter(!is.na(lossy_surprisal)) %>%
      mutate(lossy_z = scale(lossy_surprisal)[,1])

    for (dv in c("FFD", "GD", "GPT", "TRT", "reg")) {
      is_bin   <- (dv == "reg")
      data_use <- if (is_bin) df_b else df_fix_b
      fam      <- if (is_bin) binomial("logit") else Gamma(link = "log")

      f_base  <- as.formula(paste(dv,
        "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + (1|subj) + (1|sent_id)"))
      f_lossy <- as.formula(paste(dv,
        "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + lossy_z + (1|subj) + (1|sent_id)"))

      mb <- tryCatch(glmmTMB(f_base,  data = data_use, family = fam, control = ctrl),
                     error = function(e) NULL)
      ml <- tryCatch(glmmTMB(f_lossy, data = data_use, family = fam, control = ctrl),
                     error = function(e) NULL)

      if (is.null(mb) || is.null(ml)) {
        cat(sprintf("  [FAIL] %s  beta=%.1f  thresh=%s\n", dv, b, thresh_name))
        next
      }

      delta <- AIC(mb) - AIC(ml)
      cat(sprintf("  %-4s beta=%.1f ΔAIC=%6.2f\n", dv, b, delta))

      sens_rows[[length(sens_rows) + 1]] <- tibble(
        threshold    = thresh_name,
        beta         = b,
        measure      = dv,
        measure_type = case_when(
          dv %in% c("FFD", "GD")         ~ "early",
          dv %in% c("GPT", "TRT", "reg") ~ "late"
        ),
        n_rows       = nrow(data_use),
        delta_aic    = delta
      )
    }
  }
  cat("\n")
}

sens_tbl <- bind_rows(sens_rows)
write_csv(sens_tbl, "csv/sensitivity_results.csv")
cat("✓ Saved csv/sensitivity_results.csv\n")

# ── Figure: β curves per measure, one line per threshold ─────────────────────

pal_thresh <- c(standard = "#333333", loose = "#2196F3", strict = "#E91E63")

meas_labels <- c(
  FFD = "FFD (First Fixation)", GD  = "GD (Gaze Duration)",
  GPT = "GPT (Go-Past Time)",   TRT = "TRT (Total Reading Time)",
  reg = "Regression"
)
meas_order <- c("FFD", "GD", "GPT", "TRT", "reg")

p_sens <- sens_tbl %>%
  filter(!is.na(delta_aic)) %>%
  mutate(
    measure   = factor(measure, levels = meas_order, labels = meas_labels[meas_order]),
    threshold = factor(threshold, levels = c("standard", "loose", "strict"))
  ) %>%
  ggplot(aes(x = beta, y = delta_aic,
             color = threshold, linetype = threshold, group = threshold)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
  facet_wrap(~measure, nrow = 1) +
  scale_color_manual(values = pal_thresh,
                     labels = c("Standard (100–1200ms)",
                                "Loose (80–1500ms)",
                                "Strict (150–800ms)")) +
  scale_linetype_manual(values = c(standard = "solid", loose = "dashed", strict = "dotted"),
                        labels = c("Standard (100–1200ms)",
                                   "Loose (80–1500ms)",
                                   "Strict (150–800ms)")) +
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
  labs(
    title    = "Sensitivity Analysis: β Curves Under Alternative Fixation Exclusion Thresholds",
    subtitle = "ΔAIC = AIC(baseline) − AIC(+lossy)  |  higher = better fit",
    x        = "Retention parameter β  (lower = more forgetting)",
    y        = "ΔAIC",
    color    = "Exclusion threshold",
    linetype = "Exclusion threshold"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 9),
    axis.text.x     = element_text(size = 8)
  )

ggsave("fig_sensitivity.png", p_sens, width = 15, height = 5, dpi = 150)
ggsave("fig_sensitivity.pdf", p_sens, width = 15, height = 5)
cat("✓ Saved fig_sensitivity.png\n")

# ── Summary: optimal β per measure per threshold ──────────────────────────────

cat("\n═══ Optimal β by measure and threshold ═══\n\n")
sens_tbl %>%
  filter(!is.na(delta_aic)) %>%
  group_by(threshold, measure) %>%
  slice_max(delta_aic, n = 1) %>%
  select(threshold, measure, beta, delta_aic) %>%
  arrange(measure, threshold) %>%
  print(n = 50)
