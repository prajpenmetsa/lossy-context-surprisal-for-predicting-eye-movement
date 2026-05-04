# =============================================================================
# per_subject_optima.R
# For each of the 12 subjects, find the β that best predicts each ET measure.
#
# Approach: stratify by subject. For each subject × beta × measure, fit:
#   baseline : DV ~ wlen_z + freq_z + wpos_z + dep_z + surp_z + (1|sent_id)
#   lossy    : DV ~ wlen_z + freq_z + wpos_z + dep_z + surp_z + lossy_z + (1|sent_id)
#
# (1|sent_id) kept — each subject reads the same sentences.
# (1|subj) dropped — we're within one subject.
#
# Optimal β = argmax ΔAIC across betas, but only if max ΔAIC > 2
# (below 2 = no meaningful lossy surprisal effect for that subject).
#
# Loads from saved outputs of final_models.R — run that first.
# =============================================================================

library(glmmTMB)
library(tidyverse)

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

# ── Load data ──────────────────────────────────────────────────────────────────

saved  <- readRDS("csv/final_dfs.rds")
df     <- saved$df
df_fix <- saved$df_fix

lossy_file <- if (file.exists("csv/lossy_surprisal_50.csv")) {
  "csv/lossy_surprisal_50.csv"
} else {
  "csv/lossy_surprisal.csv"
}
lossy_raw <- read_csv(lossy_file, show_col_types = FALSE)
betas     <- sort(unique(lossy_raw$beta))
betas     <- betas[betas < 1.0]

subjects  <- levels(df$subj)
cat("Subjects:", subjects, "\n")
cat("Betas:",   betas, "\n\n")

# ── Main loop: subject × beta × measure ───────────────────────────────────────

all_rows <- list()
total    <- length(subjects) * length(betas) * 5
done     <- 0

for (subj_id in subjects) {

  df_s     <- df     %>% filter(subj == subj_id)
  df_fix_s <- df_fix %>% filter(subj == subj_id)

  for (b in betas) {

    lossy_b <- lossy_raw %>%
      filter(beta == b) %>%
      select(global_sent_idx, word_idx, lossy_surprisal)

    df_sb <- df_s %>%
      left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
      filter(!is.na(lossy_surprisal)) %>%
      mutate(lossy_z = scale(lossy_surprisal)[,1])

    df_fix_sb <- df_fix_s %>%
      left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
      filter(!is.na(lossy_surprisal)) %>%
      mutate(lossy_z = scale(lossy_surprisal)[,1])

    for (dv in c("FFD", "GD", "GPT", "TRT", "reg")) {
      done <- done + 1
      cat(sprintf("\r  [%3d/%d] subj=%-4s beta=%.1f dv=%-4s",
                  done, total, subj_id, b, dv))

      is_bin   <- (dv == "reg")
      data_use <- if (is_bin) df_sb else df_fix_sb
      fam      <- if (is_bin) binomial("logit") else Gamma(link = "log")

      # Need at least 50 rows and variance in lossy_z
      if (nrow(data_use) < 50 || sd(data_use$lossy_z, na.rm = TRUE) == 0) next

      f_base  <- as.formula(paste(dv,
        "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + (1|sent_id)"))
      f_lossy <- as.formula(paste(dv,
        "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + lossy_z + (1|sent_id)"))

      mb <- tryCatch(glmmTMB(f_base,  data = data_use, family = fam, control = ctrl),
                     error = function(e) NULL)
      ml <- tryCatch(glmmTMB(f_lossy, data = data_use, family = fam, control = ctrl),
                     error = function(e) NULL)

      if (is.null(mb) || is.null(ml)) next

      aic_base  <- AIC(mb)
      aic_lossy <- AIC(ml)
      if (is.na(aic_base) || is.na(aic_lossy)) next

      delta <- aic_base - aic_lossy

      coef_l <- tryCatch(
        summary(ml)$coefficients$cond["lossy_z", ],
        error = function(e) c(Estimate = NA, `Std. Error` = NA)
      )

      if (is.na(coef_l["Estimate"]) || abs(coef_l["Estimate"]) > 5) next

      all_rows[[length(all_rows) + 1]] <- tibble(
        subject      = subj_id,
        beta         = b,
        measure      = dv,
        measure_type = case_when(
          dv %in% c("FFD", "GD")          ~ "early",
          dv %in% c("GPT", "TRT", "reg")  ~ "late"
        ),
        n_obs        = nrow(data_use),
        delta_aic    = delta,
        lossy_est    = coef_l["Estimate"],
        lossy_se     = coef_l["Std. Error"]
      )
    }
  }
}
cat("\n\nDone.\n\n")

subj_tbl <- bind_rows(all_rows)
write_csv(subj_tbl, "csv/per_subject_results.csv")
cat("✓ Saved csv/per_subject_results.csv\n\n")

# ── Optimal β per subject per measure ─────────────────────────────────────────
# Only assign an optimal β if the best ΔAIC exceeds 2 (meaningful fit gain)

DAIC_THRESHOLD <- 2

optima <- subj_tbl %>%
  group_by(subject, measure) %>%
  summarise(
    max_daic   = max(delta_aic, na.rm = TRUE),
    opt_beta   = beta[which.max(delta_aic)],
    has_effect = max_daic > DAIC_THRESHOLD,
    .groups    = "drop"
  )

cat("═══ Optimal β per subject per measure ═══\n\n")
optima %>%
  arrange(measure, subject) %>%
  mutate(opt_beta_label = ifelse(has_effect, as.character(opt_beta), "no effect")) %>%
  select(measure, subject, opt_beta_label, max_daic) %>%
  print(n = 100)

write_csv(optima, "csv/per_subject_optima.csv")
cat("\n✓ Saved csv/per_subject_optima.csv\n\n")

# ── Summary: how consistent are subjects? ─────────────────────────────────────

cat("═══ Consistency summary ═══\n\n")
optima %>%
  filter(has_effect) %>%
  group_by(measure) %>%
  summarise(
    n_subjects_with_effect = n(),
    opt_betas              = paste(sort(unique(opt_beta)), collapse = ", "),
    modal_beta             = {
      tb <- table(opt_beta)
      as.numeric(names(tb)[which.max(tb)])
    },
    beta_sd                = sd(opt_beta),
    .groups = "drop"
  ) %>%
  print()

# ── Figure 1: Heatmap — subject × measure, colour = optimal β ─────────────────

beta_pal <- c("0.1" = "#1565C0", "0.3" = "#42A5F5",
              "0.5" = "#66BB6A", "0.7" = "#FFA726",
              "0.9" = "#EF5350", "no effect" = "#E0E0E0")

meas_order <- c("FFD", "GD", "GPT", "TRT", "reg")

heat_df <- optima %>%
  mutate(
    opt_label = ifelse(has_effect, as.character(opt_beta), "no effect"),
    measure   = factor(measure, levels = meas_order)
  )

p_heat <- ggplot(heat_df, aes(x = measure, y = subject, fill = opt_label)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = ifelse(has_effect,
                               sprintf("β=%.1f\n(%.1f)", opt_beta, max_daic),
                               "—")),
            size = 2.8, lineheight = 0.9) +
  scale_fill_manual(values = beta_pal,
                    name   = "Optimal β\n(or no effect)") +
  labs(
    title    = "Per-Subject Optimal Memory Decay Parameter",
    subtitle = sprintf("Optimal β = argmax ΔAIC | shown only where ΔAIC > %d", DAIC_THRESHOLD),
    x = "ET Measure", y = "Subject"
  ) +
  theme_bw(base_size = 12) +
  theme(panel.grid = element_blank(),
        axis.text  = element_text(size = 10))

ggsave("fig_per_subject_heatmap.png", p_heat, width = 9, height = 7, dpi = 150)
ggsave("fig_per_subject_heatmap.pdf", p_heat, width = 9, height = 7)
cat("✓ Saved fig_per_subject_heatmap.png\n")

# ── Figure 2: Dot plot — distribution of optimal β per measure ────────────────

dot_df <- optima %>%
  filter(has_effect) %>%
  mutate(measure = factor(measure, levels = meas_order))

p_dot <- ggplot(dot_df, aes(x = opt_beta, y = measure, color = measure)) +
  geom_jitter(height = 0.15, size = 3, alpha = 0.8) +
  stat_summary(fun = median, geom = "crossbar",
               width = 0.4, linewidth = 0.8, color = "black") +
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9),
                     limits = c(0.05, 0.95)) +
  scale_color_manual(values = c(FFD = "#2196F3", GD = "#03A9F4",
                                GPT = "#9C27B0", TRT = "#FF5722",
                                reg = "#E91E63"),
                     guide = "none") +
  labs(
    title    = "Distribution of Optimal β Across Subjects",
    subtitle = sprintf("Each dot = one subject | crossbar = median | only subjects with ΔAIC > %d shown",
                       DAIC_THRESHOLD),
    x = "Optimal retention parameter β",
    y = NULL
  ) +
  theme_bw(base_size = 13)

ggsave("fig_per_subject_dotplot.png", p_dot, width = 8, height = 5, dpi = 150)
ggsave("fig_per_subject_dotplot.pdf", p_dot, width = 8, height = 5)
cat("✓ Saved fig_per_subject_dotplot.png\n")

# ── Figure 3: β curves per subject for the two strongest measures ─────────────
# reg and TRT — the measures with clearest group-level effects

for (focal_measure in c("reg", "TRT", "GD")) {
  curve_df <- subj_tbl %>%
    filter(measure == focal_measure, !is.na(delta_aic))

  if (nrow(curve_df) == 0) next

  p_curve <- ggplot(curve_df,
                    aes(x = beta, y = delta_aic,
                        color = subject, group = subject)) +
    geom_line(linewidth = 0.8, alpha = 0.7) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_hline(yintercept = 0,  linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = DAIC_THRESHOLD, linetype = "dotted", color = "gray70") +
    scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
    labs(
      title    = sprintf("Per-Subject β Curves — %s", focal_measure),
      subtitle = "Dashed = 0, dotted = ΔAIC threshold of 2",
      x = "Retention parameter β", y = "ΔAIC",
      color = "Subject"
    ) +
    theme_bw(base_size = 13) +
    theme(legend.position = "right")

  fname <- sprintf("fig_per_subject_%s.png", tolower(focal_measure))
  ggsave(fname, p_curve, width = 9, height = 5, dpi = 150)
  cat("✓ Saved", fname, "\n")
}
