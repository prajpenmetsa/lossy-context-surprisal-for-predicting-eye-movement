# fix_figures.R
# Run after lossy_models.R — fixes fig2 y-axis and improves both figures

library(tidyverse)

tbl <- read_csv("model_comparison_table.csv", show_col_types = FALSE)

# ── Fig 1: Beta decay curves (exclude beta=1.0 — collinear with surp_z) ──────
p1 <- tbl %>%
  filter(beta < 1.0, !is.na(delta_aic)) %>%
  ggplot(aes(x = beta, y = delta_aic, color = measure, group = measure)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 4) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~ measure_type, labeller = labeller(
    measure_type = c(early = "Early measures (FFD, GD)",
                     late  = "Late measures (TRT, Regression)")
  )) +
  scale_color_manual(values = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63"),
                     labels = c(FFD = "FFD (First Fixation)",
                                GD  = "GD (Gaze Duration)",
                                TRT = "TRT (Total Reading Time)",
                                reg = "Regression")) +
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
  labs(
    title    = "Lossy-Context Surprisal: Incremental Fit by Memory Decay (β)",
    subtitle = "ΔAIC > 0 means lossy surprisal improves fit over baseline (β=1.0 excluded: collinear with standard surprisal)",
    x        = "Retention parameter β  (lower = more forgetting)",
    y        = "ΔAIC (higher = better fit)",
    color    = "Eye-tracking measure"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"))

ggsave("fig1_beta_decay_fixed.pdf", p1, width = 11, height = 5.5)
ggsave("fig1_beta_decay_fixed.png", p1, width = 11, height = 5.5, dpi = 150)
cat("✓ Fig 1 saved\n")

# ── Fig 2: Coefficients (exclude beta=1.0, fix y-axis) ───────────────────────
p2 <- tbl %>%
  filter(beta < 1.0) %>%
  ggplot(aes(x = beta,
             y = lossy_estimate,
             ymin = lossy_estimate - 1.96 * lossy_se,
             ymax = lossy_estimate + 1.96 * lossy_se,
             color = measure, fill = measure)) +
  geom_ribbon(alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 4) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~ measure_type, scales = "free_y",
             labeller = labeller(
               measure_type = c(early = "Early measures (FFD, GD)",
                                late  = "Late measures (TRT, Regression)")
             )) +
  scale_color_manual(values = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63"),
                     labels = c(FFD = "FFD", GD = "GD", TRT = "TRT", reg = "Regression")) +
  scale_fill_manual(values  = c(FFD = "#2196F3", GD = "#03A9F4",
                                 TRT = "#FF5722", reg = "#E91E63"),
                    labels  = c(FFD = "FFD", GD = "GD", TRT = "TRT", reg = "Regression")) +
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
  labs(
    title   = "Effect of Lossy-Context Surprisal on Eye-Tracking Measures",
    subtitle = "Standardised coefficient ± 95% CI; β=1.0 excluded (collinear with standard surprisal)",
    x       = "Retention parameter β  (lower = more forgetting)",
    y       = "Standardised coefficient",
    color   = "Measure", fill = "Measure"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"))

ggsave("fig2_lossy_coef_fixed.pdf", p2, width = 11, height = 5.5)
ggsave("fig2_lossy_coef_fixed.png", p2, width = 11, height = 5.5, dpi = 150)
cat("✓ Fig 2 saved\n")

# ── Print clean results summary ───────────────────────────────────────────────
cat("\n═══ Clean Results Summary (β < 1.0 only) ═══\n")
tbl %>%
  filter(beta < 1.0) %>%
  mutate(
    sig_label = case_when(
      is.na(p_lrt)   ~ "–",
      p_lrt < .001   ~ "***",
      p_lrt < .01    ~ "**",
      p_lrt < .05    ~ "*",
      TRUE           ~ "n.s."
    )
  ) %>%
  select(beta, measure, measure_type, delta_aic, p_lrt, sig_label, lossy_estimate) %>%
  print(n = 40)
