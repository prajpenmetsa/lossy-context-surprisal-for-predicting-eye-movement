# =============================================================================
# random_slopes.R
# Extend lossy surprisal models with random slopes for lossy_z by subject.
#
# Strategy per beta × measure:
#   1. Fit intercept-only model (random intercepts, same as final_models.R)
#   2. Try random slopes model: (1 + lossy_z | subj)
#   3. If slopes converge: report ΔAIC(intercepts → slopes) and use slopes
#      model as the "full" lossy model for that cell
#   4. If slopes fail: fall back to intercepts, flag it
#
# Only 12 subjects, so many slopes models will fail. That's expected and
# reportable: "random slopes were attempted for all measures; convergence
# was achieved for X of 25 cells."
#
# Loads from saved outputs of final_models.R — run that first.
# =============================================================================

library(glmmTMB)
library(tidyverse)

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

# ── Load saved data ────────────────────────────────────────────────────────────

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
cat("Betas:", betas, "\n\n")

# ── Helper: did a glmmTMB model converge cleanly? ────────────────────────────
# Returns TRUE only if no non-positive-definite Hessian and no NA gradient
converged_ok <- function(m) {
  if (is.null(m)) return(FALSE)
  conv <- m$fit$convergence
  pdh  <- !is.null(m$sdr) && !inherits(tryCatch(m$sdr, error=function(e) NULL), "error")
  # Check for non-pd Hessian via the internal slot
  warn_msg <- tryCatch(
    { capture.output(summary(m)); "" },
    warning = function(w) conditionMessage(w)
  )
  npd <- grepl("non-positive-definite", warn_msg, ignore.case = TRUE)
  return(conv == 0 && !npd)
}

# ── Main loop ─────────────────────────────────────────────────────────────────

all_rows <- list()

for (b in betas) {
  cat("════ beta =", b, "════\n")

  lossy_b <- lossy_raw %>%
    filter(beta == b) %>%
    select(global_sent_idx, word_idx, lossy_surprisal)

  df_b <- df %>%
    left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  df_fix_b <- df_fix %>%
    left_join(lossy_b, by = c("sent_id_num" = "global_sent_idx", "word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  for (dv in c("FFD", "GD", "GPT", "TRT", "reg")) {
    is_bin   <- (dv == "reg")
    data_use <- if (is_bin) df_b else df_fix_b
    fam      <- if (is_bin) binomial("logit") else Gamma(link = "log")

    # Model 1: random intercepts baseline (no lossy)
    f_base <- as.formula(paste(dv,
      "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + (1|subj) + (1|sent_id)"))

    # Model 2: random intercepts + lossy fixed effect
    f_ri <- as.formula(paste(dv,
      "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + lossy_z + (1|subj) + (1|sent_id)"))

    # Model 3: random slopes for lossy_z by subject
    f_rs <- as.formula(paste(dv,
      "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + lossy_z + (1 + lossy_z | subj) + (1|sent_id)"))

    m_base <- tryCatch(glmmTMB(f_base, data = data_use, family = fam, control = ctrl),
                       error = function(e) NULL)
    m_ri   <- tryCatch(glmmTMB(f_ri,   data = data_use, family = fam, control = ctrl),
                       error = function(e) NULL)
    m_rs   <- tryCatch(glmmTMB(f_rs,   data = data_use, family = fam, control = ctrl),
                       error = function(e) NULL)

    if (is.null(m_base) || is.null(m_ri)) {
      cat(sprintf("  [FAIL-BASE] %-4s\n", dv))
      next
    }

    rs_ok       <- converged_ok(m_rs)
    best_lossy  <- if (rs_ok) m_rs else m_ri
    slopes_note <- if (rs_ok) "slopes" else "intercepts"

    delta_lossy  <- AIC(m_base) - AIC(best_lossy)   # fit gain: baseline → best lossy
    delta_slopes <- if (!is.null(m_rs)) AIC(m_ri) - AIC(m_rs) else NA  # gain: RI → RS

    coef_l <- tryCatch(
      summary(best_lossy)$coefficients$cond["lossy_z", ],
      error = function(e) c(Estimate = NA, `Std. Error` = NA, `Pr(>|z|)` = NA)
    )

    cat(sprintf("  %-4s | ΔAIC(lossy)=%6.2f | ΔAIC(slopes)=%s | model=%s\n",
                dv,
                delta_lossy,
                ifelse(is.na(delta_slopes), "  N/A", sprintf("%6.2f", delta_slopes)),
                slopes_note))

    all_rows[[length(all_rows) + 1]] <- tibble(
      beta           = b,
      measure        = dv,
      measure_type   = case_when(
        dv %in% c("FFD", "GD")          ~ "early",
        dv %in% c("GPT", "TRT", "reg")  ~ "late"
      ),
      delta_aic_lossy   = delta_lossy,
      delta_aic_slopes  = delta_slopes,
      slopes_converged  = rs_ok,
      model_used        = slopes_note,
      lossy_estimate    = coef_l["Estimate"],
      lossy_se          = coef_l["Std. Error"]
    )
  }
  cat("\n")
}

rs_tbl <- bind_rows(all_rows)
write_csv(rs_tbl, "csv/random_slopes_results.csv")
cat("✓ Saved csv/random_slopes_results.csv\n\n")

# ── Summary ───────────────────────────────────────────────────────────────────

cat("═══ Convergence summary ═══\n\n")
cat("Cells where random slopes converged:\n")
rs_tbl %>%
  filter(slopes_converged) %>%
  select(beta, measure, delta_aic_lossy, delta_aic_slopes, lossy_estimate) %>%
  print(n = 50)

cat("\nCells where random slopes failed (intercepts used):\n")
rs_tbl %>%
  filter(!slopes_converged) %>%
  count(measure) %>%
  print()

cat("\nOverall: slopes converged in",
    sum(rs_tbl$slopes_converged, na.rm = TRUE), "of",
    nrow(rs_tbl), "cells\n\n")

# ── Compare: do point estimates shift when slopes are used? ──────────────────

cat("═══ Effect size comparison (slopes vs intercepts) ═══\n\n")
cat("Showing cells where slopes converged — does the lossy_z estimate change?\n\n")

rs_tbl %>%
  filter(slopes_converged) %>%
  select(beta, measure, lossy_estimate, lossy_se, delta_aic_lossy, delta_aic_slopes) %>%
  arrange(measure, beta) %>%
  print(n = 50)

# ── Figure: β decay curves, slopes vs intercepts overlaid ────────────────────

pal <- c(FFD = "#2196F3", GD = "#03A9F4", GPT = "#9C27B0",
         TRT = "#FF5722", reg = "#E91E63")
labs_map <- c(FFD = "FFD", GD = "GD", GPT = "GPT", TRT = "TRT", reg = "Regression")

# Load original intercept-only results for comparison
ri_tbl <- read_csv("csv/model_comparison_final.csv", show_col_types = FALSE) %>%
  mutate(model = "intercepts only")

rs_plot_tbl <- rs_tbl %>%
  filter(slopes_converged) %>%
  select(beta, measure, measure_type, delta_aic = delta_aic_lossy) %>%
  mutate(model = "with random slopes")

combined <- bind_rows(
  ri_tbl %>% select(beta, measure, measure_type, delta_aic, model),
  rs_plot_tbl
) %>%
  filter(!is.na(delta_aic)) %>%
  mutate(
    measure_type = factor(measure_type,
                          levels = c("early", "late"),
                          labels = c("Early measures (FFD, GD)",
                                     "Late measures (GPT, TRT, Regression)")),
    model = factor(model, levels = c("intercepts only", "with random slopes"))
  )

p_rs <- ggplot(combined,
               aes(x = beta, y = delta_aic,
                   color = measure, group = interaction(measure, model),
                   linetype = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~measure_type) +
  scale_color_manual(values = pal, labels = labs_map) +
  scale_linetype_manual(values = c("intercepts only" = "solid",
                                   "with random slopes" = "dotted")) +
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
  labs(
    title    = "Random Slopes vs Intercepts-Only: ΔAIC by β",
    subtitle = "Dotted = random slopes model (only shown where convergence succeeded)",
    x        = "Retention parameter β",
    y        = "ΔAIC (higher = better fit)",
    color    = "Measure",
    linetype = "Model"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

ggsave("fig_random_slopes.png", p_rs, width = 12, height = 5.5, dpi = 150)
ggsave("fig_random_slopes.pdf", p_rs, width = 12, height = 5.5)
cat("✓ Saved fig_random_slopes.png\n")
