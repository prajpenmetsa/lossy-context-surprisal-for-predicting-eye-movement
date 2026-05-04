# =============================================================================
# final_models.R
# Full analysis: FFD, GD, GPT, TRT, regression
# Covariates: word_length, log_freq, word_position, dep_length, surprisal
# Lossy surprisal: 50-sample estimates, beta 0.1–0.9
# Exclusions: fixations <100ms and >1200ms (GPT capped at 3000ms)
# =============================================================================

library(glmmTMB)
library(MuMIn)
library(tidyverse)

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

# ── 1. Load and prepare data ──────────────────────────────────────────────────

df_raw <- read_csv("csv/analysis_data.csv", show_col_types = FALSE)
dep    <- read_csv("csv/dep_length.csv",    show_col_types = FALSE) %>%
  select(global_sent_idx, word_idx, dep_length)

cat("Raw rows:", nrow(df_raw), "\n")
cat("GPT column present:", "GPT" %in% names(df_raw), "\n\n")

df <- df_raw %>%
  filter(!is.na(surprisal), !is.na(word_length), !is.na(log_freq)) %>%
  filter(word_position > 0) %>%
  left_join(dep, by = c("global_sent_idx", "word_idx")) %>%
  filter(!is.na(dep_length)) %>%
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
    dep_z  = scale(dep_length)[,1],
    subj    = factor(subj),
    sent_id = factor(sent_id),
    sent_id_num = as.numeric(as.character(sent_id))
  )

# Fixated subset with exclusions
df_fix <- df %>%
  filter(!is.na(FFD)) %>%
  filter(FFD >= 100, FFD <= 1200) %>%
  filter(GD  >= 100, GD  <= 2000) %>%
  filter(GPT >= 100, GPT <= 3000) %>%   # GPT cap higher — includes regressions
  filter(TRT >= 100, TRT <= 3000)

cat("After exclusions:\n")
cat("  df (for reg):", nrow(df), "\n")
cat("  df_fix (for FFD/GD/GPT/TRT):", nrow(df_fix), "\n\n")

# ── 2. Baseline models (all 5 measures) ───────────────────────────────────────

cat("═══ Baseline Models ═══\n\n")

baseline_results <- tibble()

for (dv in c("FFD", "GD", "GPT", "TRT", "reg")) {
  is_bin   <- (dv == "reg")
  data_use <- if (is_bin) df else df_fix
  fam      <- if (is_bin) binomial("logit") else Gamma(link = "log")

  f <- as.formula(paste(dv,
    "~ wlen_z + freq_z + wpos_z + dep_z + surp_z + (1|subj) + (1|sent_id)"))

  m <- tryCatch(glmmTMB(f, data=data_use, family=fam, control=ctrl),
                error=function(e) NULL)

  if (is.null(m)) { cat(dv, "FAILED\n"); next }

  coefs <- summary(m)$coefficients$cond
  r2    <- tryCatch(r.squaredGLMM(m)[1], error=function(e) NA)

  cat(sprintf("%-4s | AIC=%.1f | R²=%.4f\n", dv, AIC(m), r2))
  for (pred in c("surp_z","dep_z","wlen_z","freq_z","wpos_z")) {
    if (pred %in% rownames(coefs)) {
      cat(sprintf("  %-8s β=%7.4f  p=%.4f\n",
                  pred, coefs[pred,"Estimate"], coefs[pred,"Pr(>|z|)"]))
    }
  }
  cat("\n")

  baseline_results <- bind_rows(baseline_results, tibble(
    measure = dv,
    aic     = AIC(m),
    r2_marg = r2,
    surp_b  = coefs["surp_z","Estimate"],
    surp_p  = coefs["surp_z","Pr(>|z|)"],
    dep_b   = coefs["dep_z","Estimate"],
    dep_p   = coefs["dep_z","Pr(>|z|)"]
  ))
}

write_csv(baseline_results, "csv/baseline_results_final.csv")
saveRDS(list(df=df, df_fix=df_fix), "csv/final_dfs.rds")

# ── 3. Lossy models (50-sample) ───────────────────────────────────────────────

# Find the 50-sample lossy surprisal file
lossy_file <- if (file.exists("csv/lossy_surprisal_50.csv")) {
  "csv/lossy_surprisal_50.csv"
} else if (file.exists("lossy_surprisal_50.csv")) {
  "lossy_surprisal_50.csv"
} else {
  cat("[WARN] lossy_surprisal_50.csv not found, falling back to lossy_surprisal.csv\n")
  "csv/lossy_surprisal.csv"
}

cat("Loading lossy surprisal from:", lossy_file, "\n")
lossy_raw <- read_csv(lossy_file, show_col_types = FALSE)
betas     <- sort(unique(lossy_raw$beta))
betas     <- betas[betas < 1.0]
cat("Betas:", betas, "\n\n")

all_rows <- list()

for (b in betas) {
  cat("════ beta =", b, "════\n")

  lossy_b <- lossy_raw %>%
    filter(beta == b) %>%
    select(global_sent_idx, word_idx, lossy_surprisal)

  df_b <- df %>%
    left_join(lossy_b, by=c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  df_fix_b <- df_fix %>%
    left_join(lossy_b, by=c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
    filter(!is.na(lossy_surprisal)) %>%
    mutate(lossy_z = scale(lossy_surprisal)[,1])

  for (dv in c("FFD", "GD", "GPT", "TRT", "reg")) {
    is_bin   <- (dv == "reg")
    data_use <- if (is_bin) df_b else df_fix_b
    fam      <- if (is_bin) binomial("logit") else Gamma(link="log")

    f_base  <- as.formula(paste(dv,
      "~ wlen_z+freq_z+wpos_z+dep_z+surp_z+(1|subj)+(1|sent_id)"))
    f_lossy <- as.formula(paste(dv,
      "~ wlen_z+freq_z+wpos_z+dep_z+surp_z+lossy_z+(1|subj)+(1|sent_id)"))

    mb <- tryCatch(glmmTMB(f_base,  data=data_use, family=fam, control=ctrl), error=function(e) NULL)
    ml <- tryCatch(glmmTMB(f_lossy, data=data_use, family=fam, control=ctrl), error=function(e) NULL)

    if (is.null(mb) || is.null(ml)) { cat("  [FAIL]", dv, "\n"); next }

    delta  <- AIC(mb) - AIC(ml)
    lrt    <- tryCatch(anova(mb, ml), error=function(e) NULL)
    p_val  <- if (!is.null(lrt)) lrt$`Pr(>Chisq)`[2] else NA
    r2b    <- tryCatch(r.squaredGLMM(mb)[1], error=function(e) NA)
    r2l    <- tryCatch(r.squaredGLMM(ml)[1], error=function(e) NA)
    coef_l <- summary(ml)$coefficients$cond["lossy_z",]

    cat(sprintf("  %-4s | ΔAIC=%6.2f | p=%.4f | β=%7.4f (se=%.4f)\n",
                dv, delta, ifelse(is.na(p_val),NA,p_val),
                coef_l["Estimate"], coef_l["Std. Error"]))

    all_rows[[length(all_rows)+1]] <- tibble(
      beta          = b,
      measure       = dv,
      measure_type  = case_when(
        dv %in% c("FFD","GD")      ~ "early",
        dv %in% c("GPT","TRT","reg") ~ "late"
      ),
      delta_aic     = delta,
      p_lrt         = p_val,
      delta_r2      = r2l - r2b,
      lossy_estimate= coef_l["Estimate"],
      lossy_se      = coef_l["Std. Error"]
    )
  }
  cat("\n")
}

tbl_final <- bind_rows(all_rows)
write_csv(tbl_final, "csv/model_comparison_final.csv")
cat("✓ Saved csv/model_comparison_final.csv\n\n")

# ── 4. Figures ────────────────────────────────────────────────────────────────

# Color palette — 5 measures
pal <- c(FFD="#2196F3", GD="#03A9F4", GPT="#9C27B0",
         TRT="#FF5722", reg="#E91E63")
labs_map <- c(FFD="FFD (First Fixation)", GD="GD (Gaze Duration)",
              GPT="GPT (Go-Past Time)",   TRT="TRT (Total Reading Time)",
              reg="Regression")

# Fig 1: Beta decay curves
p1 <- tbl_final %>%
  filter(!is.na(delta_aic)) %>%
  mutate(measure_type = factor(measure_type,
         levels=c("early","late"),
         labels=c("Early measures (FFD, GD)",
                  "Late measures (GPT, TRT, Regression)"))) %>%
  ggplot(aes(x=beta, y=delta_aic, color=measure, group=measure)) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  geom_hline(yintercept=0, linetype="dashed", color="gray50") +
  facet_wrap(~measure_type) +
  scale_color_manual(values=pal, labels=labs_map) +
  scale_x_continuous(breaks=c(0.1,0.3,0.5,0.7,0.9)) +
  labs(
    title    = "Lossy-Context Surprisal: Incremental Fit by Memory Decay (β)",
    subtitle = "50-sample estimates | dep_length controlled | fixation outliers excluded",
    x = "Retention parameter β  (lower = more forgetting)",
    y = "ΔAIC (higher = better fit)",
    color = "Measure"
  ) +
  theme_bw(base_size=13) +
  theme(legend.position="bottom", strip.text=element_text(face="bold"))

ggsave("fig1_final.png", p1, width=12, height=5.5, dpi=150)
ggsave("fig1_final.pdf", p1, width=12, height=5.5)

# Fig 2: Coefficient plot
p2 <- tbl_final %>%
  mutate(measure_type = factor(measure_type,
         levels=c("early","late"),
         labels=c("Early measures (FFD, GD)",
                  "Late measures (GPT, TRT, Regression)"))) %>%
  ggplot(aes(x=beta, y=lossy_estimate,
             ymin=lossy_estimate-1.96*lossy_se,
             ymax=lossy_estimate+1.96*lossy_se,
             color=measure, fill=measure)) +
  geom_ribbon(alpha=0.15, color=NA) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  geom_hline(yintercept=0, linetype="dashed") +
  facet_wrap(~measure_type, scales="free_y") +
  scale_color_manual(values=pal, labels=labs_map) +
  scale_fill_manual(values=pal,  labels=labs_map) +
  scale_x_continuous(breaks=c(0.1,0.3,0.5,0.7,0.9)) +
  labs(
    title   = "Effect of Lossy-Context Surprisal on Eye-Tracking Measures (Final)",
    subtitle = "Standardised coefficient ± 95% CI",
    x = "Retention parameter β", y = "Standardised coefficient",
    color = "Measure", fill = "Measure"
  ) +
  theme_bw(base_size=13) +
  theme(legend.position="bottom", strip.text=element_text(face="bold"))

ggsave("fig2_final.png", p2, width=12, height=5.5, dpi=150)
ggsave("fig2_final.pdf", p2, width=12, height=5.5)

cat("✓ Saved fig1_final.png and fig2_final.png\n")

# ── 5. Clean summary table ────────────────────────────────────────────────────

cat("\n═══ Final Results Summary ═══\n")
tbl_final %>%
  filter(!is.na(delta_aic)) %>%
  mutate(sig = case_when(
    p_lrt < .001 ~ "***",
    p_lrt < .01  ~ "**",
    p_lrt < .05  ~ "*",
    TRUE         ~ "n.s."
  )) %>%
  select(beta, measure, measure_type, delta_aic, p_lrt, sig, lossy_estimate) %>%
  arrange(measure, beta) %>%
  print(n=50)
