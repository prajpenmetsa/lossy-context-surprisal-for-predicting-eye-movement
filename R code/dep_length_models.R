# =============================================================================
# dep_length_models.R
# Add dependency length as a predictor to baseline and lossy models.
# Run after compute_dep_length.py produces dep_length.csv.
# =============================================================================

library(glmmTMB)
library(MuMIn)
library(tidyverse)

# ── 1. Load clean data and dep_length ────────────────────────────────────────

df_fix <- readRDS("analysis_df_fix_clean.rds")
df     <- readRDS("analysis_df_clean.rds")

dep_raw <- read_csv("dep_length.csv", show_col_types = FALSE) %>%
  select(global_sent_idx, word_idx, dep_length)

cat("Dep length rows:", nrow(dep_raw), "\n")
cat("Dep length summary:\n")
print(summary(dep_raw$dep_length))

# Merge dep length into both dataframes
df_fix <- df_fix %>%
  mutate(sent_id_num = as.numeric(as.character(sent_id))) %>%
  left_join(dep_raw, by = c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
  filter(!is.na(dep_length)) %>%
  mutate(dep_z = scale(dep_length)[,1])

df <- df %>%
  mutate(sent_id_num = as.numeric(as.character(sent_id))) %>%
  left_join(dep_raw, by = c("sent_id_num"="global_sent_idx","word_idx"="word_idx")) %>%
  filter(!is.na(dep_length)) %>%
  mutate(dep_z = scale(dep_length)[,1])

cat("\nRows after dep merge:\n")
cat("  df_fix:", nrow(df_fix), "\n")
cat("  df:", nrow(df), "\n\n")

ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))

# ── 2. Baseline + dep_length models ──────────────────────────────────────────
# Formula: outcome ~ wlen_z + freq_z + wpos_z + surp_z + dep_z + (1|subj) + (1|sent_id)

cat("═══ Baseline + Dependency Length ═══\n\n")

fit_dep_model <- function(outcome, data, family, label) {
  cat("─── Fitting", label, "───\n")
  f_base <- as.formula(paste(outcome,
    "~ wlen_z + freq_z + wpos_z + surp_z + (1|subj) + (1|sent_id)"))
  f_dep  <- as.formula(paste(outcome,
    "~ wlen_z + freq_z + wpos_z + surp_z + dep_z + (1|subj) + (1|sent_id)"))

  mb <- tryCatch(glmmTMB(f_base, data=data, family=family, control=ctrl), error=function(e) NULL)
  md <- tryCatch(glmmTMB(f_dep,  data=data, family=family, control=ctrl), error=function(e) NULL)

  if (is.null(mb) || is.null(md)) { cat("  FAILED\n\n"); return(NULL) }

  delta <- AIC(mb) - AIC(md)
  lrt   <- tryCatch(anova(mb, md), error=function(e) NULL)
  p_val <- if (!is.null(lrt)) lrt$`Pr(>Chisq)`[2] else NA
  coefs <- summary(md)$coefficients$cond

  cat(sprintf("  ΔAIC (adding dep_z): %.2f  p=%.4f\n", delta, p_val))
  cat(sprintf("  dep_z:  β=%.4f  se=%.4f  p=%.4f\n",
              coefs["dep_z","Estimate"],
              coefs["dep_z","Std. Error"],
              coefs["dep_z","Pr(>|z|)"]))
  cat(sprintf("  surp_z: β=%.4f  se=%.4f  p=%.4f\n",
              coefs["surp_z","Estimate"],
              coefs["surp_z","Std. Error"],
              coefs["surp_z","Pr(>|z|)"]))
  cat("\n")
  return(md)
}

m_FFD_dep <- fit_dep_model("FFD", df_fix, Gamma(link="log"), "FFD + dep_length")
m_GD_dep  <- fit_dep_model("GD",  df_fix, Gamma(link="log"), "GD + dep_length")
m_TRT_dep <- fit_dep_model("TRT", df_fix, Gamma(link="log"), "TRT + dep_length")

cat("─── Fitting Regression + dep_length ───\n")
m_reg_base <- tryCatch(
  glmmTMB(reg ~ wlen_z+freq_z+wpos_z+surp_z+(1|subj)+(1|sent_id),
          data=df, family=binomial("logit"), control=ctrl), error=function(e) NULL)
m_reg_dep  <- tryCatch(
  glmmTMB(reg ~ wlen_z+freq_z+wpos_z+surp_z+dep_z+(1|subj)+(1|sent_id),
          data=df, family=binomial("logit"), control=ctrl), error=function(e) NULL)

if (!is.null(m_reg_base) && !is.null(m_reg_dep)) {
  delta <- AIC(m_reg_base) - AIC(m_reg_dep)
  lrt   <- tryCatch(anova(m_reg_base, m_reg_dep), error=function(e) NULL)
  p_val <- if (!is.null(lrt)) lrt$`Pr(>Chisq)`[2] else NA
  coefs <- summary(m_reg_dep)$coefficients$cond
  cat(sprintf("  ΔAIC (adding dep_z): %.2f  p=%.4f\n", delta, p_val))
  cat(sprintf("  dep_z:  β=%.4f  se=%.4f  p=%.4f\n",
              coefs["dep_z","Estimate"], coefs["dep_z","Std. Error"], coefs["dep_z","Pr(>|z|)"]))
  cat(sprintf("  surp_z: β=%.4f  se=%.4f  p=%.4f\n",
              coefs["surp_z","Estimate"], coefs["surp_z","Std. Error"], coefs["surp_z","Pr(>|z|)"]))
}

# ── 3. Lossy models with dep_length ──────────────────────────────────────────

cat("\n═══ Lossy + Dependency Length ═══\n\n")

lossy_raw <- read_csv("lossy_surprisal.csv", show_col_types = FALSE)
betas     <- sort(unique(lossy_raw$beta))
betas     <- betas[betas < 1.0]

rows <- list()

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

  for (dv in c("FFD","GD","TRT","reg")) {
    is_bin   <- (dv == "reg")
    data_use <- if (is_bin) df_b else df_fix_b
    fam      <- if (is_bin) binomial("logit") else Gamma(link="log")

    # Full model: baseline covariates + dep_z + surp_z + lossy_z
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
    coef_l <- summary(ml)$coefficients$cond["lossy_z",]

    cat(sprintf("  %-4s | ΔAIC=%6.2f | p=%.4f | β=%.4f (se=%.4f)\n",
                dv, delta, ifelse(is.na(p_val),NA,p_val),
                coef_l["Estimate"], coef_l["Std. Error"]))

    rows[[length(rows)+1]] <- tibble(
      beta=b, measure=dv,
      measure_type=ifelse(dv %in% c("FFD","GD"),"early","late"),
      delta_aic=delta, p_lrt=p_val,
      lossy_estimate=coef_l["Estimate"],
      lossy_se=coef_l["Std. Error"]
    )
  }
  cat("\n")
}

tbl_dep <- bind_rows(rows)
write_csv(tbl_dep, "model_comparison_dep.csv")

# ── 4. Figure ─────────────────────────────────────────────────────────────────

p1 <- tbl_dep %>%
  filter(!is.na(delta_aic)) %>%
  ggplot(aes(x=beta, y=delta_aic, color=measure, group=measure)) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  geom_hline(yintercept=0, linetype="dashed", color="gray50") +
  facet_wrap(~measure_type, labeller=labeller(
    measure_type=c(early="Early measures (FFD, GD)",
                   late ="Late measures (TRT, Regression)"))) +
  scale_color_manual(values=c(FFD="#2196F3",GD="#03A9F4",TRT="#FF5722",reg="#E91E63"),
                     labels=c(FFD="FFD",GD="GD",TRT="TRT",reg="Regression")) +
  scale_x_continuous(breaks=c(0.1,0.3,0.5,0.7,0.9)) +
  labs(
    title    = "Lossy-Context Surprisal + Dependency Length: Fit by β",
    subtitle = "All models include dependency length as covariate",
    x        = "Retention parameter β  (lower = more forgetting)",
    y        = "ΔAIC (higher = better fit)",
    color    = "Measure"
  ) +
  theme_bw(base_size=13) +
  theme(legend.position="bottom", strip.text=element_text(face="bold"))

ggsave("fig1_dep.png", p1, width=11, height=5.5, dpi=150)
cat("✓ Saved fig1_dep.png\n")
cat("✓ Saved model_comparison_dep.csv\n")
