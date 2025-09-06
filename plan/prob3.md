# AFT Mainline Extension for Problem 3 — Step-by-Step Plan

## 0) Overview and Notation

Let $i=1,\dots,N$ index pregnant individuals; within-individual blood draws occur at gestational weeks $t_{ij}$ with measured fetal fraction (or Y-chromosome concentration) $Y_{ij}\in[0,1]$. Define the **attainment threshold** at $c=0.04$ (4%). The **first-attainment time**

$$
T_i \;=\; \inf\{\,t:\; Y_i(t)\ge c\,\}
$$

is not directly observed but is **interval-censored** from longitudinal measurements.

* Indicator at visit $j$: $Z_{ij}=\mathbb{1}\{Y_{ij}\ge c\}$.
* Covariates (Problem 3): $X_i=(\text{BMI}_i,\text{age}_i,\text{height}_i,\text{weight}_i,\textstyle\ldots)$.
* Primary grouping variable: $\text{BMI}$. Let group $g$ be a pre-specified BMI stratum.

We extend the **AFT (Accelerated Failure Time) model** used in Problem 2 to incorporate the richer covariate set of Problem 3 and then select **optimal NIPT weeks** per BMI group via risk control.

> **Same as Problem 2 (reused as-is):** interval construction for $T_i$; AFT as the *main* survival model; Turnbull nonparametric check; patient-level splits; time-to-event metrics; threshold-based decision rule; Monte-Carlo (MC) error propagation workflow.
> **New for Problem 3:** expanded covariates and interactions; explicit collinearity control; group-wise decision $t_g^*(\tau)$; 300-run MC sensitivity (Problem 2 used the same mechanism but here we mandate $B=300$ and report per-group decision distributions).

---

## 1) Data Preprocessing (Deterministic)

1. **Inclusion / canonical variables.** Require $\{t_{ij},\,Y_{ij},\,\text{BMI}_i,\,\text{age}_i\}$; optional $\{\text{height}_i,\text{weight}_i\}$. Record unique `patient_id` and all repeated measures.

2. **De-duplication & quality flags.** Keep all valid draws per patient; flag clearly failed assays/low read-depth for sensitivity analysis.

3. **Missingness.**

   * **Same as Problem 2:** impute covariates with a principled method (e.g., MICE or RF-impute), never impute outcomes $Y_{ij}$.
   * **New:** if both height and weight exist with BMI, retain **BMI as primary** and treat height/weight only in sensitivity (to mitigate multicollinearity).

4. **Standardization.** Center/scale continuous covariates for numerical stability:

   $$
   x^{\text{std}}=\frac{x-\bar x}{s_x}.
   $$

---

## 2) Event-Time Interval Construction (Observed Censoring)

> **Same as Problem 2.** For each $i$, order $t_{i1}<\cdots<t_{i n_i}$ and construct $(L_i,R_i]$ for $T_i$:

* **Left-censored** if $Z_{i1}=1$: $(L_i,R_i]=(0,t_{i1}]$.
* **Right-censored** if all $Z_{ij}=0$: $(L_i,R_i]=(t_{i n_i},\infty)$.
* **Interval-censored** if $\exists j<k$ with $Z_{ij}=0,\,Z_{ik}=1$: $(L_i,R_i]=(t_{ij},t_{ik}]$.

---

## 3) AFT Model Specification (Mainline)

### 3.1 Baseline distributions

Use a parametric AFT with either **Weibull** or **log-logistic** baseline:

* **Weibull-AFT:** $S_0(t)=\exp\{-(t/\lambda)^k\}$, $k>0$, $\lambda>0$.
* **Log-logistic-AFT:** $S_0(t)=[1+(t/\lambda)^k]^{-1}$, $k>0$.

### 3.2 Covariate link (time-acceleration)

Let the **time-acceleration factor** be

$$
\psi(X_i) \;=\; \exp\big(\beta_0 + X_i^\top \beta\big),
$$

so that the conditional survival and density are time-scaled:

$$
S(t\mid X_i) \;=\; S_0\!\left(\frac{t}{\psi(X_i)}\right),\qquad
f(t\mid X_i) \;=\; \frac{1}{\psi(X_i)}\, f_0\!\left(\frac{t}{\psi(X_i)}\right).
$$

Equivalently, in log-time form,

$$
\log T_i \;=\; \beta_0 + X_i^\top \beta + \sigma \varepsilon_i,
$$

with $\varepsilon_i$ **extreme-value** (Weibull-AFT) or **logistic** (log-logistic-AFT).

### 3.3 Covariate set and structure

* **Core main effects (Problem 3):** $\text{BMI}$, $\text{age}$.
* **Optional nonlinearity:** restricted cubic spline $B(\text{BMI})$ if AIC improves:

  $$
  \log T_i=\beta_0+ B(\text{BMI}_i)^\top\gamma + \beta_{\text{age}}\text{age}_i + \sigma \varepsilon_i.
  $$
* **Interactions (guarded):** $\text{BMI}\times\text{age}$ considered only if supported by likelihood-ratio test (LRT) and clinically interpretable.

> **Same as Problem 2:** AFT is the primary inferential engine; model comparison via AIC; inference via MLE under interval censoring.
> **New for Problem 3:** introduce $B(\text{BMI})$ and age; height/weight reserved to sensitivity to avoid collinearity with BMI.

---

## 4) Estimation Under Interval Censoring

> **Same as Problem 2.** Estimate $\theta=(\beta_0,\beta,k,\lambda,\sigma)$ by maximizing the interval-censored log-likelihood

$$
\ell(\theta)\;=\;\sum_{i=1}^{N}\log\Big[\,S(L_i\mid X_i)-S(R_i\mid X_i)\Big],
$$

with the conventions $S(0\mid X)\equiv 1$, $S(\infty\mid X)\equiv 0$. Use robust (sandwich) standard errors. Compare Weibull vs log-logistic by AIC; retain the simpler model unless the alternative offers material calibration gains (§6).

---

## 5) Model Diagnostics and Calibration

> **Same as Problem 2 (reused):**

* **Nonparametric benchmark:** Turnbull estimator $\widehat S_{\text{TB}}(t)$ on the same intervals; overlay group-wise curves to check shape.
* **Goodness-of-fit:** compare $S(t\mid X)$ vs $\widehat S_{\text{TB}}(t)$ by integrated absolute error; report time-stratified KS-type discrepancies.
* **Predictive scoring:** interval-censored log-likelihood on held-out patients; integrated Brier score (IBS) at clinically relevant horizons.
* **Data split:** patient-level K-fold or train/valid/test split to avoid leakage across repeated measures.

> **New for Problem 3:** add **collinearity diagnostics** (VIF) and restrict the final covariate set if VIF>5 persists after standardization.

---

## 6) BMI Grouping and Optimal Week Selection

### 6.1 Group-wise survival

For a BMI group $g$ with patient set $\mathcal{I}_g$, define the **group survival** by plug-in averaging:

$$
S_g(t)\;=\;\frac{1}{|\mathcal{I}_g|}\sum_{i\in\mathcal{I}_g} S(t\mid X_i).
$$

(Equivalently, integrate over the empirical distribution of $X$ in group $g$.)

### 6.2 Threshold-based optimal week

Given an **attainment probability target** $\tau\in\{0.90,0.95\}$, define the **earliest safe NIPT week** for group $g$ as

$$
t_g^*(\tau)\;=\;\inf\{\,t:\;1-S_g(t)\;\ge\;\tau\,\}.
$$

> **Same as Problem 2:** use the same $\tau$ standards and the same reporting format for $t_g^*(\tau)$.
> **New for Problem 3:** compute $t_g^*(\tau)$ **per BMI group** (not only overall), and report **between-group contrasts** $\Delta t_{g,h}^*(\tau)=t_g^*(\tau)-t_h^*(\tau)$.

---

## 7) Uncertainty Quantification (Parametric)

> **Same as Problem 2:** obtain uncertainty for $t_g^*(\tau)$ via resampling-based methods.

* **Bootstrap over patients** (case resampling) with full refit to generate $\{t_{g,b}^*(\tau)\}_{b=1}^B$.
* Report point estimate $\bar t_g^*(\tau)$, percentile 95% CI, and **decision robustness** (share of resamples meeting operational cutoffs, if any).

---

## 8) Measurement-Error Sensitivity (300-Run Monte Carlo)

> **Same mechanism as Problem 2, but mandate $B=300$ runs and **report per-group** distributions.** Let assay-level error be additive on the measurement scale,

$$
Y_{ij}^{(b)} \;=\; Y_{ij} \;+\; \varepsilon_{ij}^{(b)},\qquad \varepsilon_{ij}^{(b)}\sim \mathcal N(0,\sigma_Y^2),
$$

with $\sigma_Y$ taken from validated QC or from Problem 2’s empirically estimated assay variability.

For each MC replicate $b=1,\ldots,300$:

1. **Noise injection:** form $Y_{ij}^{(b)}$ and re-threshold $Z_{ij}^{(b)}=\mathbb{1}\{Y_{ij}^{(b)}\ge c\}$.
2. **Interval reconstruction:** rebuild $(L_i^{(b)},R_i^{(b)}]$ as in §2.
3. **Refit AFT:** estimate $\hat\theta^{(b)}$ on $(L_i^{(b)},R_i^{(b)}], X_i$.
4. **Group decisions:** compute $t_{g,b}^*(\tau)$ for each group $g$ and $\tau\in\{0.90,0.95\}$.

Summarize, for each $g,\tau$: mean, SD, 2.5–97.5% percentile CI, and a **robustness label** (e.g., “high/medium/low” stability based on CI width and the share of replicates crossing clinical cutpoints).

---

## 9) Sensitivity Analyses (Targeted, Optional)

* **Covariate set:** add $\{\text{height},\text{weight}\}$ either
  (i) replacing BMI, or
  (ii) as residualized components orthogonal to BMI;
  accept only if AIC improves and variance inflation remains acceptable.
* **Nonlinearity:** retain $B(\text{BMI})$ only when justified by LRT and calibration plots.
* **Baseline choice:** if Weibull and log-logistic have similar AIC, choose the one with better calibration against Turnbull in the region of clinical interest (weeks 10–20, typically).

---

## 10) Assumptions and Justification

1. **Monotone event definition.** The attainment event $Y(t)\ge c$ is well-defined and *first* attainment is clinically meaningful (no reversion below $c$ invalidates the decision policy).
2. **AFT suitability.** Covariates act multiplicatively on time via $\psi(X)=\exp(X^\top\beta)$, yielding interpretable **time ratios** $e^{\beta_k}$.
3. **Independent individuals.** Dependence exists within individual repeated measures but is absorbed by interval construction; individuals $i$ are mutually independent.
4. **Parametric baseline adequacy.** Weibull or log-logistic families approximate the empirical Turnbull sufficiently well in the decision region; mis-specification is checked via §5.
5. **Assay-error model.** Additive Gaussian error on the measurement scale is a reasonable approximation around the 4% decision boundary; the 300-run MC propagates this to the group-level decision.

---

## 11) Implementation Checklist (Concrete Steps)

1. **Construct intervals $(L_i,R_i]$** from $\{t_{ij},Y_{ij}\}$. **(Same as Problem 2)**
2. **Specify AFT(Weibull & log-logistic) with $X=(\text{BMI},\text{age})$**; consider $B(\text{BMI})$ spline. **(New: added covariates/spline)**
3. **Estimate by MLE under interval censoring;** compare baselines by AIC. **(Same as Problem 2)**
4. **Diagnostics vs Turnbull;** patient-level validation; IBS/log-likelihood. **(Same as Problem 2)**
5. **Define BMI groups $g$** exactly as in Problem 2 (use identical cutpoints). Compute $S_g(t)$ and $t_g^*(\tau)$ for $\tau\in\{0.90,0.95\}$. **(Same as Problem 2 for grouping standard; New: per-group reporting)**
6. **Bootstrap CIs** for $t_g^*(\tau)$. **(Same as Problem 2)**
7. **300-run MC error sensitivity,** re-building intervals and re-fitting AFT each run; summarize $t_g^*(\tau)$ distributions per group. **(Same mechanism as Problem 2; mandated $B=300$ here)**
8. **Finalize policy:** report $t_g^*(\tau)$ (point ± 95% CI), robustness labels, and cross-group contrasts $\Delta t_{g,h}^*(\tau)$. **(New: per-group decision with contrasts)**

---

### Deliverables (to mirror Problem 2 format)

* **Tables:** AFT coefficient table (time-ratio interpretation), AIC comparison, VIF diagnostics.
* **Figures:** Turnbull vs model $S_g(t)$ overlays; calibration in 10–20 weeks window; distributions of $t_g^*(\tau)$ from 300-run MC.
* **Decisions:** For each BMI group $g$ and $\tau$, $t_g^*(\tau)$ with 95% CI and robustness label; succinct clinical summary.
