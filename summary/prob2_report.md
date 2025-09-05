# Problem Context

We study the **time-to-event** outcome for male–fetus pregnancies: the earliest gestational week $T$ at which the fetal Y-chromosome concentration (fetal fraction, FF) reaches or exceeds the clinical threshold $4\%$. The task in Problem 2 is to (i) partition maternal BMI into clinically meaningful groups, (ii) recommend for each group an “earliest safe” NIPT blood-draw week $w_g^\star$ that minimizes potential risk (i.e., keeps the sub-threshold risk acceptably low), and (iii) quantify how measurement error in FF affects these recommendations. The notebook implements an interval-censored survival analysis pipeline with model comparison, cross-validation, and Monte-Carlo robustness checks to produce a final policy table. &#x20;

# Data preprocessing

From the feature matrix built per mother, the analysis retains **238** observations with variables $\{\text{BMI}, \text{BMI}_z, L, R\}$, where $(L,R]$ encodes the censoring interval for $T$: left-censored if $L=0$, interval-censored if $0<L<R<\infty$, and right-censored if $R=\infty$. The dataset composition is: 203 left-censored, 22 interval-censored, and 13 right-censored observations; summary statistics (e.g., $\overline{\text{BMI}}=31.88$, $\text{sd}=2.59$) are reported to verify plausibility. No missingness remained.   &#x20;

**Event construction.** For each mother with repeated draws, the event time $T$ is the first week that FF$\ge 4\%$. If FF is already $\ge 4\%$ at the first draw, then $T\le R$ (left-censoring). If FF never reaches $4\%$ by the last draw, then $T>R$ (right-censoring). Otherwise $T\in(L,R]$ (interval-censoring). (Implemented as the “Event Interval Construction” step.)&#x20;

# Methodology

## Survival modeling with AFT (primary)

The notebook fits an **Accelerated Failure Time (AFT)** model for interval-censored data, with **Weibull** chosen as the primary specification and **log-logistic** as an alternative. In AFT form,

$$
\log T \;=\; \beta_0 + \beta_1\,\text{BMI}_z \;+\; \sigma\,\varepsilon,
$$

with $\varepsilon$ Gumbel (Weibull-AFT) or logistic (log-logistic AFT). From the fitted model one obtains the survival function $S(t\mid x)=\Pr(T>t\mid x)$ and hence group-specific curves $S_g(t)=\mathbb{E}_{x\in g}[S(t\mid x)]$. The “earliest safe” week for group $g$ at confidence $\tau\in(0,1)$ is defined as

$$
w_g^\star \;=\; \inf\{t:\;1-S_g(t)\ge \tau\},
$$

i.e., the earliest week by which at least a $\tau$ fraction of the group is expected to have reached FF$\ge 4\%$.&#x20;

**Model selection.** The Weibull AFT exhibited the lowest AIC (**253.54**) versus the log-logistic (**254.20**), and is taken as the primary specification in subsequent inference.&#x20;

## Non-parametric validation

To check parametric assumptions, the notebook fits a **Turnbull** non-parametric estimator for interval-censored survival (timeline 0–24.4 weeks; 27 intervals) and compares it to the AFT model on a clinically meaningful window (12.0–24.7 weeks). Agreement is quantified by MAE, RMSE, and KS statistics. &#x20;

## BMI grouping and policy mapping

Candidate BMI partitions are compared by a **risk-score** that trades off groupwise optimal weeks and complexity (penalty on number of groups), along with between/within-group variance criteria. Three strategies are reported: a CART-derived 6-group split, a 3-tertile split, and a **clinical** 3-group split. The clinical grouping is selected as **best** (lowest risk score among 3-group options), and subsequent policy is computed for these groups.  &#x20;

## Robustness to measurement error

Measurement error is modeled additively on FF as $y^{\text{obs}}=y^{\text{true}}+\epsilon$, $\epsilon\sim\mathcal N(0,\sigma^2)$. A Monte-Carlo procedure perturbs FF, **rebuilds the censoring intervals**, refits the AFT model, and recomputes $w_g^\star$ across simulations, yielding uncertainty bands and stability labels for each group and confidence level.&#x20;

# Results

## Model adequacy and validation

* **Model fit:** AIC favored **Weibull** over log-logistic (253.54 vs 254.20).&#x20;
* **Turnbull agreement:** On 12.0–24.7 weeks, agreement between AFT and Turnbull is excellent: **MAE = 0.0141**, **RMSE = 0.0186**, **KS = 0.0559**. For reference, survival at key weeks (Turnbull vs AFT) are: week 12 (0.195 vs 0.184), week 14 (0.108 vs 0.139), week 16 (0.108 vs 0.100), week 18 (0.066 vs 0.074), week 20 (0.049 vs 0.054). All metrics meet pre-specified adequacy thresholds.  &#x20;
* **Cross-validation:** 5-fold CV on the full AFT pipeline completed successfully in all folds (**5/5**). &#x20;

## Selected BMI grouping and recommended weeks

The final policy uses **three clinical BMI groups** (n=238): Overweight (25–30): $n=61$; Obese I (30–35): $n=147$; Obese II+ ($\ge 35$): $n=30$. This grouping achieved lower risk score than a 6-group CART split and better between-group separation than tertiles; it is declared “Best grouping strategy: clinical.” &#x20;

For each group, the notebook computes $w_g^\star$ at $\tau=0.90$ and $0.95$ (earliest weeks where $\Pr\{T\le t\}\ge\tau$). Point recommendations and Monte-Carlo uncertainty are:

* **Overweight (26.6–30.0)**: $w^\star_{90} = 13.3$ weeks; $w^\star_{95} = 16.7$ weeks. MC: $13.37\pm0.69$ \[11.67, 14.26] and $16.84\pm0.74$ \[15.15, 17.90]. &#x20;
* **Obese I (30.0–34.9)**: $w^\star_{90} = 15.9$ weeks; $w^\star_{95} = 20.2$ weeks. MC: $15.83\pm0.41$ \[14.99, 16.53] and $19.97\pm0.60$ \[18.79, 21.22]. &#x20;
* **Obese II+ (35.1–39.2)**: $w^\star_{90} = 21.1$ weeks; at $\tau=0.95$, the target is **not** reached within the modeled window (“Never”). MC: $20.44\pm1.12$ \[18.63, 22.89] at 90%; $24.00\pm0.66$ \[22.85, 24.96] at 95%.  &#x20;

The final policy table also reports **threshold probabilities at the recommended weeks**—e.g., in Obese I, the group reaches exactly 0.900 at 15.9 weeks and 0.951 at 20.2 weeks—and **Monte-Carlo confidence intervals** that closely match the robustness summary above. Overall CV success rate is **1.00** and Monte-Carlo stability is labeled **Good**. &#x20;

## Robustness to measurement error

Under additive Gaussian noise with $\sigma=0.0020$ (0.20% FF), **50** simulations were run with 100% success; overall stability is **Good**. Group-level stability grades (by coefficient of variation and CI width) indicate that most cells are Good/Excellent, with the only **Poor** label appearing at Obese II+ (90%) due to wider uncertainty; the 95% policy in this group is more stable. &#x20;

# Conclusion

The notebook **successfully** operationalizes Problem 2 by (i) constructing interval-censored event data per mother, (ii) fitting and validating a **Weibull AFT** model selected by AIC over log-logistic, with strong agreement against the **Turnbull** estimator, (iii) selecting a **3-group clinical BMI** partition that optimizes a risk/complexity criterion, and (iv) delivering **group-specific earliest NIPT weeks** at 90% and 95% confidence, together with **Monte-Carlo uncertainty** under realistic measurement error. The recommended testing weeks are **13.3/16.7** (Overweight), **15.9/20.2** (Obese I), and \*\*21.1/\*\*Not-attained-by-95% (Obese II+), with documented stability and cross-validated execution. These results directly meet the Problem 2 requirement to group by BMI, minimize potential risk via earliest safe weeks, and quantify the impact of measurement error on the policy.  &#x20;

**Addendum: Goodness-of-fit and reproducibility.** Formal fit diagnostics (MAE 0.0141; RMSE 0.0186; KS 0.0559 versus Turnbull over 12–24.7 weeks) and 5-fold CV (5/5 successful folds) support the adequacy and reproducibility of the AFT-based policy derivation.  &#x20;

---

**Attachments referenced:** 02\_prob2.pdf (interval construction, AFT fitting, Turnbull validation, BMI grouping evaluation, final policy table, and Monte-Carlo robustness).    &#x20;
