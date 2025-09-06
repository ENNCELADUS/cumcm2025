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

## 设置与假设

我们将“最早达标时间”视为区间删失的事件时间 $T$，其中达标阈值为Y染色体比例 $\ge 4\%$。为建模 $T$ 与孕妇BMI的关系，构建了加速失效时间（AFT）模型。主模型为Weibull-AFT，其对数时间满足

$$
\log T_i=\beta_0+\beta_1\,z(\mathrm{BMI}_i)+\varepsilon_i,\qquad \varepsilon_i\ \text{服从Gumbel分布},
$$

其中 $z(\mathrm{BMI})$ 为标准化BMI。估计表明BMI的系数显著为正（$\hat\beta_1=0.167$，$p=0.0219$），意味着BMI升高会推迟达到4%阈值的时间；AIC=253.54，$\ell=-123.77$。与非参数Turnbull估计相比，AFT在临床区间12–24.7周的拟合优度优秀（MAE 0.0141、RMSE 0.0186、KS 0.0559），支持用该参数模型传播不确定性。

检测误差采用加性高斯误差模型：对每次实验的Y浓度施加独立扰动 $\tilde{Y}=Y+\delta$，$\delta\sim\mathcal N(0,\sigma^2)$，其中 $\sigma=0.002$（绝对比例0.2%）。在每次扰动后重新构造区间删失样本并重估AFT与分组最优时点，由此考察误差传播对“最佳NIPT时点” $t_\alpha(\mathrm{BMI})$ 的影响；这里 $t_\alpha$ 定义为在给定BMI组内满足 $\Pr(T\le t)\ge \alpha$ 的最早孕周（$\alpha\in\{0.90,0.95\}$）。

## Monte Carlo 设计

以上述误差模型进行 $B=300$ 次蒙特卡罗重复（成功率100%），在每次重复中重新估计各BMI组的 $t_{0.90},t_{0.95}$。报告跨重复的均值、标准差（记为“±”）、95%置信区间以及变异系数（CV），并据此给出稳定性评级。

## 主要结果（以周为单位）

**分组与稳健性**（300次仿真；$\sigma=0.002$）：

* **低BMI（T1）**：$t_{0.90}=13.78\pm0.60$（95%CI \[12.58, 14.70]，CV 0.044）；$t_{0.95}=17.40\pm0.73$（\[15.91, 18.56]，CV 0.042）。稳定性“良好”。
* **中BMI（T2）**：$t_{0.90}=15.62\pm0.39$（\[14.92, 16.36]，CV 0.025）；$t_{0.95}=19.75\pm0.59$（\[18.71, 20.91]，CV 0.030）。稳定性“良好”；0.90的95%CI宽度1.44周（评价“好”）。
* **高BMI（T3）**：$t_{0.90}=18.24\pm0.56$（\[16.89, 19.24]，CV 0.031）；$t_{0.95}=23.03\pm0.92$（\[20.98, 24.48]，CV 0.040）。稳定性“良好”；0.95的95%CI宽度3.50周（“中等”）。
* **肥胖I（30–35）**：$t_{0.90}=15.90\pm0.38$（\[15.15, 16.52]，CV 0.024）；$t_{0.95}=20.03\pm0.56$（\[19.09, 21.21]，CV 0.028）。稳定性“良好”；0.90的95%CI宽度1.37周（“好”）。
* **超重（25–30）**：$t_{0.90}=13.55\pm0.67$（\[12.12, 14.85]，CV 0.049）；$t_{0.95}=17.05\pm0.78$（\[15.45, 18.64]，CV 0.046）。稳定性“良好”。
* **肥胖II+（$\ge 35$）**：$t_{0.90}=20.27\pm1.08$（\[18.33, 22.42]，CV 0.053），**稳定性“差”**，其95%CI宽度4.09周（“差”）；$t_{0.95}=24.06\pm0.80$（\[22.18, 25.00]，CV 0.033，“良好”）。总体稳健性评估：良好11项/中等0项/较差1项（唯一例外为肥胖II+在$\alpha=0.90$）。

**仿真设置与执行**：300次、$\sigma=0.002$（0.2%绝对误差），两级目标置信度（90%、95%），全部迭代成功。

## 统计学解释

1. **误差传播的量级**：在 $\sigma=0.2\%$ 的测量误差下，除“肥胖II+、$\alpha=0.90$”外，所有BMI组的$t_\alpha$跨重复的变异系数CV均$\le 0.049$，95%CI宽度多在1.37–3.50周之间，属于“小到中等”不确定度；这表明推荐时点对小幅检测误差总体**鲁棒**。
2. **异常与风险集中区**：仅当BMI$\ge 35$且以“90%达标”作为目标时，CI显著变宽（4.09周），说明微小测量扰动会在该边缘组引发明显的时点不确定性（误早或误迟的风险同时上升）。将目标从90%提升至95%可显著改善稳健性（CV从0.053降至0.033，CI宽度由“差”降为“中等”）。
3. **模型合理性**：AFT与Turnbull在群体生存曲线上的误差仅1–2个百分点量级（MAE 0.0141、RMSE 0.0186），确保我们对$t_\alpha$的不确定性评估主要反映**检测误差**而非模型失配。AFT中BMI效应显著为正，也与临床“高BMI降低胎儿DNA比例、推迟达标”一致。

## 面向决策的结论

* **稳健结论**：当检测误差控制在0.2%量级时，各BMI分组的最优NIPT时点对误差的敏感性总体较低；按$t_{0.95}$制定策略尤为稳健（所有组“良好”）。
* **关键例外与策略调整**：对于**BMI$\ge 35$**，若坚持“90%达标”准则，则时点的不确定性过大（CI宽4.09周）；建议将目标提高至**95%达标**或在90%方案上\*\*追加安全缓冲（$\approx$1–2周）\*\*以降低“未达标抽血”的风险。
* **总体稳健性**：300次仿真下“良好/中等/较差”为**11/0/1**，说明所给分组与时点方案在现实检测波动下具有可实施性，且风险主要集中在极高BMI人群的“较早抽血”情形。

> 注：本节所引数值与稳健性评级均来自300次蒙特卡罗仿真（$\sigma=0.002$），并与AFT-Turnbull对照的高拟合度共同支撑结论的可靠性。


# Conclusion

The notebook **successfully** operationalizes Problem 2 by (i) constructing interval-censored event data per mother, (ii) fitting and validating a **Weibull AFT** model selected by AIC over log-logistic, with strong agreement against the **Turnbull** estimator, (iii) selecting a **3-group clinical BMI** partition that optimizes a risk/complexity criterion, and (iv) delivering **group-specific earliest NIPT weeks** at 90% and 95% confidence, together with **Monte-Carlo uncertainty** under realistic measurement error. The recommended testing weeks are **13.3/16.7** (Overweight), **15.9/20.2** (Obese I), and \*\*21.1/\*\*Not-attained-by-95% (Obese II+), with documented stability and cross-validated execution. These results directly meet the Problem 2 requirement to group by BMI, minimize potential risk via earliest safe weeks, and quantify the impact of measurement error on the policy.  &#x20;

**Addendum: Goodness-of-fit and reproducibility.** Formal fit diagnostics (MAE 0.0141; RMSE 0.0186; KS 0.0559 versus Turnbull over 12–24.7 weeks) and 5-fold CV (5/5 successful folds) support the adequacy and reproducibility of the AFT-based policy derivation.  &#x20;

---

**Attachments referenced:** 02\_prob2.pdf (interval construction, AFT fitting, Turnbull validation, BMI grouping evaluation, final policy table, and Monte-Carlo robustness).    &#x20;
