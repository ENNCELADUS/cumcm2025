# Problem Context

We study the quantitative dependence of fetal **Y-chromosome concentration** $Y\in(0,1)$ on **gestational age** (weeks) and **maternal BMI**. The objective is to (i) establish an interpretable regression relationship, (ii) assess linearity/non-linearity in weeks, (iii) account for repeated measurements per patient, and (iv) provide a clinically meaningful model for the probability that $Y$ exceeds the 4% reliability threshold.

# Methodology

## 1. Correlation screening

For each observation $i$ (or subject $i$, visit $j$), Pearson and Spearman correlations were computed:

$$
r_{\text{Pearson}}=\frac{\sum (x-\bar x)(y-\bar y)}{\sqrt{\sum(x-\bar x)^2\sum(y-\bar y)^2}},\qquad
r_{\text{Spearman}}=\text{corr}(\text{rank}(x),\text{rank}(y)).
$$

This provides initial evidence of association for **weeks** and **BMI** with $Y$.

## 2. Baseline linear model and robust inference

We first fit an OLS model

$$
Y_{ij}=\beta_0+\beta_1\,\text{weeks}_{ij}+\beta_2\,\text{BMI}_{ij}+\varepsilon_{ij},\qquad
\mathbb E[\varepsilon_{ij}\mid X]=0.
$$

Global significance is assessed by the model $F$-test; coefficient significance by $t$-tests. Because residual diagnostics indicated heteroskedasticity, inference was repeated with heteroskedasticity-consistent (HC) standard errors (HC3). (Robust-SE rationale is standard in linear modeling.)

## 3. Natural spline model for non-linearity in weeks

To allow a flexible, smooth, and boundary-stable effect of gestational age, we expand weeks using **natural cubic spline** basis functions. With $K$ basis functions $\{B_k(\cdot)\}_{k=1}^{K}$ (here **df = 3**),

$$
f(\text{weeks}_{ij})=\sum_{k=1}^{K}\theta_k\, B_k(\text{weeks}_{ij}),
$$

and the model becomes

$$
Y_{ij}= \theta^\top B(\text{weeks}_{ij}) + \beta_{\text{BMI}}\,\text{BMI}_{ij} + \varepsilon_{ij}.
$$

Natural splines are cubic within the interior and constrained to be linear beyond boundary knots, providing stable extrapolation and reduced variance at the edges. ([patsy.readthedocs.io][1], [Bookdown][2])

## 4. Mixed-effects model for repeated measures (clustering)

Because many patients have multiple visits, we use a **linear mixed-effects model** with a patient-level random intercept:

$$
Y_{ij}= \theta^\top B(\text{weeks}_{ij}) + \beta_{\text{BMI}}\,\text{BMI}_{ij}
+ u_i + \varepsilon_{ij},\qquad
u_i\sim\mathcal N(0,\sigma_u^2),\ \varepsilon_{ij}\sim\mathcal N(0,\sigma^2).
$$

Estimation uses REML/ML; **likelihood-ratio tests (LRTs)** compare nested models (e.g., with/without spline or random components). For models differing **only** in random effects, REML-based LRTs are acceptable, whereas comparisons that change fixed effects should be done under ML. ([PSU | Portland State University][3], [Stata][4], [UW Computer Sciences][5])

We report **marginal $R^2$** (variance explained by fixed effects) and **conditional $R^2$** (variance explained by both fixed and random effects) following Nakagawa & Schielzeth:

$$
R^2_{\text{marg}}=\frac{\sigma_f^2}{\sigma_f^2+\sigma_u^2+\sigma^2},\qquad
R^2_{\text{cond}}=\frac{\sigma_f^2+\sigma_u^2}{\sigma_f^2+\sigma_u^2+\sigma^2}.
$$

These quantify fixed-effect versus total explained variance in LMMs. ([British Ecological Society Journals][6], [easystats][7])

The **intraclass correlation** (ICC) is $\mathrm{ICC}=\sigma_u^2/(\sigma_u^2+\sigma^2)$, measuring within-patient clustering.

## 最终模型公式解释

## 1. 模型公式（统计写法）

$$
Y_{ij} \;=\; f(\text{weeks}_{ij}) \;+\; \beta_{\text{BMI}} \cdot \text{BMI}_{ij} \;+\; u_i \;+\; \varepsilon_{ij}
$$

其中：

* $Y_{ij}$：第 $i$ 个孕妇（patient）在第 $j$ 次检测时的 **Y 染色体浓度**。
* $f(\text{weeks}_{ij})$：孕周（weeks）的 **非线性函数**，由 **自然样条 (natural splines, bs)** 表示。
* $\beta_{\text{BMI}}$：母体 BMI 的回归系数。
* $u_i$：第 $i$ 个孕妇的 **随机截距 (random intercept)**，即不同孕妇之间的平均水平差异。
* $\varepsilon_{ij}$：残差（个体内测量误差）。

---

## 2. 公式各部分解释

### (a) `bs(weeks, df=3)`

* `bs` = **basis splines**，即“样条基函数”。
* `df=3` 意味着选择 3 个自由度（degrees of freedom）来近似孕周的曲线关系。
* 具体做法：把孕周 `weeks` 转换成 **3 个新的变量**（基函数），记为 $B_1(weeks), B_2(weeks), B_3(weeks)$，每个都是由分段多项式拼接而成的光滑曲线。
* 模型实际上拟合的是：

  $$
  f(\text{weeks}) \;=\; \beta_1 B_1(\text{weeks}) \;+\; \beta_2 B_2(\text{weeks}) \;+\; \beta_3 B_3(\text{weeks})
  $$
* 好处：比直线更灵活，可以捕捉“先平缓、后快速上升”的非线性趋势。

---

### (b) `BMI`

* 母体 BMI 是一个普通的线性自变量：

  $$
  \beta_{\text{BMI}} \cdot \text{BMI}
  $$
* 解释：BMI 每增加 1 单位，Y 浓度的期望值大约减少 $\beta_{\text{BMI}}$。

---

### (c) `(1 | patient_id)`

* 这是 **random intercept term**（随机截距）。
* 数学形式：

  $$
  u_i \sim \mathcal{N}(0, \sigma^2_u)
  $$

  每个孕妇 $i$ 都有一个属于自己的随机效应 $u_i$，它代表该孕妇的平均 Y 浓度偏离总体均值的程度。
* 为什么要加这个？因为很多孕妇有多次检测，数据存在 **重复测量（clustering）**。如果不加，模型会假设每次观测都独立，从而低估标准误。
* 这样建模后，相同孕妇的多次观测会共享一个“个体基线”，提高模型对相关性的解释力。

---

## 3. 整体公式的意思

综合起来，模型就是：

$$
Y_{ij} \;=\; \underbrace{\big[\beta_1 B_1(\text{weeks}_{ij}) + \beta_2 B_2(\text{weeks}_{ij}) + \beta_3 B_3(\text{weeks}_{ij})\big]}_{\text{孕周的非线性曲线效应}}
\;+\; \underbrace{\beta_{\text{BMI}} \cdot \text{BMI}_{ij}}_{\text{BMI 的线性效应}}
\;+\; \underbrace{u_i}_{\text{孕妇特异性随机截距}}
\;+\; \underbrace{\varepsilon_{ij}}_{\text{残差误差}}
$$

* **孕周 (weeks)**：通过 spline 曲线拟合 → 非线性增长。
* **BMI**：线性负效应。
* **随机截距**：每个孕妇有自己的 baseline 水平。
* **残差**：个体内随机波动。

---

✅ **一句话总结**：

* `bs(weeks, df=3)` = 用三条样条基函数去刻画孕周与 Y 浓度的非线性关系；
* `BMI` = 一个线性自变量；
* `(1|patient_id)` = 给每个孕妇一个随机截距，处理重复测量和个体差异。

这样写的公式既能抓住 **曲线效应**，又能控制 **个体聚类**。

## 5. Binary clinical model (≥4% threshold)

For $Z_{ij}=\mathbf 1\{Y_{ij}\ge 0.04\}$, we fit a logistic model

$$
\operatorname{logit}\Pr(Z_{ij}=1)=g(\text{weeks}_{ij})+\gamma_{\text{BMI}}\,\text{BMI}_{ij},
$$

with $g(\cdot)$ a spline expansion as above. When mixed logistic (GLMM) software is unavailable, we report standard logistic estimates with cluster-robust inference as a pragmatic approximation; GLMMs are the principled target for longitudinal binary outcomes (general LMM/GLMM theory). ([Bookdown][8])

# Results

* **Correlations (N=555):** weeks shows a weak positive association with $Y$ (Pearson $r=0.1844$, $p<0.0001$; Spearman $r=0.1145$, $p=0.0069$); BMI shows a weak negative association (Pearson $r=-0.1378$, $p=0.0011$; Spearman $r=-0.1498$, $p=0.0004$).
* **Baseline OLS:** $Y\sim\text{weeks}+\text{BMI}$ is globally significant (F-statistic=18.1995, $p<0.0001$); weeks coefficient positive ($p\approx 10^{-6}$); BMI coefficient negative ($p\approx 6\times 10^{-5}$); $R^2=0.0619$ (6.19%), Adjusted $R^2=0.0585$. Residual diagnostics show heteroskedasticity and tail departures from normality.
* **Spline OLS (df=3):** Adding a natural spline for weeks yields $R^2\approx 0.094$ and AIC improvement (\~15.6), with an LRT versus baseline indicating **significant non-linearity** (p < 10⁻³).
* **Mixed-effects + spline (final):**

  $$
  Y_{ij}= \theta^\top B(\text{weeks}_{ij}) + \beta_{\text{BMI}}\,\text{BMI}_{ij} + u_i + \varepsilon_{ij}.
  $$

  The weeks spline terms are collectively significant; BMI remains a significant negative predictor ($p=0.038$). Estimated variances give **ICC = 0.7109**, confirming strong patient-level clustering. The $R^2$ decomposition (corrected using Nakagawa & Schielzeth method) indicates **marginal $R^2=0.126988$** (12.70%, fixed effects) and **conditional $R^2=0.747584$** (74.76%, fixed+random), i.e., substantial between-patient heterogeneity captured by the random intercept.
* **Clinical classification (≥4%):** Logistic modeling with splines for weeks shows high predicted probabilities of surpassing the threshold across the observed weeks; higher BMI lowers the probability, especially in early gestation.

# 显著性检验

**模型背景.** 以胎儿 Y 染色体浓度为因变量（$Y$），核心自变量为孕周（weeks）与母体 BMI。为刻画孕周—$Y$ 的潜在非线性并控制个体/批次差异，最终采用自然样条与随机截距的线性混合效应模型：

$$
Y = \beta_0 + \mathbf{B}_s(\text{weeks};\,df{=}3)^\top\boldsymbol{\beta}_w + \beta_{\mathrm{BMI}}\cdot \mathrm{BMI} + u_0 + \varepsilon,
$$

其中 $u_0\sim\mathcal{N}(0,\sigma_u^2)$。显著性检验关注：各固定效应（孕周样条项与 BMI）是否显著、随机截距是否必要，以及含样条/混合结构的模型是否优于相应简化模型。

**显著性检验方法.** 对系数层面，使用 Wald（z）检验评估单个回归系数是否显著偏离 0；对孕周样条项，采用联合显著性检验（似然比检验，LRT）；对嵌套结构（是否纳入随机截距、线性 vs 样条），在最大似然框架下使用 LRT；并以信息准则（AIC/BIC）比较相对优度。若 $p<0.05$ 视为显著；必要处报告更严格阈值（如 $p<0.001$）。

**验证过程与异常结果处理.** 我们在多种规格（线性/样条、是否纳入随机截距、不同样条自由度）下重复拟合，并结合残差诊断与影响度分析。对出现边界估计、系数不稳定或由高杠杆点主导的结果不纳入结论，仅保留在多规格中一致、诊断良好且信息准则/LRT 同向支持的**可靠检验结果**，以保证推断的稳健性与科学性。

**可靠结果（含数值）.**
（1）**固定效应显著性（最终模型）**：截距显著（0.104076，SE 0.020809，z = 5.002，$p=5.69×10^{-7}$）；孕周样条基函数中 `bs(weeks, df=3)[0]` 显著（0.027322，SE 0.010545，z = 2.591，$p=0.0096$），`bs(weeks, df=3)[1]` 不显著（$-0.003377$，SE 0.007217，z = $-0.468$，$p=0.640$），`bs(weeks, df=3)[2]` 极显著（0.058025，SE 0.005820，z = 9.971，$p=2.05×10^{-23}$）；BMI 呈显著负效应（$-0.001332$，SE 0.000642，z = $-2.073$，$p=0.038$）。
（2）**孕周非线性贡献（联合检验）**：线性 vs 样条之 LRT 给出 LR = 13.4656，df = 2，$p=0.001191$，表明引入样条后模型显著改进；信息准则亦支持该结论（AIC 改进 9.47，BIC 改进 0.83）。
（3）**随机截距的必要性（层级结构）**：在 ML 估计下，混合模型与相应固定效应模型比较的 LRT 统计量为 186.8558，df = 1，$p<10^{-6}$，显示随机截距高度显著；方差分量估计为 $\sigma_u^2=0.000743$、$\sigma_e^2=0.000302$，组内相关系数 ICC = 0.7109，提示71.1%的变异来自患者间层级。
（4）**模型优度与选择**：候选模型信息准则为——线性混合效应 AIC = $-2416.47$、BIC = $-2399.20$、logLik = 1212.24；样条混合效应（最终）AIC = $-2425.94$、BIC = $-2400.02$、logLik = 1218.97；固定效应（线性）AIC = $-2241.08$、logLik = 1125.54。综合 AIC/BIC 与 LRT，样条混合效应模型最优。
（5）**整体拟合与解释度**：相对于零模型的全局 LRT 为 104.0386，df = 4，$p<0.001$，表明整体模型高度显著。基于 Nakagawa–Schielzeth 方法（已修正），边际 $R^2=0.126988$（固定效应解释 12.70% 变异），条件 $R^2=0.747584$（固定效应与随机效应合计解释 74.76% 变异），与方差分解（固定效应方差 0.000152、随机效应方差 0.000743、残差方差 0.000302）一致。

**结论性表述.** 以上结果一致表明：孕周对 Y 染色体浓度的影响在联合检验中高度显著（LR = 13.4656，df = 2，$p=0.001191$），BMI 的负向主效应亦达到统计显著（$p=0.038$）；同时，纳入随机截距与孕周样条项的模型在 LRT 与信息准则下显著优于基准与线性规格。该显著性证据为后续问题二与问题三的分组建模与阈值优化提供了可复核的数理依据与稳健出发点。

---

### 插入/图注示例（便于直接粘贴）

* 图注（partial effect）：“图：在最终混合样条模型下，孕周（weeks）的部分效应曲线（自然样条，df=3），不同曲线对应不同 BMI 水平；阴影为 95% 置信区间。”
* 表注（coefficients）：“表：最终模型固定效应系数、标准误、z 值、p 值与 95% 置信区间（方差分量已另表报告）；固定效应显著性检验采用 Wald-type z 検验并辅以 cluster-robust 与 bootstrap 稳健性检验，详见附录。”


# 软件环境与可复现性（Software & Reproducibility）

**分析环境**
* **Python**: 3.11.x
* **主要库**: 
  - `statsmodels` 0.14+ (混合效应建模)
  - `pandas` 1.5+ (数据处理)
  - `numpy` 1.24+ (数值计算)
  - `scipy` 1.10+ (统计检验)
  - `matplotlib` 3.7+ / `seaborn` 0.12+ (可视化)
  - `patsy` 0.5+ (样条基函数)
  - `scikit-learn` 1.3+ (ROC分析)
* **随机种子**: 42 (用于bootstrap重抽样和交叉验证)
* **计算环境**: Linux Ubuntu 22.04, conda 环境管理

**统计方法学参考标准**
* 混合效应模型: Pinheiro & Bates (2000), Verbeke & Molenberghs (2000)
* R²计算: Nakagawa & Schielzeth (2013) 
* 样条回归: Hastie et al. (2009)
* 稳健推断: MacKinnon & White (1985) HC3标准误
* 信息准则: Burnham & Anderson (2002)

# Conclusion

We constructed a statistically principled pipeline for Problem 1 that (i) screens associations, (ii) establishes a baseline linear relation, (iii) **demonstrates and models non-linearity** in gestational age via **natural cubic splines** (df=3), and (iv) **accounts for repeated measures** through a **random-intercept mixed-effects** specification. The fitted mixed-effects spline model confirms that **gestational weeks** and **BMI** are significant predictors of fetal Y-chromosome concentration, with a nonlinear increasing weeks effect and a negative BMI effect; strong clustering (ICC = 0.7109) justifies the mixed framework. This final model is selected on theoretical grounds (appropriate handling of non-linearity and clustering) and empirical evidence (likelihood-ratio improvements and $R^2$ diagnostics), furnishing a rigorous basis for inference and for clinical decision support regarding reliability thresholds. ([GitHub][9], [patsy.readthedocs.io][1], [Bookdown][2], [British Ecological Society Journals][6], [Stata][4])

**Notes on notation and operators:** In the model formula

$$
\texttt{Y\_concentration ~ bs(weeks, df=3) + BMI + (1|patient\_id)},
$$

$\texttt{bs}$ denotes a **B-spline/natural spline basis expansion** for weeks, producing the vector $B(\text{weeks})$; $\texttt{(1|patient\_id)}$ denotes a **patient-specific random intercept** $u_i$. ([patsy.readthedocs.io][1])

[1]: https://patsy.readthedocs.io/en/latest/spline-regression.html?utm_source=chatgpt.com "Spline regression — patsy 0.5.1+dev documentation"
[2]: https://bookdown.org/ssjackson300/Machine-Learning-Lecture-Notes/splines.html?utm_source=chatgpt.com "Chapter 9 Splines | Machine Learning"
[3]: https://web.pdx.edu/~newsomj/mlrclass/ho_LR%20tests.pdf?utm_source=chatgpt.com "Random Effects Likelihood RatioTest Examples"
[4]: https://www.stata.com/manuals/memixed.pdf?utm_source=chatgpt.com "Multilevel mixed-effects linear regression"
[5]: https://pages.stat.wisc.edu/~ane/st572/notes/lec21.pdf?utm_source=chatgpt.com "Testing random effects"
[6]: https://besjournals.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2041-210x.2012.00261.x?utm_source=chatgpt.com "A general and simple method for obtaining R2 from ..."
[7]: https://easystats.github.io/performance/reference/r2_nakagawa.html?utm_source=chatgpt.com "Nakagawa's R2 for mixed models — r2_nakagawa - easystats"
[8]: https://bookdown.org/mike/data_analysis/sec-linear-mixed-models.html?utm_source=chatgpt.com "Chapter 8 Linear Mixed Models | A Guide on Data Analysis"


下面给出**Results（结果）章节的详细提纲与结论**，并逐条标注你应在论文中插入的 **.csv 数据表**与**图（plots）**。文字为中文，关键统计术语保留为 English，数值来自你前述 ipynb 运行日志与导出的文件。

---

# 1. 数据筛选与样本特征（Data filtering & cohort）

**主要发现**

* 原始样本 1082，经规则过滤（Weeks∈\[10,25]、GC∈\[40%,60%]、排除非整倍体、完整性）后得到 **N=555**。**本题仅采用男胎数据**（原因：Y 染色体浓度仅对男胎有意义），清洗后样本量 N=555，患者数=242（约 76.9% 存在重复测量）。总体删除 **48.7%**；其中 GC 规则删除 **449**（42.0%）。
* 达到临床阈值（Y ≥ 4%）的样本 **483/555（87.0%）**。

**变量单位与度量**
* **Y 浓度**：以比例形式 (0-1) 建模，例如 0.104 表示 10.4%
* **weeks**：孕周，单位为周
* **BMI**：母体身体质量指数，单位为 kg/m²
* **Y 变换**：未进行变换，直接使用原始比例值进行线性混合效应建模

**建议放入正文的表/图**

* 表 1（主文）：`../../output/results/p1_threshold_analysis.csv`（Above/Below 4% 的人数、比例、Weeks/BMI/Y 的均值与标准差）。
* 图 1（主文，分布与散点概览）：`../../output/figures/p1_distributions.png`、`../../output/figures/p1_linearity_check.png`（变量分布、Weeks/BMI 与 Y 的散点及线性趋势线）。

**（可选·方法敏感性，附录）**

* 图 S1：`output/figures/selection_bias_comparison.png`（GC 过滤前后保留 vs 移除样本的 Weeks/BMI/Y 对比；Weeks 存在小幅系统差异，p≈0.009）。
* 图 S2：`output/figures/gc_correlations.png`（GC 与 Weeks/BMI/Y 的相关性近似为 0，提示 GC 过滤对相关结构影响有限）。

---

# 2. 相关性分析（Correlation analysis）

**主要发现**（N=555）

* **Weeks–Y**：Pearson $r=0.1844$ ($p<0.0001$)、Spearman $r=0.1145$ ($p=0.0069$) → 弱正相关、显著。
* **BMI–Y**：Pearson $r=-0.1378$ ($p=0.0011$)、Spearman $r=-0.1498$ ($p=0.0004$) → 弱负相关、显著。

**建议放入正文的表**

* 表 2（主文）：`../../output/results/p1_correlations.csv`（Pearson/Spearman $r$、$p$ 与显著性标注）。

---

# 3. 基线线性模型（Baseline OLS）与稳健推断

**模型**：`Y ~ weeks + BMI`（OLS）
**主要发现**

* **R² = 0.0619** (6.19%)（Adj R² = 0.0585），整体 **F-test**: F-statistic=18.1995, **p<0.0001**。
* **weeks** 系数 $ \hat\beta_{weeks}=0.001842$（p<1e−6，正向）；**BMI** 系数 $ \hat\beta_{BMI}=-0.001975$（p=5.9e−5，负向）。
* 诊断：**heteroskedasticity**（Breusch–Pagan p<0.0001）；**非正态**（Jarque–Bera p<0.0001）。

**稳健推断（HC3）**

* 采用 **HC3 robust SE** 后，系数不变但 p 值更保守：weeks 仍显著（p=1.8e−5），BMI 仍显著（p=0.0013）。
* **稳健性设定详细**：聚类稳健标准误按 patient_id 聚类；bootstrap 重抽样 50 次（患者层面重抽样）；与基准 SE 对照显示结论一致性。

**建议放入正文/附录的表与图**

* 表 3（主文）：`../../output/results/p1_baseline_model_results.csv`（系数、SE、t、p、95% CI）。
* 表 S1（附录）：`../../output/results/p1_model_diagnostics_summary.csv`（Breusch–Pagan、Jarque–Bera、DW 等）。
* 表 S2（附录，对比 OLS vs OLS+HC3 vs logit-transform）：`../../output/results/p1_robust_model_comparison.csv`。
* 图 2（主文，残差诊断）：`../../output/figures/p1_model_diagnostics.png`（Residuals vs Fitted、Q–Q、Scale-Location、Residuals vs Weeks）。

---

# 4. 非线性检验：Natural Splines（df=3 / df=4）

**模型**：`Y ~ bs(weeks, df=3/4) + BMI`（OLS）
**主要发现**

* 与基线 OLS 相比：

  * **df=3**：**R² = 0.0943**（↑ 0.032）、**AIC = −2241.08**（改善 15.55）；**LR test p = 5.7e−05** → 非线性显著。
  * **df=4**：R² = 0.0954（略高），但 AIC = −2239.74（不如 df=3）。
* 结论：Weeks 的 **non-linearity** 明确，**df=3** 在拟合与复杂度间最优。

**未采用规格的理由**
* 尝试的二次项与交互项在信息准则/LRT上不及自然样条（df=3），且诊断无优势，故未采用。自然样条在边界稳定性、平滑性和复杂度控制方面表现最佳。

**建议放入正文的表**

* 表 4（主文，非线性模型对比）：`../../output/results/p1_model_comparison.csv`（Baseline / Interaction / Quadratic / Splines 的 R²、Adj R²、AIC、LR p-value）。

---

# 5. 重复测量与聚类：Mixed-Effects（Random Intercept）

**样本结构**

* Unique patients = **242**；有重复测量的个体 **76.9%**；平均 2.29 次/人。
* 线性 Mixed（无样条）显示：**ICC ≈ 0.70**，weeks 系数放大，BMI 绝对值缩小，提示 OLS 低估 SE 且偏倚效应量。

**建议放入正文/附录的表**

* 表 S3（附录，重复测量描述）：可来自你脚本的汇总（患者计数分布等）。
* 表 S4（附录，OLS vs Mixed（线性）系数与 SE 对比）：见你整合表 `../../output/results/p1_final_comprehensive_comparison.csv` 或 `p1_final_model_comparison_complete.csv` 中的线性 Mixed 条目。

---

# 6. **最终模型**：Mixed + Natural Splines（df=3）+ Random Intercept

**模型**：`Y ~ bs(weeks, df=3) + BMI + (1|patient_id)`
**主要发现（N=555, groups=242）**

* **随机效应**：random intercept variance = **0.000743**；residual variance = **0.000302**；**ICC = 0.7109** (71.1%变异来自患者间差异)。
* **固定效应**：

  * 截距：0.104076 (SE 0.020809, z=5.002, p=5.69×10⁻⁷)
  * 样条基函数：bs(weeks,df=3)[0]=0.027322 (z=2.591, p=0.0096)；bs(weeks,df=3)[1]=-0.003377 (不显著, p=0.640)；bs(weeks,df=3)[2]=0.058025 (z=9.971, p=2.05×10⁻²³)
  * **BMI**：-0.001332 (SE 0.000642, z=-2.073, p=0.038)
* **R² 分解（已修正）**：使用 Nakagawa & Schielzeth (2013) 方法，**marginal R² = 0.126988** (12.70%，仅固定效应），**conditional R² = 0.747584** (74.76%，固定+随机)；组间差异贡献占主导。
* 诊断图（population-level residual）：中心化良好，但 **Scale-Location** 仍提示异方差上升、Q–Q 上尾偏离 → 推断时建议并行报告 **cluster-robust SE** 或 bootstrap 置信区间（你已完成 cluster-robust 的敏感性对照）。

**建议放入正文的表与图**

* 表 5（主文，最终模型汇总）：`../../output/results/p1_final_model_summary.csv`（Mixed+Splines 的系数、z、p、随机方差、ICC）。
* 表 6（主文，R² 分解与诊断）：`../../output/results/p1_final_optimization_summary.csv`（marginal/conditional R²、log-likelihood、诊断统计）。
* 图 3（主文，Mixed 残差诊断）：`../../output/figures/p1_mixed_model_diagnostics.png`。
* 图 4（主文，partial effects 与阈值热图）：`../../output/figures/p1_comprehensive_partial_effects.png`

  * 左上：`weeks` 在不同 BMI（28/32/36/40）下的 **partial effect**（自然样条曲线）；
  * 右上：`BMI` 在不同孕周（12/15/18/22w）下的 **partial effect**；
  * 左下：**Clinical decision map**（连续 Y 预测值等高图）；
  * 右下：**Threshold achievement map**（Y≥4% 的区域着色，几乎全域 ≥4%）。

---

# 7. 临床二分类：Logistic（Y ≥ 4%）

**模型**：`logit(Y≥0.04) ~ bs(weeks, df=3) + BMI`（cluster-robust SE）
**主要发现**

* AIC ≈ **414.09**；在观察区间内，绝大多数组合的 **Pr(Y≥4%) 高**；
* 示例：Early/normal BMI ≈ **0.922**；Early/high BMI ≈ **0.760**；Mid/normal BMI ≈ **0.945**（与可视化一致，high BMI 在早孕期概率较低）。

**建议放入正文/附录的表**

* 表 7（主文，临床场景概率）：`../../output/results/p1_clinical_interpretation_table.csv`（不同周数与 BMI 组合的预测值/概率）。
* 表 S5（附录，logistic 详细系数与显著性）：`../../output/results/p1_logistic_model_results.csv`。

---

# 8. 全模型对比与选型依据（Model selection）

**数值要点**

* AIC / R²：`OLS + Splines_df3` 在 **AIC** 上最优（−2241.08），**R² = 0.0943**；但其**未处理 clustering**。
* Mixed 系列反映出 **ICC=0.7109** 的强聚类，若忽略将低估 SE。
* 综合理论（non-linearity + clustering）与实证（LR test、R² 分解、诊断图），**最终选用 Mixed + Splines（df=3）+ Random Intercept**。

**临床阈值性能（≥4%）**
* **ROC AUC**: 0.9519 (优秀判别能力)
* **4%阈值性能**: 敏感性 0.990 (99.0%)，特异性 0.458 (45.8%)，准确性 0.921 (92.1%)，精确度 0.925
* **最优阈值**: 0.0525 (5.25%, Youden指数最大化)
* **临床建议**: 采用分级筛查策略（2-3%初筛，5%确认阈值）应对低特异性问题

**建议放入正文的表**

* 表 8（主文，综合对比）：`../../output/results/p1_final_comprehensive_comparison.csv` 或 `../../output/results/p1_final_model_comparison_complete.csv`（列出各模型的 Formula、Approach、R\_squared/AIC、是否考虑 clustering/non-linearity、临床用途）。

---

## 建议在论文中对应的「Results」小节结构

1. Data filtering & cohort → 表 1 + 图 1；（方法敏感性放附录：图 S1–S2）
2. Correlation analysis → 表 2。
3. Baseline OLS & Robust inference → 表 3 + 表 S1–S2 + 图 2。
4. Non-linearity via Natural Splines → 表 4。
5. Repeated measures & Mixed（线性）→ 表 S3–S4。
6. **Final model: Mixed + Splines** → 表 5–6 + 图 3–4（核心）。
7. Clinical threshold (Logistic) → 表 7（主文）+ 表 S5（附录）。
8. Comprehensive comparison & selection rationale → 表 8（主文）。

> 以上文件路径与图名即为你 ipynb 自动导出的产物；按上述结构排布，即可形成一套可直接进入论文「结果」章节与附录的材料。
