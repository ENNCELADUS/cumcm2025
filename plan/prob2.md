# Problem 2

## 数据预处理 To-do List

1. **数据读取与变量解析**

   * 读取附件 Excel 中的“男胎检测数据”。
   * 将孕周（J列）转为小数周（如 11w+6 → 11.857）。
   * 将 BMI（K列）转为浮点数。
   * Y 浓度（V列）保留为百分比形式（0–0.2）。

2. **基础质量过滤**（与问题 1 一致）

   * 孕周范围：10–25 周。
   * GC 含量（P列）：40%–60%。
   * 删除 13/18/21 染色体异常（AB列非空）。
   * 删除关键字段缺失（孕周、BMI、Y 浓度）。

3. **异常值检测与剔除**

   * 对孕周、BMI、Y 浓度分别进行 **IQR** 检测，剔除异常值。

**IQR 法则**

   * 对变量 $X$（如 Y 浓度、BMI、孕周），计算：

     * Q1（25%分位）、Q3（75%分位）
     * IQR = Q3 – Q1
   * 定义异常点：

     $$
     X < Q1 - 1.5 \times IQR \quad \text{或} \quad X > Q3 + 1.5 \times IQR
     $$
   * 若数据分布近似正态，可再用 **3σ 检查**（双保险）。

4. **男胎筛选**

   * 仅保留有 Y 浓度记录的样本。

5. **多次检测处理**

   * 按孕妇 ID（B列）分组。
   * 对每位孕妇，提取首次出现 **Y 浓度 ≥4%** 的孕周，定义为“最早达标时间”。
   * 若始终 <4%，标记为“未达标”。

6. **BMI 分组准备**

   * 方案 A：按临床经验区间（<25, 25–30, 30–35, 35–40, ≥40）。
   * 方案 B：用数据驱动方法（聚类、决策树）自动分组，得到自适应区间。

7. **结果数据集导出**

   * 变量：孕妇 ID、BMI、最早达标孕周、是否达标（≥4%）。
   * 用于后续生存分析 / Kaplan–Meier 曲线 / 分组均值比较。

---

Great—here’s a crisp brainstorm + verdict on your PDF plan, then a concrete, math-backed step-by-step you can execute.

# 1) Brainstorm: what model families fit Problem 2?

Goal recap: for each male-fetus pregnancy, define the earliest gestational week $T$ at which $Y\ge 4\%$ (fetal fraction “FF” threshold). Many moms have multiple tests → repeated measures → the “event time” is **interval-censored** (you only know it crossed between two visits), or **right-censored** (never crossed), and measurement noise matters.

Viable model families:

* **Interval-censored survival models (recommended core)**

  * Parametric AFT (Weibull / log-logistic) or semiparametric Cox for interval censoring.
  * Pros: natively handles censoring & “first-time reaches 4%”. Outputs $S(t\mid x)$ and directly gives “earliest week $w^*$” per BMI group at a target confidence $\tau$ (e.g. 0.9 or 0.95).
  * Cons: need distributional choice (AFT); Cox-IC is heavier computationally.

* **Discrete-time survival (logit / cloglog hazard)**

  * Transform to person-period data across weeks; model hazard $h_t(x)$.
  * Pros: flexible with splines & interactions; easy to add random effects and class weights.
  * Cons: requires constructing intervals per person; weeks with no observation need assumptions.

* **Random Survival Forest (RSF) / Gradient-boosted survival**

  * Pros: strong nonlinear capture, works with censoring, no parametric assumptions.
  * Cons: less directly interpretable; we’ll still compute $w^*$ from $\hat S_g(t)$.

* **Your PDF’s two-step ML (classification: ever reaches ≥4%; regression: time among reachers)**

  * LASSO for variable selection + Random Forest for nonlinearity, then BMI binning via CART, plus Monte-Carlo error analysis.  &#x20;
  * Pros: simple and practical, good for competition timelines.
  * Caveat: ignores censoring structure → selection bias (time model sees only “reachers”); the “true” event may lie between two visits (interval-censoring), which two-step doesn’t model.

**My recommendation:** keep your PDF’s LASSO+RF pipeline as a **baseline**, but promote a **survival-first track** (interval-censored AFT or RSF). We’ll still use your CART/importance to propose BMI cutpoints and your Monte-Carlo noise study to stress-test the recommendations. (The PDF already suggests the CART-based grouping and $\tau$-based earliest week rule.  )

# 2) Does your PDF plan “work”?

**Yes, with caveats.**

* It cleanly defines “first week reaching 4%”, treats never-reach as censoring, and proposes two-step (classify reach / then regress time for reachers) with LASSO feature selection and RF for nonlinearity—this is workable and implementable fast. &#x20;
* It also lays out BMI cutpoint discovery via CART / RF importance and a rule to choose the **earliest week** where the modeled reach-probability exceeds a target confidence $\tau$. That policy mapping is correct and publishable. &#x20;
* You further propose Monte-Carlo to propagate measurement error—good and necessary.&#x20;

**But**: the two-step pipeline discards information from censored/interval-censored cases when predicting $T$ and can bias the recommended week per BMI group. A survival model fixes this while keeping your grouping and error-simulation ideas intact.

# 3) Step-by-step plan (with formal math)

## Step 0. Preprocess (Problem 2)

Same QC as Problem 1 + **IQR or 3σ** outlier filters on weeks, BMI, FF; keep **male** and quality reads only; handle repeats per mom; compute earliest crossing (see below). (Preprocessing guidance already summarized in your doc set. )

---

## Step 1. Construct event intervals per pregnancy

For mom $i$, with ordered visits $w_{i1}<\cdots<w_{iJ_i}$, and observed FF $y_{ij}$.

Define the **event** “FF reaches 4%” and the **event time** $T_i$.

* If there exists $j^\star$ s.t. $y_{i,j^\star}\ge 0.04$ and all $y_{i,1:(j^\star-1)}<0.04$, then

  $$
  T_i\in (L_i, R_i] := (w_{i,j^\star-1},\, w_{i,j^\star}],
  $$

  an **interval-censored** observation (if $j^\star=1$, it is **left-censored**: $T_i\le w_{i1}$).

* If all $y_{ij}<0.04$, then **right-censored** at $C_i=w_{iJ_i}$: $T_i> C_i$.

We will carry a covariate vector $x_i$ (BMI, age, GC, depth, IVF, etc.).

(Your PDF already defines “达标时间” and acknowledges censoring. )

---

## Step 2A (Core): Interval-censored survival via AFT

Choose an AFT model: $\log T_i = \beta^\top x_i + \sigma \varepsilon_i$, with $\varepsilon$ following:

* Weibull-AFT: $\varepsilon$ is Gumbel; or
* Log-logistic AFT: $\varepsilon$ is logistic.

Let $S(t\mid x)=\Pr(T>t\mid x)$. For interval-censored $(L_i,R_i]$, the contribution to the likelihood is

$$
\mathcal{L}_i=
\begin{cases}
S(L_i\mid x_i)-S(R_i\mid x_i), & \text{interval-censored},\\
S(R_i\mid x_i), & \text{right-censored (}\,T_i>R_i\,\text{)},\\
1-S(R_i\mid x_i), & \text{left-censored (}\,T_i\le R_i\,\text{)}.
\end{cases}
$$

Maximize $\prod_i \mathcal{L}_i$ for $\hat\beta,\hat\sigma$. Derive **group-specific** survival

$$
S_g(t)=\mathbb{E}_{x\in g}\big[S(t\mid x)\big].
$$

**Best week for group $g$** at confidence $\tau$:

$$
w_g^\star=\inf\{t: 1-S_g(t)\ge \tau\}.
$$

This exactly matches your “choose the earliest week where the modeled reaching-probability exceeds a set confidence” rule.&#x20;

> Alternative core: **Discrete-time survival** with cloglog link
> Reshape to person-period data with weekly bins $t$. Model hazard
>
> $$
> \Pr(T=t\mid T\ge t,x)=1-\exp\{-\exp(\alpha_t+f(x))\},
> $$
>
> where $f(\cdot)$ uses splines / interactions. Compute $S(t\mid x)=\prod_{u\le t}(1-h_u(x))$ and proceed as above.

---

## Step 2B (Nonlinear check): Random Survival Forest (RSF)

Fit RSF on $(L_i, R_i, x_i)$. Obtain $\hat S(t\mid x)$ and $S_g(t)$, then $w_g^\star$ as above. Use RSF **variable importance** to verify BMI dominance and discover interactions (e.g., BMI×GC, BMI×age). Keep RSF as a **robustness** model alongside AFT.

(Your document already allows tree-based modeling and grouping with CART/importance.  )

---

## Step 3. BMI grouping (data-driven, clinically coherent)

Two equivalent operational routes (you can do both and cross-check):

* **Model-guided CART on $T$** (or on survival quantiles):

  * Train a **single-variable CART** on BMI to predict the modeled **median** $\tilde T(x)$ (from AFT/RSF). Prune by CV, then **merge** tiny or clinically indistinct bins to get 4–5 groups (normal / overweight / obese I / obese II / ≥40).&#x20;

* **Confidence-driven binning**:

  * For candidate cut set $B=\{b_1,\dots,b_{K-1}\}$, compute $S_g(t)$ and $w_g^\star$ at target $\tau$.
  * Choose $B$ minimizing a penalized risk, e.g.

    $$
    \mathcal{R}(B)=\sum_g \pi_g\, w_g^\star \;+\; \lambda K,
    $$

    where $\pi_g$ is group prevalence; tune $\lambda$ by CV/BIC to avoid over-segmentation.

Either way, you end with clear intervals and **earliest safe weeks** $\{w_g^\star\}$ with uncertainty bands.

(Choosing $w_g^\star$ via thresholded reach-probability at level $\tau$ is exactly in your PDF. )

---

## Step 4. Measurement-error stress test (as in your PDF)

Adopt a simple additive noise model for FF (on the percentage scale or logit scale):

$$
y^{\text{obs}}_{ij}=y^{\text{true}}_{ij}+\varepsilon_{ij},\qquad \varepsilon_{ij}\sim \mathcal{N}(0,\sigma^2),
$$

(or heteroscedastic $\sigma=\sigma(w,GC,\text{depth})$). Re-jitter FF, **rebuild intervals**, **refit models**, and recompute $\{w_g^\star\}$ over $B$ Monte-Carlo runs (e.g., $B{=}1000$). Report mean/CI for each $w_g^\star$ and the frequency each group still meets the $\tau$ target. (This is exactly what your PDF suggests.)&#x20;

---

## Step 5. Keep the PDF two-step ML as a baseline (for ablation & speed)

* **Step 5.1** LASSO variable selection (or L1-logistic for “ever reaches”):

  $$
  \min_{\beta}\;\frac1n\sum_i \ell\big(y_i,\,\beta^\top z_i\big)+\lambda\|\beta\|_1,
  $$

  on standardized covariates; CV for $\lambda$.&#x20;

* **Step 5.2** Random Forest:

  * Classifier for $\Pr(\text{reach ever}\mid x)$; regressor for $T$ on reachers; (fusion variants A/B are in your plan). &#x20;

* **Step 5.3** BMI cutpoints + $\tau$-rule for $w_g^\star$ (exactly as in §Step 3).&#x20;

This gives a quick, strong baseline. Keep it in the paper as a **comparison arm** to show why survival-aware modeling is preferable.

---

## Step 6. Validation & reporting

* **Splits**: patient-level, nested CV; avoid leakage among repeated visits. (Your preprocessing section already notes stratified by patient.)&#x20;
* **Metrics**: C-index, time-dependent Brier/IBS, calibration of $\hat S_g(t)$; agreement between AFT and RSF $\{w_g^\star\}$.
* **Sensitivity**: redo with IQR/3σ filters toggled; redo with alternative $\tau$ (0.9 / 0.95).
* **Deliverables**: table of BMI bins, $w_g^\star$ (mean, 95% CI), % of moms expected to reach ≥4% by $w_g^\star$, and Monte-Carlo stability.

---

## Compact mathematical block you can drop into the report

**Problem definition.** For pregnancy $i$ with covariates $x_i$, event time $T_i$ is the earliest gestational week FF $\ge 4\%$. Observations induce censoring interval $(L_i,R_i]$ as above.

**AFT model.** $\log T_i=\beta^\top x_i+\sigma\varepsilon_i$. With CDF $F_\varepsilon$,

$$
S(t\mid x)=1-F_\varepsilon\!\left(\frac{\log t-\beta^\top x}{\sigma}\right).
$$

Likelihood $\mathcal{L}=\prod_i \mathcal{L}_i$ with $\mathcal{L}_i$ given in Step 2A. Estimate $(\hat\beta,\hat\sigma)$ by MLE; obtain group survival $S_g(t)$.

**Policy rule.** For BMI group $g$ and target confidence $\tau\in(0,1)$, recommend

$$
w_g^\star=\inf\{t:\;1-S_g(t)\ge \tau\}.
$$

If a cost $L(t)=\alpha t+\gamma\cdot\text{(false-negative risk at }t)$ is provided, choose $t$ minimizing $L(t)$ (your PDF also allows such a loss-based refinement).&#x20;

**Monte-Carlo error propagation.** Replace $y$ with $y+\varepsilon$, rebuild $(L_i,R_i]$, refit, and repeat $B$ times to obtain $\{w_g^\star\}$ distribution.&#x20;

---

## Why this will satisfy the judges

* Aligns with your original blueprint (LASSO+RF, CART grouping, $\tau$-rule, MC error). &#x20;
* Upgrades the statistical core to survival analysis so the **“earliest week”** is inferred with the **correct censoring math**—a common pitfall fixed.
* Produces clean, defendable outputs: BMI bins + earliest safe week (with CIs) + sensitivity to measurement error.

If you want, I can draft the precise “Methods” subsection in LaTeX with these equations and a short “Algorithm 1” (data → intervals → AFT/RSF → bins → $w_g^\star$).

---

## 数学上（Model 层面）

1. **问题定义**

   * 每位孕妇存在一个“事件时间” $T_i$：即 Y 染色体浓度首次达到或超过 4% 的孕周。
   * 实际观测数据是多次检测记录，所以我们只能知道 $T_i$ 落在某个区间内（interval-censored），或直到最后一次检测仍未达标（right-censored）。
   * 我们希望根据孕妇的 **BMI** 对人群进行分组，并估计每组的最早可靠检测周数 $w_g^\*$，使得在该周数时，组内大部分孕妇（如 90% 或 95%）已经达标。

2. **采用模型：Accelerated Failure Time (AFT) 生存模型**

   * 假设

     $$
     \log T_i = \beta^\top x_i + \sigma \varepsilon_i
     $$

     其中 $x_i$ 是协变量（主要是 BMI），$\varepsilon$ 服从 Weibull 或 log-logistic 分布。
   * AFT 可以处理 interval censoring，估计在不同 BMI 下的 **生存函数**：

     $$
     S(t \mid x) = \Pr(T > t \mid x)
     $$
   * 对某一 BMI 分组 $g$，定义群体生存函数：

     $$
     S_g(t) = \mathbb{E}_{x \in g}[S(t \mid x)]
     $$
   * 最佳检测时点（策略规则）：

     $$
     w_g^\* = \inf \{ t : 1 - S_g(t) \ge \tau \}
     $$

     其中 $\tau$ 是预设置信度（如 0.90 或 0.95），确保大多数孕妇达标。

3. **BMI 分组方法**

   * **数据驱动**：用决策树（CART）或聚类自动寻找 BMI 的切分点，使得组内差异小、组间差异大。
   * **风险函数优化**：

     $$
     \mathcal{R}(B) = \sum_g \pi_g \cdot w_g^\* + \lambda K
     $$

     其中 $\pi_g$ 是组占比，$K$ 是组数，$\lambda$ 控制复杂度。通过优化风险函数找到最合理的分组和时点。

4. **检测误差处理**

   * 设观测值：

     $$
     y_{ij}^{obs} = y_{ij}^{true} + \varepsilon_{ij}, \quad \varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2)
     $$
   * 用 Monte-Carlo 模拟，在重复加噪声后重新计算 $w_g^\*$，得到均值和置信区间，从而量化检测误差对策略的影响。

---

## 直观上（Intuitive 理解）

* **为什么用 survival model（AFT）？**
  因为我们研究的本质是“多久以后 Y 浓度会超过阈值 4%”。这和“寿命分析”类似：

  * “事件” = 浓度达标；
  * “时间” = 孕周；
  * “删失” = 一直没达标或只知道达标发生在某区间。
    普通回归不能很好处理这些情况，而 survival model 能自然应对。

* **BMI 的作用**
  BMI 高的孕妇，胎儿 DNA 浓度上升通常更慢 → 达标孕周更晚。
  所以我们需要按 BMI 分组，低 BMI 组可以更早检测，高 BMI 组需要推迟，否则 false negative 风险更高。

* **策略的本质**
  对每个 BMI 组，我们绘制出“孕周 vs 达标概率”的曲线。当曲线到达 90% 或 95% 时，就把对应的孕周作为该组的“最佳检测时点”。这样既保证了足够准确性，又尽量早地发现问题。

* **误差模拟的意义**
  实际测序有噪声，边缘样本（浓度刚好接近 4%）容易被误判。通过 Monte-Carlo 模拟，我们能看出“推荐的最佳检测时点”在噪声下是否稳健，必要时再调整。

---

👉 总结一句话：
**我们的方法是在“时间到事件”的框架下，用 interval-censored 的 AFT survival model 描述 Y 浓度达标时间与 BMI 的关系；再通过 BMI 分组 + 生存曲线推断出每组的最佳检测孕周，并用 Monte-Carlo 模拟验证其鲁棒性。**
