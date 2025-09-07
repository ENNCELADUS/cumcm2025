# 最快可落地方案（Problem 4）

> **目标与度量**：以 **AB** 为阳性标签（任一 T13/18/21 异常），在**低误报**约束下（固定 **FPR≤1%**）**最大化召回率**；主模型 **Elastic-Net Logistic**（可解释、近似校准），辅模型 **XGBoost**（捕捉非线性）。全流程 **Group（按孕妇ID）** 防泄漏。

---

## 1. 数据预处理（Data Preprocess）

1. **筛选样本（女胎）**
   1.1 选取女胎记录（无 Y 相关信号/标注）。
   1.2 生成标签：$Y=\mathbf{1}\{\text{AB为阳性}\}$；去除标签缺失样本。
   1.3 记录分组键（孕妇ID，记为 $g$）。

2. **基础质控（QC）**
   2.1 设全局 GC 合理区间为 **\[40%, 60%]**：超出者标记 `gc_outlier`（后续加权或剔除，默认**保留并加权**）。
   2.2 读段/比对/重复率：对 reads、map\_ratio、dup\_ratio、unique\_reads 做 1%–99% **winsorize**（仅在训练集上估计阈值）。
   2.3 删除明显异常：训练集中 **下位1%** 的总读段（或 unique\_rate）样本直接剔除；测试集仅标注不剔除。

3. **缺失与派生**
   3.1 数值缺失：训练集**中位数填补**，并为缺失比例 >1% 的变量增加缺失指示列。
   3.2 派生变量：
   $\max Z=\max(Z_{13},Z_{18},Z_{21})$；
   三条染色体指示：$\mathbb{1}\{|Z_{j}|\ge 3\}$（j∈{13,18,21}）；
   读段质量率：$\mathrm{uniq\_rate}=\frac{\text{unique\_reads}}{\text{reads}}$。
   3.3 共线精简：BMI 优先于身高/体重；保留 `reads` 或 `uniq_rate` **其一**（保 `uniq_rate`），去除与其 Spearman ρ≥0.95 的强相关项（仅在训练集判定）。

4. **特征矩阵（最终入模）**

   $$
   \mathbf{x}=\{Z_{13},Z_{18},Z_{21},Z_X,\ \text{GC}_{global},\ \text{GC}_{13/18/21},\ \text{map\_ratio},\ \text{dup\_ratio},\ \text{uniq\_rate},\ \text{BMI},\ \text{age},\ \max Z,\ \mathbb{1}\{|Z_j|\ge3\}\}
   $$

   其中 `GC_{13/18/21}` 可用则入；不可用则仅用全局 GC。**标准化**：仅对 **Logistic** 所用的数值特征做 **z-score**（以训练集均值/方差）。

---

## 2. 划分与不平衡处理（Split & Imbalance）

5. **单次固定切分（快速稳健）**
   5.1 **Group-Stratified 80/20**（按 $g$ 分组且分层于 $Y$）：训练/测试。
   5.2 在训练集再做 **Group-Stratified 90/10** 切出 **校准集**（calibration）。
   5.3 训练集内部用于交叉验证（见 §3）。

6. **类别不平衡**
   6.1 Logistic：类权重 $w_1:w_0 = \frac{n_0}{n_1}:1$。
   6.2 XGBoost：`scale_pos_weight = n_0/n_1`（在每个训练折重新计算）。
   6.3 对 `gc_outlier` 样本施加 **样本权 0.7**（训练期；测试不加权）。

---

## 3. 模型与交叉验证（Models & Cross-Validation）

7. **主模型：加权 Elastic-Net Logistic**
   目标函数（带类权与EN正则）：

   $$
   \max_{\beta}\ \sum_{i} w_{y_i}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]-\lambda\big(\alpha\|\beta\|_1+\tfrac{1-\alpha}{2}\|\beta\|_2^2\big),
   \quad p_i=\sigma(\beta_0+\beta^\top x_i)
   $$


实现建议：`LogisticRegression(solver='saga', penalty='elasticnet', max_iter=5000)`，使用 `class_weight`。

**网格（建议）**

* l1\_ratio（等价于 $\alpha$）：`[0.0, 0.25, 0.5, 0.75, 1.0]`
* C = $1/\lambda$（对数刻度）：`[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`
* penalty solver / 额外：`solver='saga'` 固定，`max_iter=5000`。
* class\_weight：`{0:1, 1: n0/n1}`（在每折计算）

**理由与实践建议**

* 扩大 C 范围（含更小值）有助于在强正则与弱正则间找到平衡；更多 l1\_ratio 值允许从纯 L2（0.0）到纯 L1（1.0）细粒度选择。
* 使用 `RandomizedSearchCV(n_iter=60)` 或 `GridSearchCV`（若计算预算允许）在 GroupKFold(k=5) 上搜索；评分主指标 `average_precision`（PR-AUC），辅指标 `recall`@FPR≤1%（阈值搜索后评估）。


8. **辅模型：XGBoost（二分类，logistic 概率）**
   固定正则与子采样以防过拟合，**不做早停**（更快）：

实现建议：使用 early stopping（`early_stopping_rounds=50`）配合内折验证以避免过拟合并缩短训练时间；并行 `n_jobs=-1`。

**网格（建议）**

* `max_depth`: `[3,4,5,6,8]`
* `learning_rate` (eta): `[0.01, 0.03, 0.05, 0.1]`
* `n_estimators`: `[200, 400, 800, 1200]`（与 early stopping 联用）
* `subsample`: `[0.6, 0.8, 1.0]`
* `colsample_bytree`: `[0.6, 0.8, 1.0]`
* `min_child_weight`: `[1,3,5]`
* `gamma`: `[0, 0.5, 1, 2]`
* `reg_alpha` (L1): `[0, 0.1, 1.0]`
* `reg_lambda` (L2): `[0.5, 1.0, 5.0]`
* `scale_pos_weight`: `n0/n1`（在每折重算）

**理由与实践建议**

* 包含更深的 `max_depth` 与更小的 `learning_rate` 有助于学习更复杂信号同时保持泛化（用更多树或 early stopping 控制）。
* 使用 `RandomizedSearchCV(n_iter=80~120)`，每次训练在内折内用一部分作为 early-stop 验证；主评分 `average_precision`。

9. **交叉验证与模型选择（单层、分组）**
   9.1 在训练集上做 **GroupKFold(k=5)**；每折：按步骤2–4的**训练流程**重建（避免泄漏）。
   9.2 **主评分**：**PR-AUC**；**次评分**：在折内阈值选定使 **FPR≤1%** 的 **Recall**。
   9.3 各模型以 **PR-AUC 平均值**选最优超参；若 PR-AUC 持平，选 **Recall\@FPR≤1%** 更高者。
   9.4 **确定赢家模型**：在 Logistic 与 XGBoost 间，按 9.3 的准则选**单一最终模型**（不做 Stacking；最快）。

10. **概率校准与阈值**
    10.1 对**最终赢家模型**在**训练集**全量拟合（含类权/scale\_pos\_weight）。
    10.2 用\*\*校准集（10%）\*\*做 **Platt scaling（逻辑校准）** 得到校准概率 $\hat p$。
    10.3 在校准集上选择阈值 $\tau^\*$：

$$
\tau^\*=\arg\max_{\tau}\ \text{Recall}(\tau)\quad \text{s.t.}\ \text{FPR}(\tau)\le 0.01
$$

---

## 4. 最终测试（Hold-out Test）

11. **一次性锁定评估**
    11.1 将“最终模型 + 校准器 + 固定阈值 $\tau^\*$”应用于**测试集**（从未参与训练/调参/校准）。
    11.2 报告：**Confusion Matrix**、Recall（含 Recall\@FPR≤1%）、FPR、Precision、PR-AUC、ROC-AUC、Brier、校准曲线与 **PPV/NPV**。
    11.3 **基线对照**：|Z|≥3 的规则法在同一测试集上的各项指标（尤其漏检率）并列表对比。
    11.4 输出 **个体级解释**：

* 若最终模型为 Logistic：系数、OR 与 95%CI；
* 若为 XGBoost：TOP-k 特征的置换重要性或 SHAP（快速版取全局均值排名）。

---

> **备注（实现细节约定）**
> • 所有预处理（winsorize/标准化/缺失填补/相关性筛除）均 **仅在训练集拟合参数**，再用于校准集/测试集变换。
> • GroupKFold 的 **group=孕妇ID**，确保同一孕妇不跨折/不跨集合。
> • 全流程随机种子固定（例：`seed=42`），确保结果可重复。
