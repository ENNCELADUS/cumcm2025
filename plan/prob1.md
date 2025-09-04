# Problem 1: Y-Chromosome Concentration Modeling Plan

## Dataset Overview

**Data Source**: `src/data/data.xlsx` with two sheets:
- **男胎检测数据**: 1,082 male fetus samples with Y chromosome data
- **女胎检测数据**: 605 female fetus samples (no Y chromosome data)

**Analysis Focus**: Male fetus data only (Y chromosome concentration available)

---

## 1. Clarify Targets & Fields

### Objective
Model how fetal Y-chromosome concentration depends on gestational age and maternal BMI, and test statistical significance of these relationships.

### Key Variables (Actual Column Names)
- **检测孕周**: Gestational weeks (format: "11w+6", "15w+6", "13w") → primary predictor
- **孕妇BMI**: Maternal BMI → primary predictor  
- **Y染色体浓度**: Y chromosome concentration (already in proportions 0.01-0.23) → response variable
- **Y染色体的Z值**: Y chromosome Z-score → alternative measure
- **孕妇代码**: Patient code → for repeated measures analysis

### Domain Context
- **Measurement window**: 10-25 weeks gestational age (typical NIPT)
- **Reliability threshold**: Y≥4% (0.04) considered reliable for male fetus detection
- **Expected patterns**: Y concentration ↑ with weeks, ↓ with BMI (literature basis)

### Deliverables
- All objectives, data dictionary, assumptions documented in **notebook cells** (markdown)
- No separate files needed - keep everything in `src/notebooks/01_data_exploration.ipynb`

## 2. Load & Tidy the Data

### Data Processing Steps
1. **Load male fetus data** from '男胎检测数据' sheet (1,082 samples)
2. **Parse gestational weeks**: Convert "11w+6" format to decimal weeks (11.86)
3. **Process variables**:
   - `weeks` = parsed decimal weeks from 检测孕周
   - `BMI` = numeric conversion of 孕妇BMI
   - `V_prop` = Y染色体浓度 (already in proportions, no conversion needed)
4. **Apply filters**: 10-25 weeks gestational age, remove missing values

### Expected Results
- **Clean dataset**: ~1,068 samples after filtering
- **Weeks range**: 11.0 to 24.9 weeks
- **Y concentration range**: 0.010 to 0.234 (proportions)
- **Above 4% threshold**: ~86% of samples

### Deliverables
- Clean dataset variable `df_clean` in notebook memory
- Data processing documented in **notebook cells** with markdown explanations
- Summary statistics displayed in notebook output
- No external files - all in `src/notebooks/01_data_exploration.ipynb`

## 3. EDA & Linearity Check

### Analysis Components
1. **Scatter plots**: Y concentration vs. weeks and vs. BMI with 4% threshold line
2. **Correlation analysis**: Pearson and Spearman correlations
3. **Pattern assessment**: Check for linearity, outliers, variance patterns
4. **Repeated measures**: Check for multiple samples per patient

### Expected Findings (Based on Data)
- **Weeks correlation**: r ≈ 0.12, p < 0.001 (positive, significant)
- **BMI correlation**: r ≈ -0.15, p < 0.001 (negative, significant)
- **Sample size**: 1,068 clean observations
- **Threshold crossing**: 86.4% above 4% reliability threshold

### Deliverables
- **Interactive plots** displayed in notebook (matplotlib/seaborn)
- Correlation results shown as DataFrame in notebook output
- EDA insights documented in **notebook markdown cells**
- No external files - visualizations embedded in notebook

## 4. Baseline Model (OLS Regression)

### Model Specification
**Core Model**: `Y_concentration ~ β₀ + β₁·weeks + β₂·BMI + ε`

**Extensions to Consider**:
- Interaction term: `β₃·weeks×BMI`
- Quadratic term: `β₄·weeks²` (if non-linearity detected)

### Statistical Tests
1. **Global significance**: F-test for overall model
2. **Individual predictors**: t-tests for β₁ (weeks), β₂ (BMI)
3. **Model fit**: R², Adjusted R², AIC/BIC
4. **Assumptions**: Residuals analysis, heteroscedasticity, multicollinearity (VIF)

### Deliverables
- **Model objects** stored in notebook variables (`model_ols`, `model_robust`, etc.)
- **Model summaries** displayed directly in notebook output
- **Diagnostic plots** embedded in notebook cells
- **Assumption test results** printed in notebook
- **Decision logic** documented in notebook markdown
- **Optional**: Save final model objects to `src/models/p1_final_model.pkl` if needed for production

## 5. Robust Alternative Models & Improvement Framework

### Overview
**Given Baseline OLS Issues**: Heteroscedasticity detected, non-normal residuals, low R² (6.2%) → Implement robust alternatives by priority to ensure reliable statistical inference and better model fit.

### Priority 1: Robust Inference (Same Formula)

#### 1.1 OLS + Robust Standard Errors (HC3)
- **Approach**: Maintain `Y ~ weeks + BMI` while replacing "nonrobust" with **HC3** (or White) robust standard errors
- **Purpose**: Provides reliable p-values and confidence intervals under heteroscedasticity
- **Implementation**: `model.fit(cov_type='HC3')` in statsmodels
- **Expected**: Coefficient significance should persist with robust SEs

#### 1.2 Weighted Least Squares (WLS) - Optional
- **Weights**: Approximate `1/Var` using `1/fitted²` or variance estimated by gestational week segments
- **Purpose**: Improve efficiency under heteroscedasticity
- **Reporting**: Use **Robust OLS as primary result** for safety, WLS as supplementary

### Priority 2: Models for Proportion-Type Dependent Variable (Recommended)

#### 2.1 Beta Regression (Logit Link)
- **Rationale**: `Y∈(0,1)` with natural heteroscedasticity → Beta regression better matches data generation mechanism
- **Formula**: `Y ~ weeks + BMI` (can extend to `Y ~ splines(weeks, df=3) + BMI`)
- **Comparison**: Compare pseudo-R² and AIC with OLS
- **Library**: Use `statsmodels.genmod` or equivalent

#### 2.2 Logit Transformation Alternative
- **If Beta regression unavailable**: Use `logit(Y)` transformation + OLS + Robust SE
- **Formula**: `logit(Y) ~ weeks + BMI` with robust standard errors
- **Purpose**: Approximate Beta regression when computational resources limited

#### 2.3 Logistic Regression (Clinical Focus)
- **Target**: Binary outcome `I(Y≥0.04)` for clinical threshold
- **Formula**: `logit Pr(Y≥0.04) ~ weeks + BMI (+ weeks×BMI)`
- **Outputs**: Odds Ratios (OR), AUC, calibration curves
- **Clinical Value**: Highly aligned with "threshold achievement" decisions
- **Designation**: **Model 3 (Clinical Decision Model)**

### Priority 3: Mild Non-linearity (Evidence-Based)

#### 3.1 Restricted Cubic Splines (RCS)
- **Rationale**: Instead of relying on small R² improvements from `weeks²` or interactions, use natural splines to capture S-shaped patterns observed in residuals
- **Formula**: `Y ~ ns(weeks, df=3) + BMI` (applied to Beta or logit-OLS)
- **Comparison**: Use LR test/AIC for model selection
- **Principle**: Avoid overfitting - use df=3 or 4 maximum

#### 3.2 Partial Effect Curves
- **Output**: Generate partial effect plots showing non-linear relationships
- **Purpose**: Visualize how Y concentration changes across gestational weeks
- **Implementation**: Marginal effects at representative values

### Priority 4: Repeated Measures (If Applicable)

#### 4.1 Check for Repeated Measurements
- **Analysis**: Examine if same `patient_code` appears multiple times
- **Impact**: Correlated residuals can cause underestimated standard errors

#### 4.2 Mixed-Effects Model
- **Formula**: Random intercept by patient - `Y ~ weeks + BMI + (1|patient_code)`
- **Purpose**: Account for patient-level clustering
- **Library**: Use `statsmodels.MixedLM` or equivalent

#### 4.3 Cluster-Robust Standard Errors
- **Alternative**: Cluster-robust SE by patient code
- **Purpose**: Correct for correlated residuals without random effects
- **Implementation**: `cov_type='cluster'` with cluster variable

### Implementation Strategy

#### Model Sequence
1. **Model 1**: Baseline OLS (already completed)
2. **Model 1R**: OLS + HC3 Robust SE
3. **Model 2**: Beta regression (logit link)
4. **Model 2S**: Beta + Natural splines (if non-linearity significant)
5. **Model 3**: Logistic regression (Y≥4%)
6. **Model 4**: Mixed-effects (if repeated measures detected)

#### Model Comparison Framework
- **Statistical**: Compare AIC, BIC, pseudo-R² across models
- **Clinical**: Evaluate predictive performance for 4% threshold
- **Robustness**: Assess coefficient stability across specifications

### Deliverables
- **Robust models** as notebook variables (`model_robust`, `model_beta`, `model_logistic`, etc.)
- **Model comparison table** with AIC, R², coefficient consistency
- **Diagnostic improvements** showing resolution of heteroscedasticity/normality issues
- **Clinical interpretation** with robust confidence intervals
- **Sensitivity analysis** demonstrating coefficient stability
- **Optional**: Export final model comparison to `output/results/p1_robust_model_comparison.csv`

## 6. Effect Visualization & Interpretation

### Key Interpretations
- **Gestational age effect**: Per-week increase in Y concentration (with CI)
- **BMI effect**: Per-unit BMI decrease in Y concentration (with CI)
- **Clinical relevance**: Weeks at which 4% threshold is reached for different BMI levels

### Visualization Strategy
1. **Effect plots**: Predicted Y vs. weeks at BMI {22, 28, 35}
2. **BMI sensitivity**: Predicted Y vs. BMI at weeks {11, 15, 20}
3. **Threshold analysis**: 4% crossing points across BMI range
4. **Confidence intervals**: Show uncertainty in predictions

### Deliverables
- **Effect visualization** embedded in notebook (interactive matplotlib plots)
- **4% threshold analysis** as DataFrame in notebook output
- **Clinical interpretation** in notebook markdown cells
- **Optional**: Key figures saved for paper if explicitly needed

## 7. Validation & Robustness Checks

### Sensitivity Analysis
1. **Leave-one-out**: Confirm coefficient signs persist
2. **Cross-model validation**: OLS vs. splines vs. mixed-effects
3. **Subsample stability**: Bootstrap confidence intervals
4. **Outlier sensitivity**: Robust regression comparison

### Expected Consistent Findings
- **Weeks effect**: Positive (Y ↑ with gestational age)
- **BMI effect**: Negative (Y ↓ with higher BMI)
- **Significance**: Both effects statistically significant (p < 0.05)

### Deliverables
- **Robustness results** as DataFrame in notebook
- **Sensitivity analysis** documented in notebook markdown
- **Reproducibility**: notebook with clear cell execution order and random seeds
- **Final summary** in last notebook cell with key findings

---

## Implementation Strategy

### Primary Workflow
1. **Data Work**: All in `src/notebooks/01_data_exploration.ipynb`
   - Data loading, cleaning, EDA, visualization
   - Model exploration and diagnostics
   - Results interpretation and documentation

2. **Model Development**: Optional `.py` files in `src/analysis/problem1/`
   - Only if complex model classes or production deployment needed
   - Keep simple for academic analysis

3. **Minimal Outputs**:
   - Notebook with embedded plots and results
   - Optional: `output/results/p1_model_comparison.csv` for paper
   - Optional: `src/models/p1_final_model.pkl` for model persistence

---

## Current Progress Summary

- **Dataset**: 555 male fetus samples (10-25 weeks) ✅
- **Primary correlations**: Weeks r=0.184, BMI r=-0.138 (both p<0.001) ✅
- **Clinical relevance**: 87.0% samples above 4% reliability threshold ✅
- **Baseline OLS Model**: Implemented and validated ✅
  - Model: `Y_concentration ~ weeks + BMI`
  - F-test: 18.20 (p = 2.22e-08), R² = 0.062
  - Coefficients: weeks = 0.00184***, BMI = -0.00198***
  - Diagnostics: heteroscedasticity detected, residuals non-normal
  - Status: **Statistically significant but requires robust alternatives**
- **Next**: Steps 5-7 - Robust models, effect visualization, validation 📊

---

# Implementation Guide

## Environment & Structure
- **Environment**: `cumcm-env` conda environment
- **Primary file**: `src/notebooks/01_data_exploration.ipynb`
- **Dependencies**: pandas, numpy, statsmodels, matplotlib, seaborn, scipy
- **Coding rules**: See `.cursor/rules/prob1-implementation.mdc`

---

## Quick Implementation Checklist

✅ **Completed Steps** (based on actual analysis):
1. ✅ Load male fetus data from correct sheet (1,082 → 555 clean samples)
2. ✅ Parse gestational weeks ("11w+6" → 11.86 format)
3. ✅ EDA & correlations (weeks: r=0.184; BMI: r=-0.138, both p<0.001)
4. ✅ Baseline OLS model with significance testing, diagnostic checks, and model extensions

🔄 **Next Steps**:
5. 🔄 Robust alternatives implementation (Priority 1: HC3, Priority 2: Beta/Logistic)
6. 🔄 Effect visualization and 4% threshold analysis
7. 🔄 Validation & robustness checks and final results documentation

---

## Key Validation Points

✅ **Data Quality**: 
- Correct sheet loading (男胎检测数据)
- Proper week parsing (11w+6 format)
- Valid Y concentration range (0.01-0.23)

✅ **Statistical Requirements**:
- Significant correlations found (both p < 0.001)
- Expected sign patterns (weeks +, BMI -)
- Adequate sample size (1,068 samples)

✅ **Clinical Relevance**:
- 86.4% samples above 4% reliability threshold
- Gestational age window appropriate (10-25 weeks)
- BMI range realistic (20.7-46.9)

---

**Implementation Status**: Steps 1-4 completed ✅, Baseline OLS validated with diagnostics, Ready for robust alternatives 📊
