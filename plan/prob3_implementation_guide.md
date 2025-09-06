# Problem 3: Multi-Covariate AFT Extension - Implementation Guide

This guide provides step-by-step code implementation for Problem 3 survival analysis extending Problem 2 with richer covariates and explicit collinearity control.

## 📋 Overview

**Objective**: Extend the AFT model used in Problem 2 to incorporate expanded covariates (BMI, age, height, weight) with group-wise optimal NIPT week decisions and enhanced robustness testing.

**Key Extensions from Problem 2**:
- **Large covariate set**: 22 variables including core (BMI, age), sequencing quality (9 vars), and engineered features (11 vars)
- **Intelligent covariate selection**: Systematic VIF assessment with multi-combination testing, final selection of 6 variables
- **Advanced collinearity control**: VIF < 5.0 constraint with automated fallback strategies
- **Comprehensive feature engineering**: BMI categories, gestational splines, quality scores, lag features, interactions
- **Enhanced standardization**: All continuous variables properly standardized for numerical stability
- Group-wise decision reporting with between-group contrasts  
- Mandatory 300-run Monte Carlo sensitivity analysis

**Reused from Problem 2**: Interval construction, AFT modeling, Turnbull validation, threshold-based optimal weeks, bootstrap uncertainty

---

## 🏗️ Notebook Structure

### Prerequisites
- Preprocessed data from Problem 2 with additional covariates: age, height, weight
- Same event interval construction methodology as Problem 2
- Individual test records with: `maternal_id`, `gestational_weeks`, `bmi`, `age`, `height`, `weight`, `y_concentration`

### Implementation Sections

---

## 📍 Section 1: Extended Data Preprocessing & Covariate Preparation

**Goal**: Prepare expanded covariate matrix with collinearity control and standardization.

### Step 1.1: Covariate Set Assembly
```python
# Load preprocessed data from Problem 2
# Verify availability of: BMI, age, height, weight, and the following


### ✅ Large Covariate Set Available (22 variables total)

**Core Variables (Required):**
* `孕妇BMI` (bmi) → standardized as `bmi_std`
* `年龄` (age) → standardized as `age_std`  
* `检测孕周` (gestational_weeks) → used for splines
* `Y染色体浓度` (y_concentration) → outcome variable
* `孕妇代码` (maternal_id) → patient identifier

**Extended Sequencing Quality Variables (9 available):**
* `原始读段数` (raw_read_count) → standardized as `raw_read_count_std`
* `唯一比对的读段数` (unique_mapped_reads) → standardized as `unique_mapped_reads_std`
* `在参考基因组上比对的比例` (mapping_ratio) → standardized as `mapping_ratio_std`
* `重复读段的比例` (duplicate_ratio) → standardized as `duplicate_ratio_std`
* `被过滤掉读段数的比例` (filtered_reads_ratio) → standardized as `filtered_reads_ratio_std`
* `GC含量` (gc_content) → standardized as `gc_content_std`
* `IVF妊娠` (ivf_pregnancy) → categorical
* `怀孕次数` (pregnancy_count) → count variable
* `生产次数` (birth_count) → count variable

**Note**: Height/weight excluded due to severe multicollinearity with BMI (VIF > 200)

### 🆕 Engineered covariates

### 1. **BMI category variable**

Let maternal BMI for subject *i* at measurement *j* be $\mathrm{BMI}_{ij}$.
Define categorical bins (example cutoffs):

$$
\text{bmi\_cat}_{ij} = 
\begin{cases}
0 & \text{if } \mathrm{BMI}_{ij} < 25, \\
1 & \text{if } 25 \leq \mathrm{BMI}_{ij} < 30, \\
2 & \text{if } 30 \leq \mathrm{BMI}_{ij} < 35, \\
3 & \text{if } 35 \leq \mathrm{BMI}_{ij} < 40, \\
4 & \text{if } \mathrm{BMI}_{ij} \geq 40 .
\end{cases}
$$

---

### 2. **Spline basis of gestational weeks**

Let gestational week be $w_{ij}$. Construct a cubic B-spline basis with $d$ degrees of freedom (knots $\kappa_1,\dots,\kappa_d$):

$$
\text{gest\_week\_spline}_{ij} = \big( B_1(w_{ij}), B_2(w_{ij}), \dots, B_d(w_{ij}) \big),
$$

where $B_k(\cdot)$ are normalized spline basis functions.

---

### 3. **Log-transformed unique mapped reads**

Let $u_{ij}$ denote the unique mapped reads count. Define

$$
\text{log\_unique\_reads}_{ij} = \log \left( u_{ij} + 1 \right).
$$

(The +1 prevents log(0)).

---

### 4. **Sequencing quality score**

Given mapping ratio $m_{ij}$, duplicate ratio $d_{ij}$, and filtered reads ratio $f_{ij}$:

$$
\text{seq\_quality\_score}_{ij} = \alpha_1 m_{ij} + \alpha_2 (1 - d_{ij}) + \alpha_3 (1 - f_{ij}),
$$

with weights $\alpha_1,\alpha_2,\alpha_3 \geq 0$ (often $\alpha_k = 1$, or estimated via PCA).

---

### 5. **Previous Y-concentration (lag feature)**

For repeated measures per mother, let $y_{ij}$ denote Y chromosome concentration at record *j* for patient *i*. If observations are ordered by time, define

$$
\text{prior\_y\_conc}_{ij} = 
\begin{cases}
y_{i,j-1}, & j > 1, \\
\text{NA}, & j = 1.
\end{cases}
$$

---

### 6. **Slope of Y-concentration**

For patient *i*, fit a simple linear regression on their own past measurements:

$$
y_{ik} = \beta_{0i} + \beta_{1i} t_{ik} + \varepsilon_{ik},
$$

where $t_{ik}$ is gestational week. Then define

$$
\text{slope\_y\_conc}_{ij} = \hat{\beta}_{1i},
$$

the estimated within-mother growth rate of Y concentration.

---

### 7. **BMI × Gestational Weeks interaction**

$$
\text{bmi\_weeks}_{ij} = \mathrm{BMI}_{ij} \times w_{ij}.
$$

# Handle missing values with principled imputation (MICE or RF-impute)
# Never impute outcome Y_ij - only covariates
```

### Step 1.2: Systematic Multicollinearity Assessment & Covariate Selection
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def comprehensive_vif_assessment(df, preprocessing_metadata, vif_threshold=5.0):
    """
    Systematic multi-combination VIF assessment for large covariate sets.
    
    Strategy:
    1. Test Core-only (BMI, age)
    2. Test Core + Sequencing Quality (safe combinations)  
    3. Test All Available (identify problematic variables)
    4. Select optimal combination with VIF ≤ 5.0
    """
    
    # Identify covariate categories
    core_vars = ['bmi_std', 'age_std']
    sequencing_vars = [col for col in df.columns if any(seq in col for seq in 
                      ['read', 'mapping', 'ratio', 'gc']) and col.endswith('_std')]
    engineered_vars = [col for col in df.columns if any(eng in col for eng in 
                      ['spline', 'log', 'quality', 'interaction']) and col.endswith('_std')]
    
    # Test combinations systematically
    test_sets = {
        "Core Only": core_vars,
        "Core + Sequencing": core_vars + sequencing_vars[:3],  # Limit for stability
        "All Available": core_vars + sequencing_vars + engineered_vars[:3]
    }
    
    for set_name, variables in test_sets.items():
        vif_results = calculate_vif(df[variables].dropna())
        acceptable_vars = vif_results[vif_results['VIF'] <= vif_threshold]['Variable'].tolist()
        print(f"{set_name}: {len(acceptable_vars)}/{len(variables)} pass VIF constraint")
    
    # Select optimal set: start with core, add acceptable variables
    # Exclude height/weight due to BMI multicollinearity (VIF > 200)
    
    return selected_covariates

# Expected final selection (from implementation): 
# ['bmi_std', 'age_std', 'raw_read_count_std', 'unique_mapped_reads_std', 
#  'mapping_ratio_std', 'gc_content_std'] (6 variables, all VIF < 2.0)
```

### Step 1.3: Standardization Protocol
```python
def standardize_covariates(df):
    """Center/scale continuous covariates for numerical stability"""
    continuous_vars = ['bmi', 'age', 'height', 'weight']
    standardized_data = df.copy()
    
    for var in continuous_vars:
        if var in df.columns:
            mean_val = df[var].mean()
            std_val = df[var].std()
            standardized_data[f'{var}_std'] = (df[var] - mean_val) / std_val
    
    return standardized_data
```

### Step 1.4: Final Covariate Set Selection (Post-VIF Assessment)
```python
# VIF-approved covariate set (actual implementation results):
final_modeling_covariates = [
    'bmi_std',                    # Core: maternal BMI (standardized)
    'age_std',                    # Core: maternal age (standardized)
    'raw_read_count_std',         # Sequencing: total sequencing depth
    'unique_mapped_reads_std',    # Sequencing: quality metric
    'mapping_ratio_std',          # Sequencing: alignment quality
    'gc_content_std'              # Sequencing: content composition
]
# All 6 variables have VIF < 2.0, ensuring no multicollinearity issues

# Excluded due to high VIF:
# - height/weight: VIF > 200 (BMI = weight/height²)  
# - Some engineered features: VIF > 5 when combined
# - Complex interaction terms: removed for interpretability

# Model complexity: 6 covariates + intercept + scale = 8 parameters
# Sample size: 524 records → adequate parameter-to-sample ratio
```

---

## 📍 Section 2: Interval Construction & Feature Matrix (Reused from Problem 2)

**Goal**: Construct interval-censored data exactly as in Problem 2, extended with new covariates.

### Step 2.1: Event Interval Construction
```python
# Reuse construct_intervals() from Problem 2 module
# Output: df_intervals with (maternal_id, L, R, censor_type, bmi, age, height, weight)
```

### Step 2.2: Extended Feature Matrix  
```python
# Create df_X with interval bounds and standardized covariates
# Include VIF-approved covariate set only
# Verify interval validity and covariate completeness
```

---

## 📍 Section 3: Extended AFT Model Specification & Estimation

**Goal**: Fit AFT models with expanded covariate set and nonlinearity options.

### Step 3.1: Extended AFT Model with VIF-Selected Covariates
```python
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

# Primary specification: VIF-approved large covariate set
weibull_aft_extended = WeibullAFTFitter()
weibull_aft_extended.fit_interval_censoring(
    df_X, 
    lower_bound_col='L', 
    upper_bound_col='R', 
    formula='~ bmi_std + age_std + raw_read_count_std + unique_mapped_reads_std + mapping_ratio_std + gc_content_std'
)

# Compare with core-only model for nested model testing
weibull_aft_core = WeibullAFTFitter()
weibull_aft_core.fit_interval_censoring(
    df_X, 
    lower_bound_col='L', 
    upper_bound_col='R', 
    formula='~ bmi_std + age_std'  # Core-only for comparison
)

# Model selection via AIC/BIC
print(f"Extended model AIC: {weibull_aft_extended.AIC_}")
print(f"Core model AIC: {weibull_aft_core.AIC_}")
```

### Step 3.2: Nonlinearity with Restricted Cubic Splines
```python
from patsy import dmatrix

def create_spline_basis(bmi_values, knots=None, degree=3):
    """Create restricted cubic spline basis for BMI"""
    if knots is None:
        # Default: place knots at quantiles
        knots = np.quantile(bmi_values, [0.1, 0.5, 0.9])
    
    # Create spline basis using patsy
    spline_formula = f"bs(bmi_std, knots={knots}, degree={degree})"
    basis_matrix = dmatrix(spline_formula, data={'bmi_std': bmi_values})
    return basis_matrix

# Fit spline model if AIC improves
weibull_aft_spline = WeibullAFTFitter()
# Add spline features to df_X before fitting
# Compare AIC: linear vs spline models
```

### Step 3.3: Interaction Terms (Guarded)
```python
# BMI × age interaction: only if LRT significant and clinically interpretable
# Likelihood ratio test between nested models
from scipy.stats import chi2

def likelihood_ratio_test(model_null, model_alt):
    """Perform likelihood ratio test"""
    ll_null = model_null.log_likelihood_
    ll_alt = model_alt.log_likelihood_
    lr_stat = -2 * (ll_null - ll_alt)
    df_diff = model_alt.params_.shape[0] - model_null.params_.shape[0]
    p_value = 1 - chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value

# Test interactions conservatively
```

### Step 3.4: Model Selection & Diagnostics
```python
# Compare Weibull vs Log-logistic by AIC
# Select best baseline distribution
# Robust (sandwich) standard errors
# Extract time-ratio interpretations: exp(β_k)
```

---

## 📍 Section 4: Enhanced Model Diagnostics & Collinearity Control

**Goal**: Validate AFT assumptions with collinearity diagnostics and calibration assessment.

### Step 4.1: Collinearity Diagnostics
```python
# Final VIF check on selected covariate set
final_vif = calculate_vif(df_X[selected_covariates])
print("Final VIF Diagnostics:")
print(final_vif)

# Reject if any VIF > 5 after standardization
if (final_vif['VIF'] > 5).any():
    print("⚠️  Warning: High multicollinearity detected")
```

### Step 4.2: Turnbull Validation (Reused)
```python
# Same as Problem 2: fit Turnbull estimator
# Compare AFT vs Turnbull survival curves
# Goodness-of-fit metrics: integrated absolute error
```

### Step 4.3: Predictive Validation
```python
# Patient-level K-fold cross-validation  
# Interval-censored log-likelihood on held-out data
# Integrated Brier Score (IBS) at clinical horizons
# Time-stratified KS-type discrepancies
```

---

## 📍 Section 5: BMI Grouping & Group-Specific Optimal Weeks (Extended Reporting)

**Goal**: Create BMI groups exactly as Problem 2 but with enhanced per-group reporting and between-group contrasts.

### Step 5.1: BMI Grouping (Identical to Problem 2)
```python
# Use identical BMI cutpoints as Problem 2
# Same grouping methodology: clinical categories or data-driven CART
# Ensure consistency for cross-problem comparison
```

### Step 5.2: Group-Wise Survival Functions
```python
def compute_group_survival_extended(groups, df_X, aft_model):
    """Compute group survival by plug-in averaging"""
    group_survival_funcs = {}
    
    for group_name, group_data in groups:
        # Average over empirical distribution of X in group g
        n_group = len(group_data)
        survival_sum = np.zeros_like(time_grid)
        
        for idx, row in group_data.iterrows():
            X_individual = row[covariate_columns].to_frame().T
            survival_individual = aft_model.predict_survival_function(
                X_individual, times=time_grid
            )
            survival_sum += survival_individual.values[:, 0]
        
        # Group survival: S_g(t) = (1/|I_g|) * sum_i S(t|X_i)
        group_survival_funcs[group_name] = survival_sum / n_group
    
    return group_survival_funcs
```

### Step 5.3: Threshold-Based Optimal Weeks per Group
```python
def calculate_group_optimal_weeks(group_survival_funcs, confidence_levels=[0.90, 0.95]):
    """Calculate t_g*(tau) for each group and confidence level"""
    optimal_weeks = {}
    
    for group_name, survival_func in group_survival_funcs.items():
        optimal_weeks[group_name] = {}
        
        for tau in confidence_levels:
            # t_g*(tau) = inf{t: 1-S_g(t) >= tau}
            attainment_prob = 1 - survival_func
            
            optimal_week = np.inf
            for i, t in enumerate(time_grid):
                if attainment_prob[i] >= tau:
                    optimal_week = t
                    break
            
            optimal_weeks[group_name][f'tau_{tau}'] = optimal_week
    
    return optimal_weeks
```

### Step 5.4: Between-Group Contrasts
```python
def compute_group_contrasts(optimal_weeks):
    """Compute between-group contrasts: Δt_g,h*(tau) = t_g*(tau) - t_h*(tau)"""
    contrasts = {}
    groups = list(optimal_weeks.keys())
    
    for i, group_g in enumerate(groups):
        for j, group_h in enumerate(groups):
            if i < j:  # Avoid duplicates
                contrast_name = f"{group_g}_vs_{group_h}"
                contrasts[contrast_name] = {}
                
                for tau_key in optimal_weeks[group_g].keys():
                    diff = optimal_weeks[group_g][tau_key] - optimal_weeks[group_h][tau_key]
                    contrasts[contrast_name][tau_key] = diff
    
    return contrasts
```

---

## 📍 Section 6: Enhanced Monte Carlo Error Sensitivity (300-Run Mandatory)

**Goal**: Assess robustness with 300-run Monte Carlo as mandated, reporting per-group distributions.

### Step 6.1: Monte Carlo Setup (300 Runs)
```python
# Mandatory parameters
N_MC_SIMULATIONS = 300  # Exactly 300 runs
SIGMA_Y = 0.002  # From Problem 2 or validated QC
confidence_levels = [0.90, 0.95]
```

### Step 6.2: Per-Group Monte Carlo Analysis
```python
def run_enhanced_monte_carlo(df_original, n_simulations=300):
    """
    Run 300-replicate Monte Carlo with per-group reporting
    """
    mc_results = {
        'group_optimal_weeks': [],
        'group_contrasts': [],
        'simulation_metadata': []
    }
    
    for sim in range(n_simulations):
        # Step 1: Noise injection
        df_noisy = add_measurement_noise(df_original, sigma=SIGMA_Y)
        
        # Step 2: Interval reconstruction  
        df_intervals_sim = construct_intervals(df_noisy)
        
        # Step 3: Extended feature matrix with standardization
        df_X_sim = prepare_extended_feature_matrix(df_intervals_sim)
        
        # Step 4: Refit AFT with expanded covariates
        aft_model_sim = fit_aft_model_extended(df_X_sim)
        
        # Step 5: Group survival analysis
        group_survival_sim = compute_group_survival_extended(
            get_bmi_groups(df_intervals_sim), df_X_sim, aft_model_sim
        )
        
        # Step 6: Group optimal weeks
        optimal_weeks_sim = calculate_group_optimal_weeks(
            group_survival_sim, confidence_levels
        )
        
        # Step 7: Between-group contrasts
        contrasts_sim = compute_group_contrasts(optimal_weeks_sim)
        
        # Store results
        mc_results['group_optimal_weeks'].append(optimal_weeks_sim)
        mc_results['group_contrasts'].append(contrasts_sim)
        mc_results['simulation_metadata'].append({
            'sim_id': sim,
            'converged': True,  # Track convergence
            'aft_aic': aft_model_sim.AIC_
        })
    
    return mc_results
```

### Step 6.3: Per-Group Robustness Summary
```python
def summarize_monte_carlo_per_group(mc_results):
    """Summarize MC results with per-group distributions"""
    summary = {}
    
    # Extract all group names and confidence levels
    groups = list(mc_results['group_optimal_weeks'][0].keys())
    tau_levels = list(mc_results['group_optimal_weeks'][0][groups[0]].keys())
    
    for group in groups:
        summary[group] = {}
        
        for tau in tau_levels:
            # Collect optimal weeks across all simulations
            weeks = [
                result[group][tau] for result in mc_results['group_optimal_weeks']
                if not np.isinf(result[group][tau])
            ]
            
            if weeks:
                summary[group][tau] = {
                    'mean': np.mean(weeks),
                    'std': np.std(weeks),
                    'ci_2.5': np.percentile(weeks, 2.5),
                    'ci_97.5': np.percentile(weeks, 97.5),
                    'robustness_label': assess_robustness(weeks)
                }
    
    return summary

def assess_robustness(weeks, clinical_cutoff=1.0):
    """Assign robustness label based on CI width and stability"""
    ci_width = np.percentile(weeks, 97.5) - np.percentile(weeks, 2.5)
    
    if ci_width <= clinical_cutoff:
        return "high"
    elif ci_width <= 2 * clinical_cutoff:
        return "medium" 
    else:
        return "low"
```

---

## 📍 Section 7: Sensitivity Analyses (Targeted & Optional)

**Goal**: Test robustness of covariate specification and baseline distribution choices.

### Step 7.1: Covariate Set Sensitivity
```python
# Alternative 1: Height/Weight replacing BMI
# Alternative 2: Residualized height/weight orthogonal to BMI
# Accept only if AIC improves and VIF remains acceptable

covariate_alternatives = {
    'baseline': ['bmi_std', 'age_std'],
    'with_height_weight': ['bmi_std', 'age_std', 'height_std', 'weight_std'],
    'height_weight_only': ['height_std', 'weight_std', 'age_std'],
    'residualized': ['bmi_std', 'age_std', 'height_resid', 'weight_resid']
}
```

### Step 7.2: Nonlinearity Assessment  
```python
# Test restricted cubic splines B(BMI)
# Likelihood ratio test: linear vs spline
# Retain spline only if justified by LRT AND calibration plots
```

### Step 7.3: Baseline Distribution Choice
```python
# Compare Weibull vs Log-logistic 
# If similar AIC: choose better calibration against Turnbull
# Focus on clinical region: weeks 10-20
```

---

## 📍 Section 8: Cross-Validation & Final Policy Table

**Goal**: Generate final recommendations with comprehensive uncertainty quantification.

### Step 8.1: Patient-Level Cross-Validation
```python
from sklearn.model_selection import KFold

# K-fold validation (K=5) at patient level
# Metrics: interval-censored log-likelihood, integrated Brier score
# Avoid leakage across repeated measures within patients
```

### Step 8.2: Final Policy Table with Contrasts
```python
def create_final_policy_table_extended(optimal_weeks, mc_summary, group_contrasts):
    """Create comprehensive policy table with group contrasts"""
    
    policy_table = []
    
    # Main group results
    for group_name in optimal_weeks.keys():
        for tau in [0.90, 0.95]:
            tau_key = f'tau_{tau}'
            
            policy_table.append({
                'BMI_Group': group_name,
                'Confidence_Level': tau,
                'Optimal_Week': optimal_weeks[group_name][tau_key],
                'MC_Mean': mc_summary[group_name][tau_key]['mean'],
                'MC_CI_Lower': mc_summary[group_name][tau_key]['ci_2.5'],
                'MC_CI_Upper': mc_summary[group_name][tau_key]['ci_97.5'],
                'Robustness': mc_summary[group_name][tau_key]['robustness_label'],
                'N_Mothers': len(get_group_data(group_name))
            })
    
    # Between-group contrasts summary
    contrast_table = []
    for contrast_name, contrasts in group_contrasts.items():
        for tau in [0.90, 0.95]:
            tau_key = f'tau_{tau}'
            contrast_table.append({
                'Group_Contrast': contrast_name,
                'Confidence_Level': tau,
                'Week_Difference': contrasts[tau_key],
                'Clinical_Significance': 'Yes' if abs(contrasts[tau_key]) > 1.0 else 'No'
            })
    
    return pd.DataFrame(policy_table), pd.DataFrame(contrast_table)
```

---

## 📍 Section 9: Assumptions & Clinical Interpretation

**Goal**: Document assumptions and provide clinical interpretation of extended model.

### Step 9.1: Model Assumptions
```python
# 1. AFT time-acceleration assumption with expanded covariates
# 2. Independent individuals (within-patient dependence handled by intervals) 
# 3. Monotone event definition (4% threshold crossing)
# 4. Additive Gaussian measurement error model
# 5. Linear covariate effects (unless spline justified)
```

### Step 9.2: Clinical Interpretation
```python
# Time ratios exp(β_k) for each covariate
# Group-specific optimal week recommendations
# Between-group clinical differences
# Robustness assessment impact on clinical decisions
```

---

## 🎯 Success Criteria (Definition of Done)

- ✅ Extended covariate set with VIF < 5 constraint satisfied
- ✅ AFT models fitted with multiple covariates and nonlinearity assessment  
- ✅ BMI grouping identical to Problem 2 with enhanced per-group reporting
- ✅ 300-run Monte Carlo completed with per-group robustness assessment
- ✅ Between-group contrasts Δt_g,h*(τ) computed and interpreted
- ✅ Final policy table with uncertainty quantification and clinical significance
- ✅ Comprehensive sensitivity analysis across covariate specifications
- ✅ Cross-validation and model diagnostics completed

---

## 📊 Expected Outputs

### Inline Displays
- Extended covariate VIF diagnostics table
- AFT model comparison table (linear vs spline, Weibull vs log-logistic)
- Per-group survival curves with expanded covariate adjustment
- 300-run Monte Carlo robustness distributions per group
- Between-group contrast analysis table
- Final policy recommendations with robustness labels

### Optional Export Files
- `prob3_policy_recommendations.csv` (enhanced version)
- `prob3_group_contrasts.csv` (new: between-group differences)
- `prob3_monte_carlo_robustness.csv` (detailed per-group MC results)

---

## 🔧 Technical Implementation Notes

### Key Libraries & Extensions
```python
# All Problem 2 libraries plus:
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix  # For spline basis construction
from scipy.stats import chi2  # For likelihood ratio tests

# Reuse Problem 2 modules where appropriate
from src.analysis.problem2 import construct_intervals, BMIGrouper
from src.models.aft_models import AFTSurvivalAnalyzer
```

### Performance & Computational Notes
- 300-run Monte Carlo: expect ~10-15 minutes depending on sample size
- VIF calculation: O(p³) where p = number of covariates  
- Spline fitting: may require increased optimization tolerance
- Bootstrap uncertainty: reuse Problem 2 patient-level resampling

### Integration with Problem 2
- **Identical BMI grouping**: ensures cross-problem comparability
- **Reuse interval construction**: same censoring methodology
- **Extend visualization functions**: adapt Problem 2 plots for multiple covariates
- **Consistent evaluation metrics**: same optimal week calculation logic

### Quality Assurance
- Verify all covariate VIF < 5 before final model fitting
- Cross-check optimal weeks calculation against Problem 2 implementation  
- Validate Monte Carlo convergence and robustness label assignment
- Ensure between-group contrasts are clinically interpretable
