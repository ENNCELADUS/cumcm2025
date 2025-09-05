# Problem 2: Y-Chromosome Threshold Analysis - Implementation Guide

This guide provides step-by-step code implementation for Problem 2 survival analysis in Jupyter Notebook format.

## 📋 Overview

**Objective**: Analyze the relationship between maternal BMI and timing of Y-chromosome concentration reaching ≥4% threshold using interval-censored survival analysis.

**Key Approach**: Use Accelerated Failure Time (AFT) models with proper interval censoring to handle multiple tests per mother and determine optimal testing weeks.

---

## 🏗️ Notebook Structure

### Prerequisites
- Preprocessed data from `02_prob2_preprocessing.ipynb`
- Individual test records with: `maternal_id`, `gestational_weeks`, `bmi`, `y_concentration`

### Implementation Sections

---

## 📍 Section 1: Event Interval Construction (Per Mother)

**Goal**: Convert multiple test records per mother into single interval-censored event records.

### Step 1.1: Data Preparation
```python
# Load preprocessed individual test data
# Apply same QC as preprocessing to get df with individual tests
# Keep columns: maternal_id, gestational_weeks, bmi, y_concentration
```

### Step 1.2: Interval Construction Logic
For each `maternal_id`, determine:

**Event Types:**
1. **Left-censored**: First observation already ≥4%
   - `L = 0`, `R = first_week`, `censor_type = 'left'`

2. **Interval-censored**: Threshold crossed between visits  
   - `L = weeks[j*-1]`, `R = weeks[j*]`, `censor_type = 'interval'`
   - Where `j*` = first visit with `y_concentration ≥ 0.04`

3. **Right-censored**: Never reached threshold
   - `L = last_week`, `R = np.inf`, `censor_type = 'right'`

### Step 1.3: Implementation
```python
def construct_intervals(df_tests):
    """Convert individual tests to interval-censored format"""
    intervals = []
    
    for maternal_id, group in df_tests.groupby('maternal_id'):
        # Sort by gestational weeks
        group = group.sort_values('gestational_weeks')
        bmi = group['bmi'].iloc[0]  # Use first BMI value
        
        # Find threshold crossings
        threshold_mask = group['y_concentration'] >= 0.04
        
        if threshold_mask.iloc[0]:
            # Left-censored
            L, R = 0, group['gestational_weeks'].iloc[0]
            censor_type = 'left'
        elif threshold_mask.any():
            # Interval-censored
            cross_idx = threshold_mask.idxmax()
            cross_pos = group.index.get_loc(cross_idx)
            L = group['gestational_weeks'].iloc[cross_pos-1] if cross_pos > 0 else 0
            R = group['gestational_weeks'].iloc[cross_pos]
            censor_type = 'interval'
        else:
            # Right-censored
            L = group['gestational_weeks'].iloc[-1]
            R = np.inf
            censor_type = 'right'
            
        intervals.append({
            'maternal_id': maternal_id,
            'bmi': bmi,
            'L': L, 'R': R,
            'censor_type': censor_type
        })
    
    return pd.DataFrame(intervals)
```

### Step 1.4: Output Analysis
```python
# Generate df_intervals
# Print censoring type counts
# Display sample intervals
```

---

## 📍 Section 2: Feature Matrix Preparation

**Goal**: Prepare covariate matrix for AFT modeling.

### Step 2.1: Standardization
```python
# Create df_X from df_intervals
# Standardize BMI: bmi_z = (bmi - mean) / std
# Keep columns: L, R, bmi, bmi_z
```

### Step 2.2: Quality Checks
```python
# Print df_X.describe()
# Check for missing values
# Verify interval validity (L < R)
```

---

## 📍 Section 3: AFT Modeling & Inference (Core Analysis)

**Goal**: Fit Accelerated Failure Time models to interval-censored data.

### Step 3.1: Model Fitting
```python
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

# Primary model: Weibull AFT
weibull_aft = WeibullAFTFitter()
weibull_aft.fit_interval_censoring(
    df_X, 
    lower_bound_col='L', 
    upper_bound_col='R', 
    formula='~ bmi_z'
)

# Alternative: Log-logistic AFT for comparison
loglogistic_aft = LogLogisticAFTFitter()
loglogistic_aft.fit_interval_censoring(df_X, 'L', 'R', formula='~ bmi_z')
```

### Step 3.2: Model Summary & Diagnostics
```python
# Display model summary
print(weibull_aft.summary)

# Extract BMI coefficient significance
bmi_coef = weibull_aft.params_['bmi_z']
bmi_pvalue = weibull_aft.summary.loc['bmi_z', 'p']
```

### Step 3.3: Survival Function Prediction
```python
# Create BMI representative values (quartiles)
bmi_quartiles = df_intervals['bmi'].quantile([0.25, 0.5, 0.75])

# Generate survival curves for different BMI levels
time_grid = np.linspace(10, 25, 100)
survival_curves = {}

for q, bmi_val in bmi_quartiles.items():
    bmi_z_val = (bmi_val - df_intervals['bmi'].mean()) / df_intervals['bmi'].std()
    X_query = pd.DataFrame({'bmi_z': [bmi_z_val]})
    survival_curves[f'BMI_q{q}'] = weibull_aft.predict_survival_function(
        X_query, times=time_grid
    )
```

### Step 3.4: Optimal Testing Week Calculation
```python
def calculate_optimal_week(survival_func, confidence_level=0.90):
    """Calculate optimal testing week based on confidence level"""
    threshold = 1 - confidence_level
    
    for t, surv_prob in survival_func.items():
        if (1 - surv_prob.iloc[0]) >= threshold:
            return t
    return np.inf

# Calculate for different confidence levels
confidence_levels = [0.90, 0.95]
optimal_weeks = {}

for conf in confidence_levels:
    optimal_weeks[conf] = {
        curve_name: calculate_optimal_week(curve, conf)
        for curve_name, curve in survival_curves.items()
    }
```

---

## 📍 Section 4: Turnbull Non-parametric Validation

**Goal**: Validate AFT assumptions using non-parametric Turnbull estimator.

### Step 4.1: Turnbull Fitting
```python
from lifelines import IntervalCensoringFitter

# Fit Turnbull estimator
turnbull = IntervalCensoringFitter()
turnbull.fit_interval_censoring(df_X, 'L', 'R')
```

### Step 4.2: Model Comparison
```python
# Compare Turnbull vs AFT survival curves
# Plot overlapping curves
# Assess goodness of fit
```

---

## 📍 Section 5: BMI Grouping & Group-specific Optimal Weeks

**Goal**: Create BMI groups and calculate optimal testing weeks for each group.

### Route A: CART-based Grouping

#### Step 5A.1: Predicted Median Times
```python
from sklearn.tree import DecisionTreeRegressor

# Calculate individual predicted median survival times
median_times = []
for idx, row in df_intervals.iterrows():
    bmi_z = (row['bmi'] - df_intervals['bmi'].mean()) / df_intervals['bmi'].std()
    X_query = pd.DataFrame({'bmi_z': [bmi_z]})
    median_time = weibull_aft.predict_percentile(X_query, p=0.5).iloc[0]
    median_times.append(median_time)

df_intervals['predicted_median'] = median_times
```

#### Step 5A.2: CART Grouping
```python
# Fit decision tree for BMI grouping
tree = DecisionTreeRegressor(
    max_depth=3, 
    min_samples_leaf=50,
    random_state=42
)

X_tree = df_intervals[['bmi']].values
y_tree = df_intervals['predicted_median'].values
tree.fit(X_tree, y_tree)

# Extract BMI cutpoints
# Create BMI group labels
```

### Route B: Confidence-driven Grouping

#### Step 5B.1: Candidate Cutpoints
```python
# Clinical BMI categories or quantile-based
candidate_cuts = [
    [25, 30],           # Clinical: Normal/Overweight/Obese
    [23, 27, 32],       # Modified clinical
    df_intervals['bmi'].quantile([0.33, 0.67]).tolist()  # Tertiles
]
```

#### Step 5B.2: Optimization
```python
def evaluate_grouping(cuts, df, penalty_lambda=1.0):
    """Evaluate BMI grouping strategy"""
    # Create groups based on cuts
    # Calculate group-specific optimal weeks
    # Return risk score: sum(pi_g * w_g*) + lambda * K
    pass

# Select optimal grouping
best_cuts = min(candidate_cuts, key=lambda x: evaluate_grouping(x, df_intervals))
```

### Step 5.3: Group-specific Analysis
```python
# Apply final BMI grouping
# Calculate survival functions for each group
# Compute optimal weeks (tau=0.90, 0.95) for each group

group_results = []
for group_name, group_data in grouped_data:
    # Group survival function (average or representative BMI)
    # Optimal weeks calculation
    # Store results
    pass
```

---

## 📍 Section 6: Monte Carlo Measurement Error Testing

**Goal**: Assess robustness to Y-concentration measurement errors.

### Step 6.1: Error Model Setup
```python
# Measurement error: y_obs = y_true + epsilon
# epsilon ~ N(0, sigma^2)
sigma_error = 0.002  # 0.2% absolute concentration error
n_simulations = 300  # 300+ recommended for smooth CI and stable assessment
```

### Step 6.2: Monte Carlo Loop
```python
mc_results = []

for sim in range(n_simulations):
    # Add noise to y_concentration
    df_noisy = df_original.copy()
    noise = np.random.normal(0, sigma_error, len(df_noisy))
    df_noisy['y_concentration'] += noise
    
    # Reconstruct intervals with noisy data
    df_intervals_sim = construct_intervals(df_noisy)
    
    # Refit AFT model
    # Recalculate group optimal weeks
    # Store results
    pass
```

### Step 6.3: Robustness Analysis
```python
# Summarize MC results: mean, std, 95% CI for each group
# Create uncertainty plots
# Assess stability of recommendations
```

---

## 📍 Section 7: Baseline Two-step ML Comparison (Optional)

**Goal**: Compare against traditional machine learning approach.

### Step 7.1: Classification Component
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Binary outcome: ever reached 4%
# Features: BMI (+ other covariates)
# Metrics: AUC, PR-AUC, Calibration
```

### Step 7.2: Regression Component
```python
from sklearn.ensemble import RandomForestRegressor

# Continuous outcome: time to threshold (interval midpoint approximation)
# Features: BMI
# Metrics: MAE, RMSE
```

### Step 7.3: Comparison
```python
# Map ML predictions to group-level recommendations
# Compare optimal weeks: AFT vs ML baseline
```

---

## 📍 Section 8: Validation & Final Policy Table

**Goal**: Generate final recommendations with uncertainty quantification.

### Step 8.1: Cross-validation
```python
from sklearn.model_selection import KFold

# K-fold validation (K=5)
# Metrics: C-index, time-dependent Brier score
# Calibration assessment
```

### Step 8.2: Sensitivity Analysis
```python
# Test different confidence levels (tau = 0.85, 0.90, 0.95)
# Compare AFT distributions (Weibull vs Log-logistic)
# Toggle preprocessing filters (IQR on/off)
```

### Step 8.3: Final Policy Table
```python
# Create comprehensive results table
policy_table = pd.DataFrame({
    'BMI_Range': [...],
    'n_mothers': [...],
    'optimal_week_90': [...],
    'optimal_week_95': [...],
    'threshold_prob_at_optimal': [...],
    'mc_ci_low': [...],
    'mc_ci_high': [...]
})

# Display inline
# Optional export: prob2_policy_recommendations.csv
```

---

## 🎯 Success Criteria (Definition of Done)

- ✅ `df_intervals` constructed with proper censoring types
- ✅ AFT models fitted and validated against Turnbull
- ✅ BMI grouping with group-specific optimal weeks
- ✅ Monte Carlo robustness assessment completed
- ✅ Final policy table with confidence intervals
- ✅ Optional baseline ML comparison
- ✅ Sensitivity analysis across key parameters

---

## 📊 Expected Outputs

### Inline Displays
- Censoring type distribution
- AFT model summaries and coefficients
- Survival curve comparisons (Turnbull vs AFT)
- BMI group-specific survival curves
- Monte Carlo uncertainty plots
- Final policy recommendations table

### Optional Export
- `prob2_policy_recommendations.csv` (single file only)

---

## 🔧 Technical Implementation Notes

### Key Libraries
```python
# Core survival analysis
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, IntervalCensoringFitter

# Machine learning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Standard data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Performance Tips
- Start Monte Carlo with small sample size (B=50-100) for debugging
- Use `random_state` for reproducibility
- Consider parallel processing for Monte Carlo simulations
- Cache AFT model fits to avoid repeated computation

### Common Pitfalls
- Ensure proper interval validity (L < R for all observations)
- Handle edge cases in optimal week calculation (survival never drops below threshold)
- Account for numerical precision in threshold comparisons
- Validate BMI grouping creates balanced groups

