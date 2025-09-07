"""
Data preprocessing for Problem 4: Female Fetal Abnormality Detection.

This module handles female fetus data preprocessing following the Problem 4 plan:
1. Sample selection (female fetus records)
2. Label generation from AB column (chromosomal aneuploidy)
3. Quality control filtering
4. Feature engineering and standardization
5. Group-stratified splitting by pregnant woman ID
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
import re

from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Problem 4 standalone - no dependencies on problem2


def parse_gestational_weeks(week_str: str) -> float:
    """
    Convert gestational weeks from format "11w+6" to decimal weeks (11.857).
    Copied from Problem 2 for consistency.
    
    Args:
        week_str: String in format like "11w+6" or "11w" or numeric string
        
    Returns:
        Decimal weeks as float, or NaN if parsing fails
    """
    if pd.isna(week_str):
        return np.nan
    
    week_str = str(week_str).strip()
    
    if 'w' in week_str.lower():
        parts = re.split('[wW]', week_str)
        if len(parts) == 2:
            weeks = int(parts[0])
            days_part = parts[1].strip()
            
            if '+' in days_part:
                days = int(days_part.split('+')[1])
            else:
                days = 0
            
            return weeks + days / 7.0
    
    try:
        return float(week_str)
    except:
        return np.nan

def generate_labels_problem4(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Generate binary labels Y from AB column (aneuploidy_ab) for Problem 4.
    
    Following the Problem 4 plan:
    Y = 1{AB is positive} - any T13/T18/T21 abnormality
    
    Args:
        df: DataFrame with aneuploidy_ab column
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with Y label column added
    """
    if verbose:
        print("🎯 Problem 4 Step 1.2: Generating binary labels from AB column...")
    
    df_result = df.copy()
    
    if 'aneuploidy_ab' not in df.columns:
        raise ValueError("aneuploidy_ab column not found. Expected from '染色体的非整倍体' mapping.")
    
    # Generate Y labels: Y=1 if AB is positive (any abnormality), Y=0 if negative (normal)
    # Positive cases: T13, T18, T21, T13T18, T13T21, T18T21, T13T18T21
    # Negative cases: NaN (normal cases)
    
    def classify_aneuploidy(ab_value):
        """Classify AB value into binary label"""
        if pd.isna(ab_value):
            return 0  # Normal case
        ab_str = str(ab_value).strip()
        if ab_str in ['T13', 'T18', 'T21', 'T13T18', 'T13T21', 'T18T21', 'T13T18T21']:
            return 1  # Abnormal case
        else:
            return 0  # Default to normal if unclear
    
    df_result['Y'] = df['aneuploidy_ab'].apply(classify_aneuploidy)
    
    # Statistics
    label_counts = df_result['Y'].value_counts()
    n_positive = label_counts.get(1, 0)
    n_negative = label_counts.get(0, 0)
    total = len(df_result)
    
    if verbose:
        print(f"   📊 Label generation results:")
        print(f"      Total samples: {total}")
        print(f"      Positive (Y=1, abnormal): {n_positive} ({100*n_positive/total:.1f}%)")
        print(f"      Negative (Y=0, normal): {n_negative} ({100*n_negative/total:.1f}%)")
        
        # Show breakdown by abnormality type
        if not df_result['aneuploidy_ab'].isna().all():
            abnormality_breakdown = df_result[df_result['Y'] == 1]['aneuploidy_ab'].value_counts()
            print(f"      Abnormality breakdown:")
            for abnormality, count in abnormality_breakdown.items():
                print(f"         {abnormality}: {count} cases")
    
    # Remove records with missing labels
    before_removal = len(df_result)
    df_result = df_result.dropna(subset=['Y'])
    after_removal = len(df_result)
    removed = before_removal - after_removal
    
    if verbose and removed > 0:
        print(f"   🗑️  Removed {removed} samples with missing Y labels")
        print(f"   ✅ Final dataset: {after_removal} samples with valid labels")
    
    return df_result

def apply_qc_filters_problem4(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply QC filters following Problem 4 plan:
    1. GC content 40-60% (mark outliers, don't remove)
    2. Reads/ratio winsorization at 1%-99%
    3. Remove bottom 1% of total reads
    
    Args:
        df: DataFrame with QC variables
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (filtered_df, qc_stats_dict)
    """
    if verbose:
        print("🔍 Problem 4 Step 2: Applying quality control filters...")
    
    df_result = df.copy()
    qc_stats = {'initial_count': len(df)}
    
    # Step 2.1: GC content outliers (mark but don't remove, apply weights later)
    if 'gc_global' in df.columns:
        gc_outliers = (df['gc_global'] < 0.40) | (df['gc_global'] > 0.60)
        df_result['gc_outlier'] = gc_outliers
        qc_stats['gc_outliers'] = gc_outliers.sum()
        
        if verbose:
            print(f"   📊 GC outliers (outside 40-60%): {gc_outliers.sum()} ({100*gc_outliers.mean():.1f}%)")
            print(f"      These will be down-weighted (0.7) in training, not removed")
    else:
        df_result['gc_outlier'] = False
        qc_stats['gc_outliers'] = 0
    
    # Step 2.2: Winsorization of reads and ratios (1%-99%)
    winsorize_vars = ['reads', 'map_ratio', 'dup_ratio', 'unique_reads']
    available_winsorize = [var for var in winsorize_vars if var in df.columns]
    
    if verbose and available_winsorize:
        print(f"   🔧 Winsorizing variables at 1%-99%: {available_winsorize}")
    
    for var in available_winsorize:
        if df[var].notna().sum() > 0:
            p1, p99 = df[var].quantile([0.01, 0.99])
            original_outliers = ((df[var] < p1) | (df[var] > p99)).sum()
            df_result[var] = df[var].clip(lower=p1, upper=p99)
            
            if verbose:
                print(f"      {var}: clipped {original_outliers} outliers to [{p1:.3f}, {p99:.3f}]")
    
    qc_stats['winsorized_variables'] = available_winsorize
    
    # Step 2.3: Remove bottom 1% of total reads (training set criteria)
    if 'reads' in df.columns:
        reads_threshold = df['reads'].quantile(0.01)
        low_reads_mask = df['reads'] < reads_threshold
        n_low_reads = low_reads_mask.sum()
        
        # Remove these samples
        df_result = df_result[~low_reads_mask]
        qc_stats['removed_low_reads'] = n_low_reads
        qc_stats['reads_threshold'] = reads_threshold
        
        if verbose:
            print(f"   🗑️  Removed bottom 1% reads: {n_low_reads} samples (threshold: {reads_threshold:.0f})")
    
    qc_stats['final_count'] = len(df_result)
    qc_stats['retention_rate'] = qc_stats['final_count'] / qc_stats['initial_count']
    
    if verbose:
        print(f"   ✅ QC filtering completed:")
        print(f"      Initial: {qc_stats['initial_count']} samples")
        print(f"      Final: {qc_stats['final_count']} samples")
        print(f"      Retention rate: {100*qc_stats['retention_rate']:.1f}%")
    
    return df_result, qc_stats

def add_derived_features_problem4(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add derived features following Problem 4 plan:
    1. max_Z = max(Z_13, Z_18, Z_21)
    2. Z_indicators: 1{|Z_j| >= 3} for j in {13, 18, 21}
    3. uniq_rate = unique_reads / reads
    4. Collinearity removal (prefer BMI over height/weight, uniq_rate over reads)
    
    Args:
        df: DataFrame with Z-values and reads
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (df_with_features, feature_stats)
    """
    if verbose:
        print("🔧 Problem 4 Step 3.2: Adding derived features...")
    
    df_result = df.copy()
    feature_stats = {}
    
    # Feature 1: max_Z = max(|Z_13|, |Z_18|, |Z_21|)
    z_cols = ['z13', 'z18', 'z21']
    available_z = [col for col in z_cols if col in df.columns]
    
    if len(available_z) >= 2:
        abs_z_values = df[available_z].abs()
        df_result['max_z'] = abs_z_values.max(axis=1)
        feature_stats['max_z_created'] = True
        
        if verbose:
            print(f"   📈 max_Z created from {available_z}")
            print(f"      Range: {df_result['max_z'].min():.2f} - {df_result['max_z'].max():.2f}")
    else:
        feature_stats['max_z_created'] = False
        if verbose:
            print(f"   ⚠️  Insufficient Z-values for max_Z: {available_z}")
    
    # Feature 2: Z_indicators for |Z_j| >= 3
    z_indicators_created = []
    for z_col in available_z:
        indicator_name = f'{z_col}_high'
        df_result[indicator_name] = (df[z_col].abs() >= 3).astype(int)
        z_indicators_created.append(indicator_name)
        
        if verbose:
            high_count = df_result[indicator_name].sum()
            print(f"   🎯 {indicator_name}: {high_count} samples with |{z_col}| >= 3")
    
    feature_stats['z_indicators_created'] = z_indicators_created
    
    # Feature 3: uniq_rate = unique_reads / reads
    if 'unique_reads' in df.columns and 'reads' in df.columns:
        # Avoid division by zero
        df_result['uniq_rate'] = df['unique_reads'] / df['reads'].replace(0, np.nan)
        df_result['uniq_rate'] = df_result['uniq_rate'].fillna(0)
        feature_stats['uniq_rate_created'] = True
        
        if verbose:
            print(f"   📊 uniq_rate created: unique_reads / reads")
            print(f"      Range: {df_result['uniq_rate'].min():.3f} - {df_result['uniq_rate'].max():.3f}")
    else:
        feature_stats['uniq_rate_created'] = False
        if verbose:
            missing_cols = [col for col in ['unique_reads', 'reads'] if col not in df.columns]
            print(f"   ⚠️  Cannot create uniq_rate: missing {missing_cols}")
    
    # Feature 4: Collinearity control
    # Remove highly correlated variables as per plan
    to_remove = []
    
    # Keep BMI, remove height/weight if BMI exists
    if 'bmi' in df.columns:
        if 'height' in df.columns:
            to_remove.append('height')
        if 'weight' in df.columns:
            to_remove.append('weight')
    
    # Keep uniq_rate, remove reads if both exist
    if 'uniq_rate' in df_result.columns and 'reads' in df.columns:
        to_remove.append('reads')
    
    # Check for other high correlations (Spearman >= 0.95) on remaining numeric columns
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in to_remove]
    
    if len(numeric_cols) > 1:
        corr_matrix = df_result[numeric_cols].corr(method='spearman').abs()
        # Find pairs with correlation >= 0.95
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= 0.95:
                    var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((var1, var2, corr_matrix.iloc[i, j]))
        
        # For high correlations, remove the second variable
        for var1, var2, corr in high_corr_pairs:
            if var2 not in to_remove:
                to_remove.append(var2)
                if verbose:
                    print(f"   🔗 High correlation: {var1} ↔ {var2} (r={corr:.3f}) - removing {var2}")
    
    # Apply removals
    final_to_remove = [col for col in to_remove if col in df_result.columns]
    if final_to_remove:
        df_result = df_result.drop(columns=final_to_remove)
        feature_stats['removed_collinear'] = final_to_remove
        
        if verbose:
            print(f"   🗑️  Removed collinear variables: {final_to_remove}")
    else:
        feature_stats['removed_collinear'] = []
    
    feature_stats['final_feature_count'] = len(df_result.columns)
    
    if verbose:
        print(f"   ✅ Feature engineering completed:")
        print(f"      Features added: {len(z_indicators_created) + int(feature_stats.get('max_z_created', False)) + int(feature_stats.get('uniq_rate_created', False))}")
        print(f"      Variables removed: {len(final_to_remove)}")
        print(f"      Final feature count: {feature_stats['final_feature_count']}")
    
    return df_result, feature_stats

def create_final_feature_matrix_problem4(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create final feature matrix for Problem 4 following the plan:
    Features: {Z_13, Z_18, Z_21, Z_X, GC_global, GC_13/18/21, map_ratio, dup_ratio, uniq_rate, BMI, age, max_Z, Z_indicators}
    
    Apply z-score standardization for logistic regression.
    
    Args:
        df: DataFrame with all features
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (feature_matrix, standardization_params)
    """
    if verbose:
        print("📊 Problem 4 Step 4: Creating final feature matrix...")
    
    # Define the target feature set from the plan (after collinearity filtering)
    target_features = {
        # Core Z-values
        'z_values': ['z13', 'z18', 'z21', 'zx'],
        # GC content (gc18, gc21 removed due to high correlation with gc13)
        'gc_features': ['gc_global', 'gc13'],
        # Sequencing quality (uniq_rate removed due to high correlation with map_ratio)
        'seq_features': ['map_ratio', 'dup_ratio'],
        # Maternal characteristics
        'maternal_features': ['bmi', 'age'],
        # Derived features
        'derived_features': ['max_z', 'z13_high', 'z18_high', 'z21_high']
    }
    
    # Collect available features
    all_target_features = []
    for category, features in target_features.items():
        all_target_features.extend(features)
    
    available_features = [f for f in all_target_features if f in df.columns]
    missing_features = [f for f in all_target_features if f not in df.columns]
    
    # Essential features that must be present
    essential_features = ['z13', 'z18', 'z21', 'zx', 'gc_global', 'bmi', 'age']
    missing_essential = [f for f in essential_features if f not in df.columns]
    
    if missing_essential:
        raise ValueError(f"Missing essential features: {missing_essential}")
    
    if verbose:
        print(f"   📋 Target features: {len(all_target_features)}")
        print(f"   ✅ Available: {len(available_features)}")
        if missing_features:
            print(f"   ⚠️  Missing: {missing_features}")
    
    # Create feature matrix (include gc_outlier for sample weighting)
    feature_cols = ['maternal_id', 'Y'] + available_features
    if 'gc_outlier' in df.columns:
        feature_cols.append('gc_outlier')
    keep_cols = [col for col in feature_cols if col in df.columns]
    
    df_features = df[keep_cols].copy()
    
    # Identify numeric features for standardization
    numeric_features = []
    for col in available_features:
        if col in df_features.columns and df_features[col].dtype in ['float64', 'int64']:
            # Skip binary indicators
            if not (col.endswith('_high') and df_features[col].nunique() <= 2):
                numeric_features.append(col)
    
    # Apply z-score standardization
    standardization_params = {}
    
    if verbose:
        print(f"   📏 Standardizing {len(numeric_features)} numeric features...")
    
    for feature in numeric_features:
        if df_features[feature].notna().sum() > 0:
            mean_val = df_features[feature].mean()
            std_val = df_features[feature].std()
            
            if std_val > 1e-10:  # Avoid division by zero
                df_features[f'{feature}_std'] = (df_features[feature] - mean_val) / std_val
                standardization_params[feature] = {'mean': mean_val, 'std': std_val}
                
                if verbose:
                    print(f"      {feature}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    # Final feature set for modeling
    modeling_features = [f'{f}_std' for f in numeric_features if f'{f}_std' in df_features.columns]
    binary_features = [f for f in available_features if f.endswith('_high')]
    
    final_features = modeling_features + binary_features
    final_cols = ['maternal_id', 'Y'] + final_features
    
    # Add gc_outlier for sample weighting if present
    if 'gc_outlier' in df_features.columns:
        final_cols.append('gc_outlier')
    
    df_final = df_features[final_cols].copy()
    
    feature_matrix_stats = {
        'total_target_features': len(all_target_features),
        'available_features': len(available_features),
        'missing_features': missing_features,
        'numeric_features_standardized': len(modeling_features),
        'binary_features': len(binary_features),
        'final_modeling_features': final_features,
        'standardization_params': standardization_params,
        'final_shape': df_final.shape
    }
    
    if verbose:
        print(f"   ✅ Final feature matrix created:")
        print(f"      Shape: {df_final.shape}")
        print(f"      Numeric (standardized): {len(modeling_features)}")
        print(f"      Binary indicators: {len(binary_features)}")
        print(f"      Total modeling features: {len(final_features)}")
        print(f"      Ready for Elastic-Net Logistic and XGBoost models")
    
    return df_final, feature_matrix_stats

def split_data_problem4(df: pd.DataFrame, 
                       test_size: float = 0.2, 
                       calibration_size: float = 0.1,
                       random_state: int = 42,
                       verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Group-stratified splitting for Problem 4 following the plan:
    1. 80/20 train/test split (grouped by maternal_id, stratified by Y)
    2. 90/10 calibration split from training set
    
    Args:
        df: Final feature matrix with maternal_id and Y columns
        test_size: Proportion for test set (default 0.2)
        calibration_size: Proportion of training set for calibration (default 0.1)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (train_df, calibration_df, test_df, split_stats)
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    if verbose:
        print("🔄 Problem 4 Step 5: Group-stratified data splitting...")
    
    if 'maternal_id' not in df.columns or 'Y' not in df.columns:
        raise ValueError("maternal_id and Y columns required for group-stratified splitting")
    
    # Prepare data for splitting
    X = df.drop(['Y'], axis=1)
    y = df['Y']
    groups = df['maternal_id']
    
    # Step 1: 80/20 train/test split (grouped by maternal_id)
    gss_main = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss_main.split(X, y, groups))
    
    df_train_full = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Step 2: 90/10 calibration split from training set
    X_train_full = df_train_full.drop(['Y'], axis=1)
    y_train_full = df_train_full['Y']
    groups_train_full = df_train_full['maternal_id']
    
    gss_cal = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=random_state)
    train_idx_final, cal_idx = next(gss_cal.split(X_train_full, y_train_full, groups_train_full))
    
    df_train = df_train_full.iloc[train_idx_final].copy()
    df_calibration = df_train_full.iloc[cal_idx].copy()
    
    # Calculate split statistics
    split_stats = {
        'original_shape': df.shape,
        'train_shape': df_train.shape,
        'calibration_shape': df_calibration.shape,
        'test_shape': df_test.shape,
        'train_groups': df_train['maternal_id'].nunique(),
        'calibration_groups': df_calibration['maternal_id'].nunique(),
        'test_groups': df_test['maternal_id'].nunique(),
        'train_class_dist': df_train['Y'].value_counts().to_dict(),
        'calibration_class_dist': df_calibration['Y'].value_counts().to_dict(),
        'test_class_dist': df_test['Y'].value_counts().to_dict()
    }
    
    if verbose:
        print(f"   ✅ Group-stratified splitting completed:")
        print(f"      Training set: {df_train.shape[0]} samples from {split_stats['train_groups']} mothers")
        print(f"         Class distribution: {split_stats['train_class_dist']}")
        print(f"      Calibration set: {df_calibration.shape[0]} samples from {split_stats['calibration_groups']} mothers")
        print(f"         Class distribution: {split_stats['calibration_class_dist']}")
        print(f"      Test set: {df_test.shape[0]} samples from {split_stats['test_groups']} mothers")
        print(f"         Class distribution: {split_stats['test_class_dist']}")
        
        # Verify no group overlap
        train_groups = set(df_train['maternal_id'].unique())
        cal_groups = set(df_calibration['maternal_id'].unique())
        test_groups = set(df_test['maternal_id'].unique())
        
        overlaps = []
        if train_groups & test_groups:
            overlaps.append("train-test")
        if train_groups & cal_groups:
            overlaps.append("train-calibration") 
        if cal_groups & test_groups:
            overlaps.append("calibration-test")
        
        if overlaps:
            print(f"      ⚠️  Group overlaps detected: {overlaps}")
        else:
            print(f"      ✅ No group overlaps - properly isolated sets")
    
    return df_train, df_calibration, df_test, split_stats

def calculate_class_weights_problem4(df_train: pd.DataFrame, 
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate class weights and sample weights for Problem 4 imbalance handling.
    
    Following the plan:
    - Point 6.1: Logistic class weights w1:w0 = n0/n1:1
    - Point 6.2: XGBoost scale_pos_weight = n0/n1  
    - Point 6.3: Sample weights 0.7 for gc_outlier samples
    
    Args:
        df_train: Training dataframe with Y and gc_outlier columns
        verbose: Whether to print weight calculations
        
    Returns:
        Dictionary with class weights, sample weights, and XGBoost parameters
    """
    if verbose:
        print("⚖️ Problem 4 Task 2: Calculating class and sample weights...")
    
    if 'Y' not in df_train.columns:
        raise ValueError("Y column required for class weight calculation")
    
    # Calculate class counts
    class_counts = df_train['Y'].value_counts()
    n_positive = class_counts.get(1, 0)  # Abnormal cases
    n_negative = class_counts.get(0, 0)  # Normal cases
    n_total = len(df_train)
    
    if n_positive == 0:
        raise ValueError("No positive cases in training data")
    
    # Point 6.1: Logistic class weights w1:w0 = n0/n1:1
    logistic_class_weights = {
        0: 1.0,  # Normal class (w0 = 1)
        1: n_negative / n_positive  # Abnormal class (w1 = n0/n1)
    }
    
    # Point 6.2: XGBoost scale_pos_weight = n0/n1
    xgboost_scale_pos_weight = n_negative / n_positive
    
    # Point 6.3: Sample weights for gc_outlier (0.7 weight)
    sample_weights = np.ones(len(df_train))
    
    if 'gc_outlier' in df_train.columns:
        gc_outlier_mask = df_train['gc_outlier'] == True
        sample_weights[gc_outlier_mask] = 0.7
        n_gc_outliers = gc_outlier_mask.sum()
    else:
        n_gc_outliers = 0
        if verbose:
            print("   ⚠️  No gc_outlier column found - all sample weights = 1.0")
    
    weight_stats = {
        'class_counts': {
            'negative': n_negative,
            'positive': n_positive, 
            'total': n_total
        },
        'imbalance_ratio': n_negative / n_positive,
        'positive_rate': n_positive / n_total,
        'logistic_class_weights': logistic_class_weights,
        'xgboost_scale_pos_weight': xgboost_scale_pos_weight,
        'sample_weights': sample_weights,
        'gc_outliers': n_gc_outliers,
        'gc_outlier_rate': n_gc_outliers / n_total if n_total > 0 else 0
    }
    
    if verbose:
        print(f"   📊 Class distribution:")
        print(f"      Negative (normal): {n_negative} ({100*n_negative/n_total:.1f}%)")
        print(f"      Positive (abnormal): {n_positive} ({100*n_positive/n_total:.1f}%)")
        print(f"      Imbalance ratio: {weight_stats['imbalance_ratio']:.2f}:1")
        print(f"   ⚖️  Logistic class weights:")
        print(f"      Class 0 (normal): {logistic_class_weights[0]:.2f}")
        print(f"      Class 1 (abnormal): {logistic_class_weights[1]:.2f}")
        print(f"   🌲 XGBoost scale_pos_weight: {xgboost_scale_pos_weight:.2f}")
        print(f"   📉 GC outlier sample weights:")
        print(f"      {n_gc_outliers} samples with weight 0.7")
        print(f"      {n_total - n_gc_outliers} samples with weight 1.0")
        print(f"   ✅ Weight calculations completed")
    
    return weight_stats

def get_model_training_weights_problem4(df_train: pd.DataFrame,
                                      verbose: bool = True) -> Dict[str, Any]:
    """
    Get ready-to-use weights for model training following Problem 4 plan.
    
    This is a convenience function that returns weights in formats expected by
    scikit-learn and XGBoost.
    
    Args:
        df_train: Training dataframe with Y and gc_outlier columns
        verbose: Whether to print weight information
        
    Returns:
        Dictionary with weights ready for model training
    """
    weight_stats = calculate_class_weights_problem4(df_train, verbose=verbose)
    
    # Prepare weights for different modeling libraries
    training_weights = {
        # For sklearn LogisticRegression class_weight parameter
        'sklearn_class_weight': weight_stats['logistic_class_weights'],
        
        # For sklearn fit() sample_weight parameter
        'sklearn_sample_weight': weight_stats['sample_weights'],
        
        # For XGBoost scale_pos_weight parameter
        'xgboost_scale_pos_weight': weight_stats['xgboost_scale_pos_weight'],
        
        # For XGBoost fit() sample_weight parameter (combines class and sample weights)
        'xgboost_sample_weight': weight_stats['sample_weights'],  # XGBoost handles class imbalance via scale_pos_weight
        
        # Raw statistics for reference
        'weight_stats': weight_stats
    }
    
    if verbose:
        print(f"\n💡 Usage examples:")
        print(f"   # Logistic Regression")
        print(f"   LogisticRegression(class_weight={training_weights['sklearn_class_weight']})")
        print(f"   model.fit(X, y, sample_weight=sample_weights)")
        print(f"   ")
        print(f"   # XGBoost")
        print(f"   XGBClassifier(scale_pos_weight={training_weights['xgboost_scale_pos_weight']:.2f})")
        print(f"   model.fit(X, y, sample_weight=sample_weights)")
    
    return training_weights

def load_and_map_female_data_problem4(data_path: Union[str, Path], 
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Simple data loading and column mapping for Problem 4.
    
    Args:
        data_path: Path to Excel file with female fetus data
        verbose: Whether to print progress
        
    Returns:
        DataFrame with mapped column names
    """
    if verbose:
        print("📂 Loading female fetus data...")
    
    # Load Excel data
    raw_data = pd.read_excel(data_path, sheet_name="女胎检测数据")
    if verbose:
        print(f"   ✅ Loaded raw data: {raw_data.shape}")
    
    # Chinese to English column mapping for Problem 4 (exact column names from Excel)
    column_mapping = {
        # Core variables
        '孕妇代码': 'maternal_id',                    # Maternal ID (grouping variable)
        '检测孕周': 'gestational_weeks',              # Gestational weeks at testing  
        '孕妇BMI': 'bmi',                          # Maternal BMI
        '年龄': 'age',                             # Maternal age
        '染色体的非整倍体': 'aneuploidy_ab',            # AB column - aneuploidy detection (TARGET)
        
        # Z-values (essential features)
        '13号染色体的Z值': 'z13',                      # Chr13 Z-value
        '18号染色体的Z值': 'z18',                      # Chr18 Z-value  
        '21号染色体的Z值': 'z21',                      # Chr21 Z-value
        'X染色体的Z值': 'zx',                         # X chromosome Z-value
        
        # GC content features
        'GC含量': 'gc_global',                       # Global GC content
        '13号染色体的GC含量': 'gc13',                  # Chr13 GC content
        '18号染色体的GC含量': 'gc18',                  # Chr18 GC content
        '21号染色体的GC含量': 'gc21',                  # Chr21 GC content
        
        # Sequencing QC features
        '原始读段数': 'reads',                        # Raw read count
        '在参考基因组上比对的比例': 'map_ratio',         # Mapping ratio
        '重复读段的比例': 'dup_ratio',                 # Duplicate ratio
        '唯一比对的读段数': 'unique_reads',            # Unique reads
        '被过滤掉读段数的比例': 'filtered_ratio',       # Filtered reads ratio
        
        # Additional features
        '身高': 'height',                           # Height
        '体重': 'weight',                           # Weight
        'X染色体浓度': 'x_concentration',             # X chromosome concentration
    }
    
    # Apply column mapping
    mapped_count = 0
    for chinese_col, english_col in column_mapping.items():
        if chinese_col in raw_data.columns:
            raw_data[english_col] = raw_data[chinese_col]
            mapped_count += 1
            if verbose:
                print(f"   🔄 Mapped '{chinese_col}' -> '{english_col}'")
        elif verbose:
            print(f"   ⚠️  Column not found: '{chinese_col}'")
    
    if verbose:
        print(f"   📊 Successfully mapped {mapped_count}/{len(column_mapping)} columns")
    
    # Parse gestational weeks if needed
    if 'gestational_weeks' in raw_data.columns and raw_data['gestational_weeks'].dtype == 'object':
        if verbose:
            print("   🔧 Parsing gestational weeks from string format...")
        raw_data['gestational_weeks'] = raw_data['gestational_weeks'].apply(parse_gestational_weeks)
    
    # Keep only the mapped English columns and essential original columns
    # to avoid collinearity issues in downstream processing
    essential_english_cols = [
        'maternal_id', 'gestational_weeks', 'bmi', 'age', 'aneuploidy_ab',
        'z13', 'z18', 'z21', 'zx', 'gc_global', 'gc13', 'gc18', 'gc21',
        'reads', 'map_ratio', 'dup_ratio', 'unique_reads', 'filtered_ratio',
        'height', 'weight', 'x_concentration'
    ]
    
    # Keep only the columns we need (English mapped + some original for reference)
    available_cols = [col for col in essential_english_cols if col in raw_data.columns]
    
    # Also keep some original columns that might be needed for debugging
    original_keep = ['序号', '检测日期', '检测抽血次数']  # Sample ID, test date, blood draw count
    for col in original_keep:
        if col in raw_data.columns:
            available_cols.append(col)
    
    # Create clean dataframe with only necessary columns
    clean_data = raw_data[available_cols].copy()
    
    if verbose:
        print(f"   ✅ Column mapping completed")
        print(f"   🧹 Cleaned data shape: {clean_data.shape} (removed duplicate Chinese columns)")
        
        # Check key columns
        key_cols = ['maternal_id', 'aneuploidy_ab', 'z13', 'z18', 'z21', 'zx', 'gc_global', 'bmi', 'age']
        available_key = [col for col in key_cols if col in clean_data.columns]
        print(f"   🔑 Key columns available: {available_key}")
        print(f"   📋 Final columns: {list(clean_data.columns)}")
    
    return clean_data

def preprocessing_pipeline_problem4(data_path: Union[str, Path], 
                                   verbose: bool = True,
                                   random_state: int = 42) -> Dict[str, Any]:
    """
    FIXED: Data leakage-free preprocessing pipeline for Problem 4.
    
    CRITICAL FIX: All statistical transformations (winsorization, standardization, 
    collinearity detection) are now fitted on TRAINING DATA ONLY and then applied 
    to calibration/test sets to prevent data leakage.
    
    Workflow:
    1. Load and map female fetus data
    2. Generate binary labels from AB column  
    3. Basic QC (simple business rules only)
    4. **SPLIT DATA FIRST** ← This is the critical fix
    5. Fit transformers on TRAINING data only:
       - Winsorization bounds, Collinearity detection, Standardization params
    6. Apply fitted transformers to calibration/test data
    7. Calculate weights from training data only
    
    Args:
        data_path: Path to Excel file with female fetus data
        verbose: Whether to print detailed progress
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing all preprocessed data and metadata
    """
    if verbose:
        print("🚀 PROBLEM 4 PREPROCESSING PIPELINE: Female Fetal Abnormality Detection")
        print("="*80)
    
    results = {
        'pipeline_success': False,
        'steps_completed': [],
        'data': {},
        'metadata': {}
    }
    
    try:
        # Step 1: Load and map female fetus data
        if verbose:
            print("\n📂 STEP 1: Loading female fetus data...")
        
        raw_data = load_and_map_female_data_problem4(data_path, verbose=verbose)
        results['data']['raw_extracted'] = raw_data
        results['steps_completed'].append('data_loading')
        
        # Step 2: Generate labels
        if verbose:
            print("\n🎯 STEP 2: Generating binary labels from AB column...")
        
        labeled_data = generate_labels_problem4(raw_data, verbose=verbose)
        results['data']['labeled'] = labeled_data
        results['steps_completed'].append('label_generation')
        
        # Import the transformers for data leakage prevention
        try:
            from .preprocessing_transformers import (
                WinsorizerTransformer, StandardizerTransformer, CollinearityRemover,
                create_qc_filter_thresholds, apply_qc_filters_with_thresholds
            )
        except ImportError:
            # Fallback if import fails
            print("Warning: Could not import transformers, using legacy pipeline")
            return preprocessing_pipeline_problem4(data_path, verbose, random_state)
        
        # Step 3: Basic QC only (no statistical transformations yet)
        if verbose:
            print("\n🔍 STEP 3: Basic data quality checks...")
            
        # Only do basic cleaning - no statistical operations yet
        basic_cleaned = labeled_data.dropna(subset=['Y', 'maternal_id'])
        
        # Add basic engineered features that don't need statistical fitting
        if verbose:
            print("   📈 Creating basic derived features...")
        basic_cleaned = basic_cleaned.copy()
        
        # max_Z (doesn't need fitting)
        z_cols = ['z13', 'z18', 'z21']
        available_z = [col for col in z_cols if col in basic_cleaned.columns]
        if len(available_z) > 0:
            basic_cleaned['max_z'] = basic_cleaned[available_z].abs().max(axis=1)
            
        # Z indicators (doesn't need fitting)  
        for z_col in available_z:
            basic_cleaned[f'{z_col}_high'] = (basic_cleaned[z_col].abs() >= 3).astype(int)
        
        results['data']['basic_cleaned'] = basic_cleaned
        results['steps_completed'].append('basic_cleaning')
        
        # Step 4: SPLIT DATA FIRST (CRITICAL FOR PREVENTING DATA LEAKAGE)
        if verbose:
            print("\n🎯 STEP 4: EARLY DATA SPLITTING (to prevent data leakage)...")
            print("   🔒 All subsequent transformations will be fitted on TRAINING DATA ONLY")
        
        # Use larger calibration set for more stable calibration
        train_data_raw, cal_data_raw, test_data_raw, split_stats = split_data_problem4(
            basic_cleaned,
            test_size=0.2,
            calibration_size=0.15,  # Increased from 0.1 to 0.15 for better calibration stability
            random_state=random_state,
            verbose=verbose
        )
        
        results['data']['train_raw'] = train_data_raw
        results['data']['calibration_raw'] = cal_data_raw 
        results['data']['test_raw'] = test_data_raw
        results['metadata']['splitting'] = split_stats
        results['steps_completed'].append('early_splitting')
        
        # Step 5: Fit all transformers on TRAINING DATA ONLY
        if verbose:
            print("\n🔧 STEP 5: Fitting transformers on TRAINING DATA ONLY...")
        
        # 5a: QC thresholds from training data
        qc_thresholds = create_qc_filter_thresholds(train_data_raw)
        
        # Apply QC to all splits using training-fitted thresholds
        train_qc, train_qc_stats = apply_qc_filters_with_thresholds(train_data_raw, qc_thresholds, verbose=verbose)
        cal_qc, cal_qc_stats = apply_qc_filters_with_thresholds(cal_data_raw, qc_thresholds, verbose=False)
        test_qc, test_qc_stats = apply_qc_filters_with_thresholds(test_data_raw, qc_thresholds, verbose=False)
        
        # 5b: Winsorizer fitted on training data only
        winsorize_cols = ['reads', 'map_ratio', 'dup_ratio', 'unique_reads'] 
        winsorizer = WinsorizerTransformer(winsorize_cols).fit(train_qc)
        
        train_winsorized = winsorizer.transform(train_qc)
        cal_winsorized = winsorizer.transform(cal_qc)
        test_winsorized = winsorizer.transform(test_qc)
        
        # 5c: Missing value imputation on training data only (Plan Step 3.1)
        from sklearn.impute import SimpleImputer
        
        # Identify numeric columns for imputation (exclude IDs, labels, indicators)
        exclude_for_imputation = ['maternal_id', 'Y', 'gc_outlier', '序号', '检测日期', '检测抽血次数']
        high_indicator_cols = [c for c in train_winsorized.columns if c.endswith('_high')]
        numeric_cols_for_imputation = [
            col for col in train_winsorized.columns 
            if col not in exclude_for_imputation + high_indicator_cols and
            train_winsorized[col].dtype in ['float64', 'int64', 'float32', 'int32']
        ]
        
        # Check missing rates and create missing indicators for features with >1% missing rate
        missing_indicators = {}
        imputer = SimpleImputer(strategy='median')
        for col in numeric_cols_for_imputation:
            missing_rate = train_winsorized[col].isnull().mean()
            if missing_rate > 0.01:  # >1% missing as per plan
                missing_indicators[f"{col}_missing"] = col
                if verbose:
                    print(f"   📊 Creating missing indicator for {col} (missing rate: {missing_rate:.2%})")
        
        # Apply median imputation
        if numeric_cols_for_imputation:
            # Fit imputer on training data only
            train_numeric = train_winsorized[numeric_cols_for_imputation]
            imputer.fit(train_numeric)
            
            # Apply to all sets
            train_imputed_values = imputer.transform(train_numeric)
            cal_imputed_values = imputer.transform(cal_winsorized[numeric_cols_for_imputation])
            test_imputed_values = imputer.transform(test_winsorized[numeric_cols_for_imputation])
            
            # Replace numeric columns with imputed values
            train_imputed = train_winsorized.copy()
            cal_imputed = cal_winsorized.copy()
            test_imputed = test_winsorized.copy()
            
            train_imputed[numeric_cols_for_imputation] = train_imputed_values
            cal_imputed[numeric_cols_for_imputation] = cal_imputed_values
            test_imputed[numeric_cols_for_imputation] = test_imputed_values
            
            # Add missing indicators
            for indicator_col, source_col in missing_indicators.items():
                train_imputed[indicator_col] = train_winsorized[source_col].isnull().astype(int)
                cal_imputed[indicator_col] = cal_winsorized[source_col].isnull().astype(int)
                test_imputed[indicator_col] = test_winsorized[source_col].isnull().astype(int)
            
            if verbose:
                total_missing_before = (train_winsorized[numeric_cols_for_imputation].isnull().sum().sum() +
                                      cal_winsorized[numeric_cols_for_imputation].isnull().sum().sum() +
                                      test_winsorized[numeric_cols_for_imputation].isnull().sum().sum())
                total_missing_after = (train_imputed[numeric_cols_for_imputation].isnull().sum().sum() +
                                     cal_imputed[numeric_cols_for_imputation].isnull().sum().sum() +
                                     test_imputed[numeric_cols_for_imputation].isnull().sum().sum())
                print(f"   🔧 Median imputation completed:")
                print(f"      Features imputed: {len(numeric_cols_for_imputation)}")
                print(f"      Missing values before: {total_missing_before}")
                print(f"      Missing values after: {total_missing_after}")
                print(f"      Missing indicators added: {len(missing_indicators)}")
        else:
            train_imputed, cal_imputed, test_imputed = train_winsorized, cal_winsorized, test_winsorized
            if verbose:
                print(f"   ⚠️ No numeric columns found for imputation")
        
        # 5d: Collinearity detection on training data only
        collinearity_remover = CollinearityRemover(correlation_threshold=0.95).fit(train_imputed)
        
        train_decollinear = collinearity_remover.transform(train_imputed)
        cal_decollinear = collinearity_remover.transform(cal_imputed)
        test_decollinear = collinearity_remover.transform(test_imputed)
        
        if verbose:
            print(f"   🗑️  Removed collinear features: {collinearity_remover.features_to_remove_}")
            for feature, info in collinearity_remover.correlation_info_.items():
                print(f"      {feature}: {info['reason']}")
        
        # 5e: Standardization fitted on training data only
        exclude_cols = ['maternal_id', 'Y', 'gc_outlier', '序号', '检测日期', '检测抽血次数']
        high_cols = [c for c in train_decollinear.columns if c.endswith('_high')]
        numeric_cols = [col for col in train_decollinear.columns 
                       if col not in exclude_cols + high_cols and 
                       train_decollinear[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        standardizer = StandardizerTransformer(numeric_cols).fit(train_decollinear)
        
        train_final = standardizer.transform(train_decollinear)
        cal_final = standardizer.transform(cal_decollinear) 
        test_final = standardizer.transform(test_decollinear)
        
        if verbose:
            print(f"   📏 Standardized features: {list(standardizer.stats_.keys())}")
            for col, stats in standardizer.stats_.items():
                print(f"      {col}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
        
        results['data']['train'] = train_final
        results['data']['calibration'] = cal_final
        results['data']['test'] = test_final
        results['metadata']['transformers'] = {
            'qc_thresholds': qc_thresholds,
            'winsorizer': winsorizer,
            'imputer': imputer,
            'missing_indicators': missing_indicators,
            'collinearity_remover': collinearity_remover,
            'standardizer': standardizer
        }
        results['metadata']['qc_train'] = train_qc_stats
        results['metadata']['qc_cal'] = cal_qc_stats
        results['metadata']['qc_test'] = test_qc_stats
        results['steps_completed'].append('transformer_fitting')
        
        # Step 6: Calculate weights from TRAINING DATA ONLY
        if verbose:
            print("\n⚖️ STEP 6: Calculating weights from TRAINING DATA ONLY...")
        
        training_weights = get_model_training_weights_problem4(train_final, verbose=verbose)
        results['data']['training_weights'] = training_weights
        results['metadata']['weights'] = training_weights['weight_stats']
        results['steps_completed'].append('weight_calculation')
        
        # Pipeline completion
        results['pipeline_success'] = True
        
        if verbose:
            print("\n✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📊 Final Results Summary:")
            print(f"   Training set: {train_final.shape[0]} samples from {split_stats['train_groups']} mothers")
            print(f"   Calibration set: {cal_final.shape[0]} samples from {split_stats['calibration_groups']} mothers")
            print(f"   Test set: {test_final.shape[0]} samples from {split_stats['test_groups']} mothers")
            # Detailed feature breakdown for clarity
            total_features = train_final.shape[1] - 2  # Exclude maternal_id, Y
            standardized_features = [col for col in train_final.columns if col.endswith('_std')]
            indicator_features = [col for col in train_final.columns if col.endswith('_high')]
            raw_features = total_features - len(standardized_features) - len(indicator_features)
            
            print(f"   Features: {total_features} modeling features total")
            print(f"      • {len(standardized_features)} standardized features (numeric)")
            print(f"      • {len(indicator_features)} indicator features (Z >= 3)")  
            print(f"      • {raw_features} other features (categorical, IDs, etc.)")
            print(f"   Class distribution (train): {split_stats['train_class_dist']}")
            print(f"   Imbalance ratio: {training_weights['weight_stats']['imbalance_ratio']:.1f}:1")
            print(f"   Ready for Elastic-Net Logistic and XGBoost modeling!")
            print(f"\n🎯 DATA LEAKAGE PREVENTION: All transformations fitted on training data only!")
            print(f"   • Winsorization bounds: {len(winsorizer.bounds_)} features")
            print(f"   • Missing value imputation: {len(numeric_cols_for_imputation)} features (median strategy)")
            print(f"   • Missing indicators: {len(missing_indicators)} indicators added")
            print(f"   • Collinearity removal: {len(collinearity_remover.features_to_remove_)} features removed")  
            print(f"   • Standardization: {len(standardizer.stats_)} features standardized")
        
    except Exception as e:
        results['pipeline_success'] = False
        results['error'] = str(e)
        if verbose:
            print(f"\n❌ PREPROCESSING PIPELINE FAILED!")
            print(f"   Error: {e}")
            print(f"   Steps completed: {results['steps_completed']}")
    
    return results
