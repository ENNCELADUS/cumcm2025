"""
Machine Learning baseline comparison for Problem 2.

This module provides traditional ML approaches for comparison against AFT survival models:
1. Classification component: Binary outcome (ever reached 4%)
2. Regression component: Time to threshold (interval midpoint approximation)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.calibration import calibration_curve
import warnings


def prepare_ml_dataset(df_intervals: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare dataset for ML baseline comparison.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        verbose: Whether to print preparation details
        
    Returns:
        Tuple of (prepared_dataset, preparation_info)
    """
    if verbose:
        print("🔧 Preparing dataset for ML baseline comparison...")
    
    df_ml = df_intervals.copy()
    
    # Binary outcome: ever reached 4% threshold
    df_ml['ever_reached_threshold'] = (df_ml['censor_type'] != 'right').astype(int)
    
    # Continuous outcome: time to threshold (interval midpoint approximation)
    def calculate_time_approximation(row):
        if row['censor_type'] == 'left':
            # Reached threshold before first measurement
            return row['R'] / 2  # Use half of first measurement time
        elif row['censor_type'] == 'interval':
            # Reached threshold between measurements
            return (row['L'] + row['R']) / 2  # Use interval midpoint
        else:  # right-censored
            # Never reached threshold - use last observation time + offset
            return row['L'] + 2.0  # Assume would have reached 2 weeks later
    
    df_ml['time_to_threshold_approx'] = df_ml.apply(calculate_time_approximation, axis=1)
    
    # Features for ML models
    feature_cols = ['bmi']
    
    # Create bmi_z if it doesn't exist (standardized BMI)
    if 'bmi_z' not in df_ml.columns:
        bmi_mean = df_ml['bmi'].mean()
        bmi_std = df_ml['bmi'].std()
        df_ml['bmi_z'] = (df_ml['bmi'] - bmi_mean) / bmi_std
        if verbose:
            print(f"  Created bmi_z column (standardized BMI): mean={bmi_mean:.2f}, std={bmi_std:.2f}")
    
    feature_cols.append('bmi_z')
    
    # Additional derived features
    df_ml['bmi_squared'] = df_ml['bmi'] ** 2
    df_ml['bmi_log'] = np.log(df_ml['bmi'])
    feature_cols.extend(['bmi_squared', 'bmi_log'])
    
    # Create feature matrix
    X = df_ml[feature_cols].copy()
    
    # Handle any potential infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        if verbose:
            print("⚠️ Found NaN values in features, using forward fill")
        X = X.fillna(method='ffill').fillna(X.mean())
    
    preparation_info = {
        'n_samples': len(df_ml),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'ever_reached_rate': df_ml['ever_reached_threshold'].mean(),
        'mean_time_approx': df_ml['time_to_threshold_approx'].mean(),
        'censoring_distribution': df_ml['censor_type'].value_counts().to_dict()
    }
    
    if verbose:
        print(f"✅ ML dataset prepared:")
        print(f"  Samples: {preparation_info['n_samples']}")
        print(f"  Features: {preparation_info['n_features']} ({', '.join(feature_cols)})")
        print(f"  Ever reached threshold: {preparation_info['ever_reached_rate']:.3f}")
        print(f"  Mean time approximation: {preparation_info['mean_time_approx']:.2f} weeks")
    
    return df_ml, preparation_info


def train_classification_models(X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5, random_state: int = 42,
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Train and evaluate classification models for binary outcome.
    
    Args:
        X: Feature matrix
        y: Binary outcome (ever reached threshold)
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        verbose: Whether to print training details
        
    Returns:
        Dictionary with model results
    """
    if verbose:
        print("🎯 Training classification models for binary outcome...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    
    results = {}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        if verbose:
            print(f"\n  🔹 Training {name}...")
        
        # Fit full model
        model.fit(X, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        # Predictions for calibration
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calibration assessment
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=5, strategy='quantile'
        )
        
        # Calculate Brier score (mean squared difference between predicted probabilities and outcomes)
        brier_score = np.mean((y_pred_proba - y) ** 2)
        
        results[name] = {
            'model': model,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'full_auc': roc_auc_score(y, y_pred_proba),
            'brier_score': brier_score,
            'calibration': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }
        
        if verbose:
            print(f"    AUC (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"    AUC (full): {roc_auc_score(y, y_pred_proba):.3f}")
            print(f"    Brier score: {brier_score:.3f}")
    
    # Select best model by CV AUC
    best_name = max(results.keys(), key=lambda x: results[x]['cv_auc_mean'])
    
    if verbose:
        print(f"\n✅ Best classification model: {best_name}")
        print(f"  Cross-validation AUC: {results[best_name]['cv_auc_mean']:.3f}")
    
    results['best_model'] = best_name
    return results


def train_regression_models(X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5, random_state: int = 42,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Train and evaluate regression models for time to threshold.
    
    Args:
        X: Feature matrix
        y: Continuous outcome (time to threshold approximation)
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        verbose: Whether to print training details
        
    Returns:
        Dictionary with model results
    """
    if verbose:
        print("📈 Training regression models for time to threshold...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
    }
    
    results = {}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        if verbose:
            print(f"\n  🔹 Training {name}...")
        
        # Fit full model
        model.fit(X, y)
        
        # Cross-validation scores
        cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        
        # Full model predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = model.score(X, y)
        
        results[name] = {
            'model': model,
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_rmse_mean': cv_rmse.mean(), 
            'cv_rmse_std': cv_rmse.std(),
            'full_mae': mae,
            'full_rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        if verbose:
            print(f"    MAE (CV): {cv_mae.mean():.3f} ± {cv_mae.std():.3f} weeks")
            print(f"    RMSE (CV): {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} weeks") 
            print(f"    MAE (full): {mae:.3f} weeks")
            print(f"    R²: {r2:.3f}")
    
    # Select best model by CV MAE
    best_name = min(results.keys(), key=lambda x: results[x]['cv_mae_mean'])
    
    if verbose:
        print(f"\n✅ Best regression model: {best_name}")
        print(f"  Cross-validation MAE: {results[best_name]['cv_mae_mean']:.3f} weeks")
    
    results['best_model'] = best_name
    return results


def map_ml_to_group_recommendations(classification_results: Dict[str, Any],
                                   regression_results: Dict[str, Any],
                                   df_ml: pd.DataFrame,
                                   confidence_levels: List[float] = [0.90, 0.95],
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Map ML predictions to group-level recommendations for comparison with AFT.
    
    Args:
        classification_results: Results from classification models
        regression_results: Results from regression models  
        df_ml: ML dataset with BMI groups
        confidence_levels: List of confidence levels
        verbose: Whether to print mapping details
        
    Returns:
        Dictionary with group-level ML recommendations
    """
    if verbose:
        print("🔄 Mapping ML predictions to group-level recommendations...")
    
    # Get best models
    best_classifier = classification_results[classification_results['best_model']]['model']
    best_regressor = regression_results[regression_results['best_model']]['model']
    
    # Create feature matrix
    feature_cols = ['bmi', 'bmi_z', 'bmi_squared', 'bmi_log']
    X = df_ml[feature_cols]
    
    # Get predictions
    threshold_probs = best_classifier.predict_proba(X)[:, 1]
    predicted_times = best_regressor.predict(X)
    
    # Add predictions to dataframe
    df_with_preds = df_ml.copy()
    df_with_preds['ml_threshold_prob'] = threshold_probs
    df_with_preds['ml_predicted_time'] = predicted_times
    
    # Create simple BMI groups for comparison (clinical categories)
    df_with_preds['bmi_group_ml'] = df_with_preds['bmi'].apply(assign_clinical_bmi_group)
    
    # Group-level aggregation
    group_results = {}
    
    for group in df_with_preds['bmi_group_ml'].unique():
        group_data = df_with_preds[df_with_preds['bmi_group_ml'] == group]
        
        # Group statistics
        group_stats = {
            'n_mothers': len(group_data),
            'bmi_mean': group_data['bmi'].mean(),
            'bmi_range': f"{group_data['bmi'].min():.1f}-{group_data['bmi'].max():.1f}",
            'threshold_prob_mean': group_data['ml_threshold_prob'].mean(),
            'predicted_time_mean': group_data['ml_predicted_time'].mean(),
            'predicted_time_std': group_data['ml_predicted_time'].std()
        }
        
        # Approximate optimal weeks using quantiles of predicted times
        # This is a simplified mapping from ML predictions to testing recommendations
        ml_optimal_weeks = {}
        
        for conf in confidence_levels:
            # Use the (1-conf) quantile of predicted times as "optimal" week
            # Logic: Test when conf% of mothers in group are expected to have reached threshold
            quantile = 1 - conf  
            optimal_week = group_data['ml_predicted_time'].quantile(quantile)
            ml_optimal_weeks[conf] = optimal_week
        
        group_stats['ml_optimal_weeks'] = ml_optimal_weeks
        group_results[group] = group_stats
        
        if verbose:
            print(f"\n  📊 Group: {group}")
            print(f"    n = {group_stats['n_mothers']}, BMI = {group_stats['bmi_mean']:.1f}")
            print(f"    Threshold prob: {group_stats['threshold_prob_mean']:.3f}")
            print(f"    Predicted time: {group_stats['predicted_time_mean']:.2f} ± {group_stats['predicted_time_std']:.2f}")
            for conf in confidence_levels:
                week = ml_optimal_weeks[conf]
                print(f"    ML optimal week ({int(conf*100)}%): {week:.2f}")
    
    return {
        'group_results': group_results,
        'classification_model': classification_results['best_model'],
        'regression_model': regression_results['best_model'],
        'df_with_predictions': df_with_preds
    }


def assign_clinical_bmi_group(bmi: float) -> str:
    """Assign clinical BMI categories for ML baseline."""
    if bmi < 25:
        return "Normal (<25)"
    elif bmi < 30:
        return "Overweight (25-30)"
    elif bmi < 35:
        return "Obese I (30-35)"
    else:
        return "Obese II+ (≥35)"


def compare_aft_vs_ml_recommendations(aft_group_results: Dict[str, Any],
                                     ml_group_results: Dict[str, Any],
                                     confidence_levels: List[float] = [0.90, 0.95],
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Compare AFT vs ML group-specific recommendations.
    
    Args:
        aft_group_results: Results from AFT group-specific analysis
        ml_group_results: Results from ML baseline analysis
        confidence_levels: List of confidence levels
        verbose: Whether to print comparison
        
    Returns:
        DataFrame with side-by-side comparison
    """
    if verbose:
        print("⚖️ Comparing AFT vs ML baseline recommendations...")
    
    comparison_data = []
    
    # Get AFT group results
    aft_groups = aft_group_results['group_results'] if 'group_results' in aft_group_results else aft_group_results
    ml_groups = ml_group_results['group_results']
    
    # Find common groups (by name matching)
    common_groups = set(aft_groups.keys()) & set(ml_groups.keys())
    
    if verbose:
        print(f"  Comparing {len(common_groups)} common BMI groups...")
    
    for group in common_groups:
        aft_data = aft_groups[group]
        ml_data = ml_groups[group]
        
        row = {
            'BMI_Group': group,
            'n_mothers_AFT': aft_data.get('n_mothers', 0),
            'n_mothers_ML': ml_data.get('n_mothers', 0),
            'bmi_mean_AFT': aft_data.get('representative_bmi', 0),
            'bmi_mean_ML': ml_data.get('bmi_mean', 0)
        }
        
        # Add optimal weeks for each confidence level
        for conf in confidence_levels:
            conf_pct = int(conf * 100)
            
            # AFT optimal weeks
            if 'optimal_weeks' in aft_data and conf in aft_data['optimal_weeks']:
                aft_week = aft_data['optimal_weeks'][conf]
                aft_week_str = f"{aft_week:.2f}" if aft_week != np.inf else "Never"
            else:
                aft_week_str = "N/A"
                aft_week = np.nan
            
            # ML optimal weeks
            if 'ml_optimal_weeks' in ml_data and conf in ml_data['ml_optimal_weeks']:
                ml_week = ml_data['ml_optimal_weeks'][conf]
                ml_week_str = f"{ml_week:.2f}" if not np.isnan(ml_week) else "N/A"
            else:
                ml_week_str = "N/A"
                ml_week = np.nan
            
            # Calculate difference
            if not (np.isnan(aft_week) or np.isnan(ml_week) or aft_week == np.inf):
                diff = aft_week - ml_week
                diff_str = f"{diff:+.2f}"
            else:
                diff_str = "N/A"
            
            row[f'AFT_week_{conf_pct}'] = aft_week_str
            row[f'ML_week_{conf_pct}'] = ml_week_str
            row[f'diff_{conf_pct}'] = diff_str
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if verbose and not comparison_df.empty:
        print("📊 AFT vs ML Comparison:")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        print(f"\n📈 Comparison Summary:")
        for conf in confidence_levels:
            conf_pct = int(conf * 100)
            diff_col = f'diff_{conf_pct}'
            
            if diff_col in comparison_df.columns:
                # Extract numeric differences (excluding "N/A")
                numeric_diffs = []
                for diff_str in comparison_df[diff_col]:
                    if diff_str != "N/A":
                        try:
                            numeric_diffs.append(float(diff_str))
                        except:
                            pass
                
                if numeric_diffs:
                    mean_diff = np.mean(numeric_diffs)
                    std_diff = np.std(numeric_diffs)
                    print(f"  {conf_pct}% confidence level:")
                    print(f"    Mean difference (AFT - ML): {mean_diff:+.2f} ± {std_diff:.2f} weeks")
                    print(f"    Agreement: {'Good' if abs(mean_diff) < 1.0 else 'Moderate' if abs(mean_diff) < 2.0 else 'Poor'}")
    
    return comparison_df


def run_ml_baseline_comparison(df_intervals: pd.DataFrame,
                              aft_group_results: Dict[str, Any],
                              confidence_levels: List[float] = [0.90, 0.95],
                              cv_folds: int = 5,
                              random_state: int = 42,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Run complete ML baseline comparison pipeline.
    
    Args:
        df_intervals: Interval-censored dataset
        aft_group_results: Results from AFT group-specific analysis
        confidence_levels: List of confidence levels
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with complete ML baseline results
    """
    if verbose:
        print("🤖 Running ML baseline comparison pipeline...")
    
    # Step 1: Prepare ML dataset
    df_ml, prep_info = prepare_ml_dataset(df_intervals, verbose=verbose)
    
    # Step 2: Train classification models
    X = df_ml[['bmi', 'bmi_z', 'bmi_squared', 'bmi_log']]
    y_binary = df_ml['ever_reached_threshold']
    
    classification_results = train_classification_models(
        X, y_binary, cv_folds=cv_folds, random_state=random_state, verbose=verbose
    )
    
    # Step 3: Train regression models  
    y_continuous = df_ml['time_to_threshold_approx']
    
    regression_results = train_regression_models(
        X, y_continuous, cv_folds=cv_folds, random_state=random_state, verbose=verbose
    )
    
    # Step 4: Map to group recommendations
    ml_group_results = map_ml_to_group_recommendations(
        classification_results, regression_results, df_ml, 
        confidence_levels=confidence_levels, verbose=verbose
    )
    
    # Step 5: Compare with AFT results
    comparison_df = compare_aft_vs_ml_recommendations(
        aft_group_results, ml_group_results, 
        confidence_levels=confidence_levels, verbose=verbose
    )
    
    if verbose:
        print(f"\n✅ ML baseline comparison completed")
    
    return {
        'preparation_info': prep_info,
        'classification_results': classification_results,
        'regression_results': regression_results,
        'ml_group_results': ml_group_results,
        'comparison_df': comparison_df,
        'df_ml': df_ml
    }
