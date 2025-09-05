"""
Validation and final policy table generation for Problem 2.

This module provides comprehensive validation methods and generates
the final policy recommendation table with uncertainty quantification.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import KFold
from pathlib import Path
import warnings

# Import project paths
try:
    from src.config.settings import RESULTS_DIR
except ImportError:
    # Fallback to absolute path calculation
    RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "output" / "results"


def perform_cross_validation_analysis(df_intervals: pd.DataFrame,
                                     construct_intervals_func,
                                     fit_aft_models_func,
                                     perform_group_analysis_func,
                                     k_folds: int = 5,
                                     confidence_levels: List[float] = [0.90, 0.95],
                                     random_state: int = 42,
                                     verbose: bool = True) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation of the AFT modeling pipeline.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        construct_intervals_func: Function to construct intervals
        fit_aft_models_func: Function to fit AFT models
        perform_group_analysis_func: Function for group analysis
        k_folds: Number of cross-validation folds
        confidence_levels: List of confidence levels
        random_state: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with cross-validation results
    """
    if verbose:
        print(f"🔄 Performing {k_folds}-fold cross-validation of AFT pipeline...")
    
    # Use maternal_id for proper splitting (avoid data leakage)
    unique_mothers = df_intervals['maternal_id'].unique()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_mothers)):
        if verbose:
            print(f"\n  📊 Fold {fold_idx + 1}/{k_folds}")
        
        # Split by maternal_id
        train_mothers = unique_mothers[train_idx]
        test_mothers = unique_mothers[test_idx]
        
        train_data = df_intervals[df_intervals['maternal_id'].isin(train_mothers)]
        test_data = df_intervals[df_intervals['maternal_id'].isin(test_mothers)]
        
        if verbose:
            print(f"    Train: {len(train_data)} intervals from {len(train_mothers)} mothers")
            print(f"    Test: {len(test_data)} intervals from {len(test_mothers)} mothers")
        
        try:
            # Prepare training data
            train_X = train_data.copy()
            bmi_mean = train_X['bmi'].mean()
            bmi_std = train_X['bmi'].std()
            train_X['bmi_z'] = (train_X['bmi'] - bmi_mean) / bmi_std
            
            # Fit AFT model on training data
            primary_model, primary_name, all_models = fit_aft_models_func(
                train_X, formula='~ bmi_z', verbose=False
            )
            
            if primary_model is None:
                if verbose:
                    print(f"    ❌ Model fitting failed")
                cv_results.append({
                    'fold': fold_idx,
                    'success': False,
                    'error': 'Model fitting failed'
                })
                continue
            
            # Perform group analysis on training data
            time_grid = np.linspace(10, 25, 100)
            group_results = perform_group_analysis_func(
                train_data,
                primary_model,
                confidence_levels=confidence_levels,
                time_grid=time_grid,
                grouping_methods=['clinical', 'tertile'],
                verbose=False
            )
            
            # Evaluate on test data (predict optimal weeks for test mothers)
            test_predictions = []
            
            for _, test_row in test_data.iterrows():
                test_bmi_z = (test_row['bmi'] - bmi_mean) / bmi_std
                X_query = pd.DataFrame({'bmi_z': [test_bmi_z]})
                
                try:
                    # Predict survival function
                    surv_func = primary_model.predict_survival_function(X_query, times=time_grid)
                    survival_values = surv_func.iloc[:, 0].values
                    
                    # Calculate optimal weeks
                    optimal_weeks = {}
                    for conf in confidence_levels:
                        threshold = 1 - conf
                        crossing_indices = np.where(survival_values <= threshold)[0]
                        
                        if len(crossing_indices) > 0:
                            optimal_week = time_grid[crossing_indices[0]]
                        else:
                            optimal_week = np.inf
                        
                        optimal_weeks[conf] = optimal_week
                    
                    test_predictions.append({
                        'maternal_id': test_row['maternal_id'],
                        'bmi': test_row['bmi'],
                        'actual_censor_type': test_row['censor_type'],
                        'predicted_optimal_weeks': optimal_weeks
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"    ⚠️ Prediction failed for mother {test_row['maternal_id']}: {e}")
            
            cv_results.append({
                'fold': fold_idx,
                'success': True,
                'model_name': primary_name,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'n_test_predictions': len(test_predictions),
                'group_results': group_results,
                'test_predictions': test_predictions
            })
            
            if verbose:
                print(f"    ✅ Fold completed: {len(test_predictions)} test predictions")
        
        except Exception as e:
            if verbose:
                print(f"    ❌ Fold failed: {e}")
            cv_results.append({
                'fold': fold_idx,
                'success': False,
                'error': str(e)
            })
    
    # Analyze cross-validation results
    successful_folds = [r for r in cv_results if r['success']]
    
    if verbose:
        print(f"\n📊 Cross-validation Summary:")
        print(f"  Successful folds: {len(successful_folds)}/{k_folds}")
    
    return {
        'k_folds': k_folds,
        'successful_folds': len(successful_folds),
        'total_folds': k_folds,
        'cv_results': cv_results,
        'success_rate': len(successful_folds) / k_folds
    }


def perform_sensitivity_analysis(df_intervals: pd.DataFrame,
                                fit_aft_models_func,
                                perform_group_analysis_func,
                                confidence_levels_variants: List[List[float]] = None,
                                aft_model_variants: List[str] = ['Weibull', 'LogLogistic'],
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Perform sensitivity analysis across different parameters.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        fit_aft_models_func: Function to fit AFT models
        perform_group_analysis_func: Function for group analysis
        confidence_levels_variants: List of confidence level combinations to test
        aft_model_variants: List of AFT model types to compare
        verbose: Whether to print progress
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    if verbose:
        print("🔬 Performing sensitivity analysis...")
    
    if confidence_levels_variants is None:
        confidence_levels_variants = [
            [0.85, 0.90, 0.95],
            [0.90, 0.95],
            [0.80, 0.90, 0.95]
        ]
    
    # Prepare data
    df_X = df_intervals.copy()
    bmi_mean = df_X['bmi'].mean()
    bmi_std = df_X['bmi'].std()
    df_X['bmi_z'] = (df_X['bmi'] - bmi_mean) / bmi_std
    
    sensitivity_results = {}
    
    # Test different confidence levels
    if verbose:
        print("\n  🎯 Testing different confidence levels...")
    
    for i, conf_levels in enumerate(confidence_levels_variants):
        conf_name = f"confidence_variant_{i+1}"
        
        if verbose:
            conf_str = [f"{int(c*100)}%" for c in conf_levels]
            print(f"    Variant {i+1}: {conf_str}")
        
        try:
            # Fit primary model
            primary_model, primary_name, all_models = fit_aft_models_func(
                df_X, formula='~ bmi_z', verbose=False
            )
            
            if primary_model is not None:
                # Perform group analysis
                group_results = perform_group_analysis_func(
                    df_intervals,
                    primary_model,
                    confidence_levels=conf_levels,
                    time_grid=np.linspace(10, 25, 100),
                    grouping_methods=['clinical'],
                    verbose=False
                )
                
                sensitivity_results[conf_name] = {
                    'confidence_levels': conf_levels,
                    'group_results': group_results,
                    'success': True
                }
                
                if verbose:
                    print(f"      ✅ Success")
            else:
                if verbose:
                    print(f"      ❌ Model fitting failed")
                sensitivity_results[conf_name] = {'success': False, 'error': 'Model fitting failed'}
        
        except Exception as e:
            if verbose:
                print(f"      ❌ Error: {e}")
            sensitivity_results[conf_name] = {'success': False, 'error': str(e)}
    
    # Test different AFT distributions (if multiple models available)
    if verbose:
        print(f"\n  📈 Testing AFT model robustness...")
    
    try:
        primary_model, primary_name, all_models = fit_aft_models_func(
            df_X, formula='~ bmi_z', verbose=False
        )
        
        model_comparison = {}
        
        for model_name, model_result in all_models.items():
            if model_result.fit_success:
                model_comparison[model_name] = {
                    'aic': model_result.aic,
                    'log_likelihood': model_result.log_likelihood,
                    'bmi_coefficient': model_result.bmi_coefficient,
                    'bmi_pvalue': model_result.bmi_pvalue
                }
                
                if verbose:
                    print(f"    {model_name}: AIC = {model_result.aic:.2f}")
        
        sensitivity_results['model_comparison'] = model_comparison
        
    except Exception as e:
        if verbose:
            print(f"    ❌ Model comparison failed: {e}")
        sensitivity_results['model_comparison'] = {'error': str(e)}
    
    if verbose:
        print(f"\n✅ Sensitivity analysis completed")
    
    return sensitivity_results


def create_final_policy_table(group_analysis_results: Dict[str, Any],
                             mc_results: Optional[Dict[str, Any]] = None,
                             cv_results: Optional[Dict[str, Any]] = None,
                             confidence_levels: List[float] = [0.90, 0.95],
                             verbose: bool = True) -> pd.DataFrame:
    """
    Create comprehensive final policy recommendation table.
    
    Args:
        group_analysis_results: Results from group-specific analysis
        mc_results: Monte Carlo robustness results (optional)
        cv_results: Cross-validation results (optional)
        confidence_levels: List of confidence levels
        verbose: Whether to print the table
        
    Returns:
        DataFrame with final policy recommendations
    """
    if verbose:
        print("📋 Creating final policy recommendation table...")
    
    group_results = group_analysis_results['group_results']
    
    policy_data = []
    
    for group_name, group_data in group_results.items():
        if 'optimal_weeks' not in group_data:
            continue
        
        row = {
            'BMI_Range': group_name,
            'n_mothers': group_data['n_mothers'],
            'representative_BMI': group_data['representative_bmi'],
            'bmi_range': group_data['bmi_range']
        }
        
        # Add AFT optimal weeks
        for conf in confidence_levels:
            if conf in group_data['optimal_weeks']:
                week = group_data['optimal_weeks'][conf]
                week_str = f"{week:.1f}" if week != np.inf else "Never"
                row[f'optimal_week_{int(conf*100)}'] = week_str
                
                # Calculate threshold probability at optimal week
                if 'survival_curve' in group_data and week != np.inf:
                    times = group_data['survival_curve']['times']
                    survival = group_data['survival_curve']['survival']
                    
                    # Find closest time point
                    week_idx = np.argmin(np.abs(times - week))
                    surv_prob = survival[week_idx]
                    threshold_prob = 1 - surv_prob
                    row[f'threshold_prob_at_optimal_{int(conf*100)}'] = f"{threshold_prob:.3f}"
                else:
                    row[f'threshold_prob_at_optimal_{int(conf*100)}'] = "N/A"
        
        # Add Monte Carlo confidence intervals if available
        if mc_results and 'analysis' in mc_results:
            mc_summaries = mc_results['analysis']['group_summaries']
            
            if group_name in mc_summaries:
                for conf in confidence_levels:
                    if conf in mc_summaries[group_name]:
                        mc_summary = mc_summaries[group_name][conf]
                        
                        if 'ci_95_lower' in mc_summary and 'ci_95_upper' in mc_summary:
                            ci_low = mc_summary['ci_95_lower']
                            ci_high = mc_summary['ci_95_upper']
                            row[f'mc_ci_{int(conf*100)}'] = f"[{ci_low:.1f}, {ci_high:.1f}]"
                        else:
                            row[f'mc_ci_{int(conf*100)}'] = "N/A"
        
        # Add validation metrics if available
        if cv_results and cv_results['success_rate'] > 0:
            row['cv_success_rate'] = f"{cv_results['success_rate']:.2f}"
        
        policy_data.append(row)
    
    policy_df = pd.DataFrame(policy_data)
    
    # Add metadata
    metadata = {
        'analysis_method': group_analysis_results.get('best_grouping_method', 'Clinical'),
        'confidence_levels': confidence_levels,
        'total_mothers': policy_df['n_mothers'].sum() if not policy_df.empty else 0,
        'n_groups': len(policy_df)
    }
    
    if verbose and not policy_df.empty:
        print("🎯 Final Policy Recommendation Table:")
        print("="*100)
        print(policy_df.to_string(index=False))
        
        print(f"\n📊 Policy Summary:")
        print(f"  Analysis method: {metadata['analysis_method']}")
        print(f"  Total mothers: {metadata['total_mothers']}")
        print(f"  BMI groups: {metadata['n_groups']}")
        print(f"  Confidence levels: {[f'{int(c*100)}%' for c in confidence_levels]}")
        
        if mc_results:
            overall_stability = mc_results['analysis']['stability_metrics']['overall_stability']
            print(f"  Monte Carlo stability: {overall_stability}")
        
        if cv_results:
            print(f"  Cross-validation success rate: {cv_results['success_rate']:.2f}")
    
    # Store metadata as attributes
    for key, value in metadata.items():
        policy_df.attrs[key] = value
    
    return policy_df


def export_final_results(policy_df: pd.DataFrame,
                        output_path: Optional[Path] = None,
                        verbose: bool = True) -> Optional[Path]:
    """
    Export final policy table to CSV file.
    
    Args:
        policy_df: Final policy recommendation DataFrame
        output_path: Output file path (optional)
        verbose: Whether to print export details
        
    Returns:
        Path to exported file or None if not exported
    """
    if output_path is None:
        # Use default path in proper output directory
        output_path = RESULTS_DIR / "prob2_policy_recommendations.csv"
    
    output_path = Path(output_path)
    
    try:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        policy_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"💾 Final policy table exported to: {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return output_path
        
    except Exception as e:
        if verbose:
            print(f"❌ Export failed: {e}")
        return None


def run_comprehensive_validation(df_intervals: pd.DataFrame,
                                group_analysis_results: Dict[str, Any],
                                construct_intervals_func,
                                fit_aft_models_func,
                                perform_group_analysis_func,
                                mc_results: Optional[Dict[str, Any]] = None,
                                k_folds: int = 5,
                                confidence_levels: List[float] = [0.90, 0.95],
                                export_results: bool = True,
                                output_path: Optional[Path] = None,
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive validation and generate final policy table.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        group_analysis_results: Results from group-specific analysis
        construct_intervals_func: Function to construct intervals
        fit_aft_models_func: Function to fit AFT models
        perform_group_analysis_func: Function for group analysis
        mc_results: Monte Carlo results (optional)
        k_folds: Number of cross-validation folds
        confidence_levels: List of confidence levels
        export_results: Whether to export final table to CSV
        output_path: Output file path for export
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with comprehensive validation results
    """
    if verbose:
        print("🏁 Running comprehensive validation and final policy generation...")
    
    # Step 1: Cross-validation analysis
    cv_results = perform_cross_validation_analysis(
        df_intervals,
        construct_intervals_func,
        fit_aft_models_func,
        perform_group_analysis_func,
        k_folds=k_folds,
        confidence_levels=confidence_levels,
        verbose=verbose
    )
    
    # Step 2: Sensitivity analysis
    sensitivity_results = perform_sensitivity_analysis(
        df_intervals,
        fit_aft_models_func,
        perform_group_analysis_func,
        verbose=verbose
    )
    
    # Step 3: Create final policy table
    policy_df = create_final_policy_table(
        group_analysis_results,
        mc_results=mc_results,
        cv_results=cv_results,
        confidence_levels=confidence_levels,
        verbose=verbose
    )
    
    # Step 4: Export results if requested
    exported_path = None
    if export_results and not policy_df.empty:
        exported_path = export_final_results(
            policy_df,
            output_path=output_path,
            verbose=verbose
        )
    
    if verbose:
        print(f"\n✅ Comprehensive validation completed")
        print(f"  Cross-validation success: {cv_results['success_rate']:.2f}")
        print(f"  Final policy groups: {len(policy_df)}")
        if exported_path:
            print(f"  Results exported to: {exported_path}")
    
    return {
        'cv_results': cv_results,
        'sensitivity_results': sensitivity_results,
        'policy_table': policy_df,
        'exported_path': exported_path,
        'validation_summary': {
            'cv_success_rate': cv_results['success_rate'],
            'n_policy_groups': len(policy_df),
            'total_mothers': policy_df['n_mothers'].sum() if not policy_df.empty else 0,
            'confidence_levels': confidence_levels
        }
    }
