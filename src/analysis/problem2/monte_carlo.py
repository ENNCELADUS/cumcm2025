"""
Monte Carlo robustness testing for Problem 2.

This module provides Monte Carlo simulation functions to assess the robustness
of AFT model results to Y-chromosome concentration measurement errors.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def add_measurement_noise(df_tests: pd.DataFrame, sigma_error: float = 0.002, 
                         random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Add normally distributed measurement error to Y-chromosome concentrations.
    
    Args:
        df_tests: DataFrame with Y-chromosome test data
        sigma_error: Standard deviation of measurement error (default 0.002 = 0.2%)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with noisy Y-chromosome concentrations
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    df_noisy = df_tests.copy()
    n_tests = len(df_noisy)
    
    # Add Gaussian noise: y_observed = y_true + epsilon
    noise = np.random.normal(0, sigma_error, n_tests)
    df_noisy['y_concentration'] += noise
    
    # Ensure concentrations remain non-negative
    df_noisy['y_concentration'] = np.maximum(df_noisy['y_concentration'], 0.0)
    
    return df_noisy


def run_single_monte_carlo_iteration(args: Tuple) -> Dict[str, Any]:
    """
    Run a single Monte Carlo iteration.
    
    Args:
        args: Tuple containing (iteration_idx, df_tests, sigma_error, random_seed, 
                               construct_intervals_func, fit_aft_models_func, 
                               perform_group_analysis_func, confidence_levels)
    
    Returns:
        Dictionary with iteration results
    """
    (iteration_idx, df_tests, sigma_error, random_seed, 
     construct_intervals_func, fit_aft_models_func, 
     perform_group_analysis_func, confidence_levels) = args
    
    try:
        # Add measurement noise
        df_noisy = add_measurement_noise(df_tests, sigma_error, random_seed + iteration_idx)
        
        # Reconstruct intervals with noisy data
        df_intervals_sim = construct_intervals_func(df_noisy, verbose=False)
        
        # Prepare feature matrix (simplified version)
        bmi_mean = df_intervals_sim['bmi'].mean()
        bmi_std = df_intervals_sim['bmi'].std()
        df_intervals_sim['bmi_z'] = (df_intervals_sim['bmi'] - bmi_mean) / bmi_std
        
        # Refit AFT model
        primary_model, primary_name, all_models = fit_aft_models_func(
            df_intervals_sim, formula='~ bmi_z', verbose=False
        )
        
        if primary_model is None:
            return {
                'iteration': iteration_idx,
                'success': False,
                'error': 'AFT model fitting failed'
            }
        
        # Perform group-specific analysis
        time_grid = np.linspace(10, 25, 100)
        group_results = perform_group_analysis_func(
            df_intervals_sim, 
            primary_model,
            confidence_levels=confidence_levels,
            time_grid=time_grid,
            grouping_methods=['clinical', 'tertile'],
            verbose=False
        )
        
        # Extract key results
        group_optimal_weeks = {}
        for group_name, group_data in group_results['group_results'].items():
            if 'optimal_weeks' in group_data:
                group_optimal_weeks[group_name] = group_data['optimal_weeks']
        
        return {
            'iteration': iteration_idx,
            'success': True,
            'model_name': primary_name,
            'n_mothers': len(df_intervals_sim),
            'group_optimal_weeks': group_optimal_weeks,
            'best_grouping_method': group_results['best_grouping_method']
        }
        
    except Exception as e:
        return {
            'iteration': iteration_idx,
            'success': False,
            'error': str(e)
        }


def run_monte_carlo_robustness_test(df_tests: pd.DataFrame,
                                   construct_intervals_func,
                                   fit_aft_models_func,
                                   perform_group_analysis_func,
                                   n_simulations: int = 300,
                                   sigma_error: float = 0.002,
                                   confidence_levels: List[float] = [0.90, 0.95],
                                   random_seed: int = 42,
                                   n_workers: Optional[int] = None,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run Monte Carlo robustness testing with measurement error.
    
    Args:
        df_tests: Original test data
        construct_intervals_func: Function to construct intervals
        fit_aft_models_func: Function to fit AFT models
        perform_group_analysis_func: Function for group-specific analysis
        n_simulations: Number of Monte Carlo iterations (300+ recommended for smooth CI)
        sigma_error: Standard deviation of measurement error
        confidence_levels: List of confidence levels
        random_seed: Random seed for reproducibility
        n_workers: Number of parallel workers (None for single-threaded)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with Monte Carlo results and summary statistics
    """
    if verbose:
        print(f"🎲 Running Monte Carlo robustness test...")
        print(f"  Simulations: {n_simulations}")
        print(f"  Measurement error σ: {sigma_error:.4f} ({100*sigma_error:.2f}%)")
        print(f"  Confidence levels: {[f'{int(c*100)}%' for c in confidence_levels]}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i in range(n_simulations):
        args_list.append((
            i, df_tests, sigma_error, random_seed, 
            construct_intervals_func, fit_aft_models_func, 
            perform_group_analysis_func, confidence_levels
        ))
    
    # Run simulations
    mc_results = []
    
    if n_workers is None or n_workers == 1:
        # Single-threaded execution
        if verbose:
            print("🔄 Running simulations (single-threaded)...")
        
        for i, args in enumerate(args_list):
            if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                print(f"  Progress: {i + 1}/{n_simulations} ({100*(i+1)/n_simulations:.1f}%)")
            
            result = run_single_monte_carlo_iteration(args)
            mc_results.append(result)
    
    else:
        # Parallel execution
        if verbose:
            print(f"🔄 Running simulations (parallel with {n_workers} workers)...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_iteration = {
                executor.submit(run_single_monte_carlo_iteration, args): args[0] 
                for args in args_list
            }
            
            completed = 0
            for future in as_completed(future_to_iteration):
                result = future.result()
                mc_results.append(result)
                completed += 1
                
                if verbose and completed % max(1, n_simulations // 10) == 0:
                    print(f"  Progress: {completed}/{n_simulations} ({100*completed/n_simulations:.1f}%)")
    
    # Sort results by iteration number
    mc_results.sort(key=lambda x: x['iteration'])
    
    # Analyze results
    if verbose:
        print("📊 Analyzing Monte Carlo results...")
    
    analysis = analyze_monte_carlo_results(mc_results, confidence_levels, verbose=verbose)
    
    return {
        'parameters': {
            'n_simulations': n_simulations,
            'sigma_error': sigma_error,
            'confidence_levels': confidence_levels,
            'random_seed': random_seed
        },
        'raw_results': mc_results,
        'analysis': analysis
    }


def analyze_monte_carlo_results(mc_results: List[Dict], 
                               confidence_levels: List[float],
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze Monte Carlo simulation results.
    
    Args:
        mc_results: List of Monte Carlo iteration results
        confidence_levels: List of confidence levels used
        verbose: Whether to print analysis results
        
    Returns:
        Dictionary with analysis results
    """
    # Filter successful iterations
    successful_results = [r for r in mc_results if r['success']]
    n_successful = len(successful_results)
    n_total = len(mc_results)
    
    if verbose:
        print(f"✅ Monte Carlo Analysis:")
        print(f"  Successful iterations: {n_successful}/{n_total} ({100*n_successful/n_total:.1f}%)")
    
    if n_successful == 0:
        if verbose:
            print("❌ No successful iterations to analyze")
        return {'success_rate': 0, 'error': 'No successful iterations'}
    
    # Collect group-specific optimal weeks across iterations
    group_weeks_collection = {}
    
    for result in successful_results:
        for group_name, optimal_weeks in result['group_optimal_weeks'].items():
            if group_name not in group_weeks_collection:
                group_weeks_collection[group_name] = {conf: [] for conf in confidence_levels}
            
            for conf in confidence_levels:
                if conf in optimal_weeks:
                    week = optimal_weeks[conf]
                    if week != np.inf:  # Only include finite weeks
                        group_weeks_collection[group_name][conf].append(week)
    
    # Calculate summary statistics for each group and confidence level
    group_summaries = {}
    
    for group_name, weeks_by_conf in group_weeks_collection.items():
        if verbose:
            print(f"\n  📊 Group: {group_name}")
        
        group_summary = {}
        
        for conf in confidence_levels:
            weeks = weeks_by_conf[conf]
            
            if len(weeks) > 0:
                weeks_array = np.array(weeks)
                summary = {
                    'n_samples': len(weeks),
                    'mean': np.mean(weeks_array),
                    'std': np.std(weeks_array),
                    'median': np.median(weeks_array),
                    'q25': np.percentile(weeks_array, 25),
                    'q75': np.percentile(weeks_array, 75),
                    'min': np.min(weeks_array),
                    'max': np.max(weeks_array),
                    'cv': np.std(weeks_array) / np.mean(weeks_array) if np.mean(weeks_array) > 0 else np.inf
                }
                
                # 95% confidence interval
                if len(weeks) >= 3:
                    ci_lower = np.percentile(weeks_array, 2.5)
                    ci_upper = np.percentile(weeks_array, 97.5)
                    summary['ci_95_lower'] = ci_lower
                    summary['ci_95_upper'] = ci_upper
                else:
                    summary['ci_95_lower'] = summary['min']
                    summary['ci_95_upper'] = summary['max']
                
                group_summary[conf] = summary
                
                if verbose:
                    conf_pct = int(conf * 100)
                    print(f"    {conf_pct}% confidence:")
                    print(f"      Mean: {summary['mean']:.2f} ± {summary['std']:.2f} weeks")
                    print(f"      Median: {summary['median']:.2f} weeks")
                    print(f"      95% CI: [{summary['ci_95_lower']:.2f}, {summary['ci_95_upper']:.2f}]")
                    print(f"      CV: {summary['cv']:.3f}")
            
            else:
                if verbose:
                    print(f"    {int(conf*100)}% confidence: No valid samples")
                group_summary[conf] = {'n_samples': 0, 'error': 'No valid samples'}
        
        group_summaries[group_name] = group_summary
    
    # Overall stability assessment
    if verbose:
        print(f"\n🎯 Robustness Assessment:")
    
    stability_metrics = assess_stability(group_summaries, verbose=verbose)
    
    return {
        'success_rate': n_successful / n_total,
        'n_successful': n_successful,
        'n_total': n_total,
        'group_summaries': group_summaries,
        'stability_metrics': stability_metrics
    }


def assess_stability(group_summaries: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Assess the stability of group-specific optimal weeks across Monte Carlo iterations.
    
    Args:
        group_summaries: Summary statistics for each group
        verbose: Whether to print assessment results
        
    Returns:
        Dictionary with stability metrics
    """
    stability_thresholds = {
        'cv_excellent': 0.05,    # CV < 5% = excellent stability
        'cv_good': 0.10,         # CV < 10% = good stability  
        'cv_moderate': 0.20,     # CV < 20% = moderate stability
        'ci_width_good': 2.0,    # CI width < 2 weeks = good precision
        'ci_width_moderate': 4.0 # CI width < 4 weeks = moderate precision
    }
    
    stability_assessment = {}
    
    for group_name, group_data in group_summaries.items():
        group_stability = {}
        
        for conf, summary in group_data.items():
            if isinstance(summary, dict) and 'cv' in summary:
                cv = summary['cv']
                ci_width = summary.get('ci_95_upper', 0) - summary.get('ci_95_lower', 0)
                
                # CV-based stability assessment
                if cv <= stability_thresholds['cv_excellent']:
                    cv_stability = 'Excellent'
                elif cv <= stability_thresholds['cv_good']:
                    cv_stability = 'Good'
                elif cv <= stability_thresholds['cv_moderate']:
                    cv_stability = 'Moderate'
                else:
                    cv_stability = 'Poor'
                
                # CI width-based precision assessment
                if ci_width <= stability_thresholds['ci_width_good']:
                    ci_precision = 'Good'
                elif ci_width <= stability_thresholds['ci_width_moderate']:
                    ci_precision = 'Moderate'
                else:
                    ci_precision = 'Poor'
                
                group_stability[conf] = {
                    'cv': cv,
                    'cv_stability': cv_stability,
                    'ci_width': ci_width,
                    'ci_precision': ci_precision,
                    'overall_rating': 'Good' if cv_stability in ['Excellent', 'Good'] and ci_precision in ['Good', 'Moderate'] else 'Moderate' if cv_stability == 'Moderate' or ci_precision == 'Moderate' else 'Poor'
                }
                
                if verbose:
                    conf_pct = int(conf * 100)
                    print(f"  {group_name} ({conf_pct}%): {group_stability[conf]['overall_rating']} stability")
                    print(f"    CV: {cv:.3f} ({cv_stability})")
                    print(f"    95% CI width: {ci_width:.2f} weeks ({ci_precision})")
        
        stability_assessment[group_name] = group_stability
    
    # Overall assessment
    all_ratings = []
    for group_data in stability_assessment.values():
        for conf_data in group_data.values():
            if 'overall_rating' in conf_data:
                all_ratings.append(conf_data['overall_rating'])
    
    if all_ratings:
        rating_counts = {rating: all_ratings.count(rating) for rating in ['Good', 'Moderate', 'Poor']}
        overall_stability = max(rating_counts.keys(), key=lambda x: rating_counts[x])
        
        if verbose:
            print(f"\n  Overall Stability: {overall_stability}")
            print(f"    Good: {rating_counts.get('Good', 0)}")
            print(f"    Moderate: {rating_counts.get('Moderate', 0)}")
            print(f"    Poor: {rating_counts.get('Poor', 0)}")
    else:
        overall_stability = 'Unknown'
    
    return {
        'group_stability': stability_assessment,
        'overall_stability': overall_stability,
        'thresholds': stability_thresholds
    }


def create_monte_carlo_summary_table(mc_analysis_results: Dict[str, Any], 
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Create a summary table of Monte Carlo robustness results.
    
    Args:
        mc_analysis_results: Results from run_monte_carlo_robustness_test()
        verbose: Whether to print the summary table
        
    Returns:
        DataFrame with Monte Carlo summary
    """
    group_summaries = mc_analysis_results['analysis']['group_summaries']
    confidence_levels = mc_analysis_results['parameters']['confidence_levels']
    
    summary_data = []
    
    for group_name, group_data in group_summaries.items():
        row = {'BMI_Group': group_name}
        
        for conf in confidence_levels:
            if conf in group_data and 'mean' in group_data[conf]:
                summary = group_data[conf]
                conf_pct = int(conf * 100)
                
                # Mean ± std
                row[f'optimal_week_{conf_pct}_mean'] = f"{summary['mean']:.2f}"
                row[f'optimal_week_{conf_pct}_std'] = f"±{summary['std']:.2f}"
                
                # 95% CI
                ci_lower = summary.get('ci_95_lower', summary.get('min', 0))
                ci_upper = summary.get('ci_95_upper', summary.get('max', 0))
                row[f'optimal_week_{conf_pct}_ci'] = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
                
                # Stability rating
                stability = mc_analysis_results['analysis']['stability_metrics']['group_stability']
                if group_name in stability and conf in stability[group_name]:
                    row[f'stability_{conf_pct}'] = stability[group_name][conf]['overall_rating']
                else:
                    row[f'stability_{conf_pct}'] = 'Unknown'
            else:
                conf_pct = int(conf * 100)
                row[f'optimal_week_{conf_pct}_mean'] = 'N/A'
                row[f'optimal_week_{conf_pct}_std'] = 'N/A'
                row[f'optimal_week_{conf_pct}_ci'] = 'N/A'
                row[f'stability_{conf_pct}'] = 'N/A'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose and not summary_df.empty:
        print("📋 Monte Carlo Robustness Summary:")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Overall summary
        parameters = mc_analysis_results['parameters']
        analysis = mc_analysis_results['analysis']
        
        print(f"\n📊 Monte Carlo Test Summary:")
        print(f"  Measurement error σ: {parameters['sigma_error']:.4f} ({100*parameters['sigma_error']:.2f}%)")
        print(f"  Simulations: {parameters['n_simulations']}")
        print(f"  Success rate: {100*analysis['success_rate']:.1f}%")
        print(f"  Overall stability: {analysis['stability_metrics']['overall_stability']}")
    
    return summary_df
