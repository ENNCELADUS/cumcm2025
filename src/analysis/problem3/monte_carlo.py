"""
Enhanced Monte Carlo robustness testing for Problem 3.

This module implements mandatory 300-run Monte Carlo sensitivity analysis
with per-group reporting and robustness assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path

# Import base Monte Carlo from Problem 2
from ..problem2.monte_carlo import add_measurement_noise
from .data_preprocessing import prepare_extended_feature_matrix, standardize_covariates_extended
from .survival_analysis import fit_aft_model_extended  
from .bmi_grouping import compute_group_survival_extended, calculate_group_optimal_weeks


def run_enhanced_monte_carlo(df_original: pd.DataFrame,
                           selected_covariates: List[str],
                           n_simulations: int = 300,
                           sigma_y: float = 0.002,
                           confidence_levels: List[float] = [0.90, 0.95],
                           parallel: bool = True,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Run 300-replicate Monte Carlo with per-group reporting (mandatory for Problem 3).
    
    Args:
        df_original: Original test data with individual measurements
        selected_covariates: VIF-approved covariate list
        n_simulations: Number of MC simulations (default: 300)
        sigma_y: Measurement error standard deviation
        confidence_levels: Confidence levels for optimal weeks
        parallel: Whether to use parallel processing
        random_state: Random seed for reproducibility
        
    Returns:
        Comprehensive MC results with per-group distributions
    """
    print(f"🚀 Starting Enhanced Monte Carlo Analysis ({n_simulations} simulations)")
    print(f"   Measurement error: σ_Y = {sigma_y}")
    print(f"   Confidence levels: {confidence_levels}")
    print(f"   Selected covariates: {selected_covariates}")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Initialize results storage
    mc_results = {
        'group_optimal_weeks': [],
        'group_contrasts': [],  
        'model_metadata': [],
        'simulation_params': {
            'n_simulations': n_simulations,
            'sigma_y': sigma_y,
            'confidence_levels': confidence_levels,
            'selected_covariates': selected_covariates,
            'random_state': random_state
        }
    }
    
    if parallel and n_simulations > 10:
        # Parallel execution for large simulations
        mc_results = _run_parallel_monte_carlo(
            df_original, selected_covariates, n_simulations, 
            sigma_y, confidence_levels, random_state
        )
    else:
        # Sequential execution
        mc_results = _run_sequential_monte_carlo(
            df_original, selected_covariates, n_simulations,
            sigma_y, confidence_levels, random_state
        )
    
    print(f"✅ Monte Carlo Analysis Complete")
    print(f"   Successful simulations: {len(mc_results['group_optimal_weeks'])}")
    
    return mc_results


def _run_sequential_monte_carlo(df_original: pd.DataFrame,
                              selected_covariates: List[str],
                              n_simulations: int,
                              sigma_y: float,
                              confidence_levels: List[float],
                              random_state: int) -> Dict[str, Any]:
    """Run Monte Carlo simulations sequentially."""
    
    from .data_preprocessing import construct_intervals_extended
    from ..problem2.bmi_grouping import BMIGrouper
    
    mc_results = {
        'group_optimal_weeks': [],
        'group_contrasts': [],
        'model_metadata': [],
        'simulation_params': {
            'n_simulations': n_simulations,
            'sigma_y': sigma_y,
            'confidence_levels': confidence_levels,
            'selected_covariates': selected_covariates,
            'random_state': random_state
        }
    }
    
    # Initialize BMI grouper (reuse Problem 2 grouping)
    grouper = BMIGrouper()
    
    successful_sims = 0
    
    for sim in range(n_simulations):
        try:
            # Progress reporting
            if (sim + 1) % 50 == 0:
                print(f"   Progress: {sim + 1}/{n_simulations} simulations...")
            
            # Step 1: Add measurement noise
            df_noisy = add_measurement_noise(df_original, sigma_error=sigma_y, random_seed=random_state + sim)
            
            # Step 2: Reconstruct intervals  
            df_intervals_sim = construct_intervals_extended(df_noisy)
            
            # Step 3: Extended preprocessing
            df_standardized, _ = standardize_covariates_extended(df_intervals_sim)
            
            # Step 4: Add BMI grouping (identical to Problem 2)
            df_intervals_sim = grouper.get_all_groupings(df_standardized)
            
            # Step 5: Prepare extended feature matrix
            df_X_sim = prepare_extended_feature_matrix(df_intervals_sim, selected_covariates)
            
            # Step 6: Fit extended AFT model
            model_results = fit_aft_model_extended(df_X_sim, selected_covariates, test_nonlinearity=False)
            
            if model_results.get('best_model') is None:
                warnings.warn(f"Simulation {sim}: Model fitting failed")
                continue
            
            best_model = model_results['best_model']['model']
            
            # Step 7: Group survival analysis
            time_grid = np.linspace(10, 25, 100)
            
            # Extract BMI groups from the dataframe
            from .bmi_grouping import create_enhanced_bmi_groups
            bmi_groups, _ = create_enhanced_bmi_groups(df_intervals_sim, method='clinical', verbose=False)
            
            group_survival_funcs = compute_group_survival_extended(
                bmi_groups, df_X_sim, best_model, selected_covariates, time_grid, verbose=False
            )
            
            # Step 8: Calculate group optimal weeks
            optimal_weeks_sim = calculate_group_optimal_weeks(
                group_survival_funcs, time_grid, confidence_levels
            )
            
            # Step 9: Compute group contrasts
            from .bmi_grouping import compute_group_contrasts
            contrasts_sim = compute_group_contrasts(optimal_weeks_sim)
            
            # Store results
            mc_results['group_optimal_weeks'].append(optimal_weeks_sim)
            mc_results['group_contrasts'].append(contrasts_sim)
            mc_results['model_metadata'].append({
                'sim_id': sim,
                'converged': True,
                'aic': model_results['best_model'].get('aic', np.nan),
                'model_key': model_results['best_model'].get('model_key', 'unknown'),
                'n_observations': len(df_X_sim)
            })
            
            successful_sims += 1
            
        except Exception as e:
            warnings.warn(f"Simulation {sim} failed: {e}")
            mc_results['model_metadata'].append({
                'sim_id': sim,
                'converged': False,
                'error': str(e)
            })
            continue
    
    print(f"   Monte Carlo completed: {successful_sims}/{n_simulations} successful")
    
    return mc_results


def _run_parallel_monte_carlo(df_original: pd.DataFrame,
                            selected_covariates: List[str], 
                            n_simulations: int,
                            sigma_y: float,
                            confidence_levels: List[float],
                            random_state: int) -> Dict[str, Any]:
    """Run Monte Carlo simulations in parallel (simplified for now)."""
    # For now, fall back to sequential to avoid complex multiprocessing setup
    return _run_sequential_monte_carlo(
        df_original, selected_covariates, n_simulations, 
        sigma_y, confidence_levels, random_state
    )


def summarize_monte_carlo_per_group(mc_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Summarize Monte Carlo results with per-group distributions and robustness assessment.
    
    Args:
        mc_results: Results from run_enhanced_monte_carlo
        
    Returns:
        Per-group summary statistics with robustness labels
    """
    print("📊 Summarizing Monte Carlo Results per Group...")
    
    if not mc_results['group_optimal_weeks']:
        warnings.warn("No successful Monte Carlo simulations to summarize")
        return {}
    
    # Extract structure from first successful simulation
    first_result = mc_results['group_optimal_weeks'][0]
    
    summary = {}
    
    # The actual data structure is: {group_name: {tau_key: value}}
    # Not: {method_name: {group_name: {tau_key: value}}}
    
    for group_name, group_taus in first_result.items():
        summary[group_name] = {}
        
        # Get all tau levels for this group
        tau_levels = list(group_taus.keys())
        
        for tau_key in tau_levels:
            # Collect optimal weeks across all successful simulations
            weeks = []
            for result in mc_results['group_optimal_weeks']:
                try:
                    week = result[group_name][tau_key]
                    if not np.isinf(week) and not np.isnan(week):
                        weeks.append(week)
                except KeyError:
                    continue
            
            if weeks:
                weeks = np.array(weeks)
                
                summary[group_name][tau_key] = {
                    'n_simulations': len(weeks),
                    'mean': np.mean(weeks),
                    'std': np.std(weeks),
                    'median': np.median(weeks),
                    'ci_2.5': np.percentile(weeks, 2.5),
                    'ci_97.5': np.percentile(weeks, 97.5),
                    'ci_width': np.percentile(weeks, 97.5) - np.percentile(weeks, 2.5),
                    'robustness_label': assess_robustness(weeks),
                    'raw_weeks': weeks.tolist()  # Store for detailed analysis
                }
            else:
                summary[group_name][tau_key] = {
                    'n_simulations': 0,
                    'robustness_label': 'insufficient_data'
                }
    
    # Print summary
    for group_name, group_summary in summary.items():
        print(f"\n📊 Group {group_name}:")
        for tau_key, stats in group_summary.items():
            if stats.get('n_simulations', 0) > 0:
                mean_val = stats['mean']
                ci_low = stats['ci_2.5']
                ci_high = stats['ci_97.5']
                robustness = stats['robustness_label']
                print(f"    {tau_key}: {mean_val:.1f} (95% CI: {ci_low:.1f}-{ci_high:.1f}) [{robustness}]")
    
    return summary


def assess_robustness(weeks: np.ndarray, 
                     clinical_cutoffs: List[float] = [0.5, 1.0, 2.0]) -> str:
    """
    Assign robustness label based on CI width and stability criteria.
    
    Args:
        weeks: Array of optimal weeks from MC simulations
        clinical_cutoffs: Clinical significance thresholds
        
    Returns:
        Robustness label: 'high', 'medium', 'low', or 'unstable'
    """
    if len(weeks) < 10:
        return 'insufficient_data'
    
    ci_width = np.percentile(weeks, 97.5) - np.percentile(weeks, 2.5)
    cv_percent = (np.std(weeks) / np.mean(weeks)) * 100
    
    # Stability criteria
    stable_ci = ci_width <= clinical_cutoffs[1]  # CI width <= 1 week
    stable_cv = cv_percent <= 10  # CV <= 10%
    
    # Assess crossing clinical cutoffs
    clinical_boundaries = [12, 16, 20]  # Example clinical decision boundaries
    crosses_boundaries = 0
    
    for boundary in clinical_boundaries:
        crosses = (np.percentile(weeks, 2.5) < boundary < np.percentile(weeks, 97.5))
        if crosses:
            crosses_boundaries += 1
    
    # Robustness classification
    if stable_ci and stable_cv and crosses_boundaries == 0:
        return 'high'
    elif (stable_ci or stable_cv) and crosses_boundaries <= 1:
        return 'medium'
    elif crosses_boundaries <= 2:
        return 'low'
    else:
        return 'unstable'


def analyze_monte_carlo_convergence(mc_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze Monte Carlo convergence properties and simulation quality.
    
    Args:
        mc_results: Results from run_enhanced_monte_carlo
        
    Returns:
        Convergence analysis results
    """
    convergence_analysis = {
        'simulation_success_rate': 0,
        'model_convergence_rate': 0,
        'aic_stability': {},
        'parameter_stability': {},
        'recommendation': ''
    }
    
    # Basic success rates
    total_sims = len(mc_results['model_metadata'])
    successful_sims = sum(1 for meta in mc_results['model_metadata'] if meta.get('converged', False))
    
    convergence_analysis['simulation_success_rate'] = successful_sims / total_sims if total_sims > 0 else 0
    convergence_analysis['model_convergence_rate'] = successful_sims / total_sims if total_sims > 0 else 0
    
    # AIC stability
    aics = [meta.get('aic', np.nan) for meta in mc_results['model_metadata'] if meta.get('converged', False)]
    if aics:
        aics = [aic for aic in aics if not np.isnan(aic)]
        if aics:
            convergence_analysis['aic_stability'] = {
                'mean': np.mean(aics),
                'std': np.std(aics),
                'cv_percent': (np.std(aics) / np.mean(aics)) * 100,
                'range': (min(aics), max(aics))
            }
    
    # Recommendations
    success_rate = convergence_analysis['simulation_success_rate']
    if success_rate >= 0.95:
        convergence_analysis['recommendation'] = 'excellent_convergence'
    elif success_rate >= 0.90:
        convergence_analysis['recommendation'] = 'good_convergence'
    elif success_rate >= 0.80:
        convergence_analysis['recommendation'] = 'acceptable_convergence'
    else:
        convergence_analysis['recommendation'] = 'poor_convergence_investigate'
    
    print(f"🔍 Monte Carlo Convergence Analysis:")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Recommendation: {convergence_analysis['recommendation']}")
    
    return convergence_analysis


def create_robustness_distribution_plots(mc_summary: Dict[str, Dict[str, Dict[str, Any]]],
                                       method_name: str = 'clinical',
                                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive plots showing MC robustness distributions per group.
    
    Args:
        mc_summary: Summary from summarize_monte_carlo_per_group
        method_name: BMI grouping method to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure with robustness plots
    """
    if method_name not in mc_summary:
        available_methods = list(mc_summary.keys())
        if available_methods:
            method_name = available_methods[0]
        else:
            raise ValueError("No MC summary data available")
    
    method_data = mc_summary[method_name]
    groups = list(method_data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Monte Carlo Robustness Analysis - {method_name.title()} BMI Grouping', fontsize=16)
    
    # Color scheme for groups
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
    
    # Plot 1: Distribution histograms for 90% confidence
    ax1 = axes[0, 0]
    for i, group_name in enumerate(groups):
        tau_data = method_data[group_name].get('tau_0.90', {})
        if 'raw_weeks' in tau_data and tau_data['raw_weeks']:
            ax1.hist(tau_data['raw_weeks'], bins=20, alpha=0.6, 
                    label=f'Group {group_name}', color=colors[i])
    
    ax1.set_xlabel('Optimal Week (90% Confidence)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('MC Distribution - 90% Confidence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution histograms for 95% confidence
    ax2 = axes[0, 1]
    for i, group_name in enumerate(groups):
        tau_data = method_data[group_name].get('tau_0.95', {})
        if 'raw_weeks' in tau_data and tau_data['raw_weeks']:
            ax2.hist(tau_data['raw_weeks'], bins=20, alpha=0.6,
                    label=f'Group {group_name}', color=colors[i])
    
    ax2.set_xlabel('Optimal Week (95% Confidence)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('MC Distribution - 95% Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence interval widths
    ax3 = axes[0, 2]
    
    tau_levels = ['tau_0.90', 'tau_0.95']
    ci_widths = {tau: [] for tau in tau_levels}
    group_labels = []
    
    for group_name in groups:
        group_labels.append(f'Group {group_name}')
        for tau in tau_levels:
            tau_data = method_data[group_name].get(tau, {})
            ci_width = tau_data.get('ci_width', 0)
            ci_widths[tau].append(ci_width)
    
    x = np.arange(len(groups))
    width = 0.35
    
    ax3.bar(x - width/2, ci_widths['tau_0.90'], width, label='90% Confidence', alpha=0.8)
    ax3.bar(x + width/2, ci_widths['tau_0.95'], width, label='95% Confidence', alpha=0.8)
    
    ax3.set_xlabel('BMI Group')
    ax3.set_ylabel('95% CI Width (weeks)')
    ax3.set_title('Monte Carlo Uncertainty')
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Robustness labels summary
    ax4 = axes[1, 0]
    
    robustness_counts = {'high': 0, 'medium': 0, 'low': 0, 'unstable': 0, 'insufficient_data': 0}
    
    for group_data in method_data.values():
        for tau_data in group_data.values():
            label = tau_data.get('robustness_label', 'insufficient_data')
            robustness_counts[label] += 1
    
    # Filter out zero counts
    filtered_counts = {k: v for k, v in robustness_counts.items() if v > 0}
    
    if filtered_counts:
        ax4.pie(filtered_counts.values(), labels=filtered_counts.keys(), 
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Robustness Classification')
    
    # Plot 5: Coefficient of variation
    ax5 = axes[1, 1]
    
    cv_90 = []
    cv_95 = []
    
    for group_name in groups:
        for tau, cv_list in [('tau_0.90', cv_90), ('tau_0.95', cv_95)]:
            tau_data = method_data[group_name].get(tau, {})
            if 'mean' in tau_data and 'std' in tau_data and tau_data['mean'] > 0:
                cv = (tau_data['std'] / tau_data['mean']) * 100
                cv_list.append(cv)
            else:
                cv_list.append(0)
    
    x = np.arange(len(groups))
    ax5.bar(x - width/2, cv_90, width, label='90% Confidence', alpha=0.8)
    ax5.bar(x + width/2, cv_95, width, label='95% Confidence', alpha=0.8)
    
    ax5.set_xlabel('BMI Group')
    ax5.set_ylabel('Coefficient of Variation (%)')
    ax5.set_title('MC Variability (CV)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(group_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Mean vs CI Width scatter
    ax6 = axes[1, 2]
    
    for tau, marker in [('tau_0.90', 'o'), ('tau_0.95', 's')]:
        means = []
        ci_widths_tau = []
        
        for group_name in groups:
            tau_data = method_data[group_name].get(tau, {})
            if 'mean' in tau_data and 'ci_width' in tau_data:
                means.append(tau_data['mean'])
                ci_widths_tau.append(tau_data['ci_width'])
        
        if means and ci_widths_tau:
            ax6.scatter(means, ci_widths_tau, marker=marker, alpha=0.7, s=80,
                       label=tau.replace('tau_', '').replace('0.', '') + '% Confidence')
    
    ax6.set_xlabel('Mean Optimal Week')
    ax6.set_ylabel('95% CI Width')
    ax6.set_title('Precision vs Location')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def export_monte_carlo_results(mc_results: Dict[str, Any],
                             mc_summary: Dict[str, Dict[str, Dict[str, Any]]],
                             output_path: Path) -> Dict[str, Path]:
    """
    Export Monte Carlo results to CSV files.
    
    Args:
        mc_results: Raw MC results
        mc_summary: Summarized results
        output_path: Output directory path
        
    Returns:
        Dictionary of exported file paths
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    # Export detailed summary
    summary_rows = []
    for method_name, method_data in mc_summary.items():
        for group_name, group_data in method_data.items():
            for tau_key, stats in group_data.items():
                summary_rows.append({
                    'BMI_Grouping_Method': method_name,
                    'BMI_Group': group_name,
                    'Confidence_Level': tau_key.replace('tau_', ''),
                    'N_Simulations': stats.get('n_simulations', 0),
                    'Mean_Optimal_Week': stats.get('mean', np.nan),
                    'Std_Optimal_Week': stats.get('std', np.nan),
                    'Median_Optimal_Week': stats.get('median', np.nan),
                    'CI_2.5_Percentile': stats.get('ci_2.5', np.nan),
                    'CI_97.5_Percentile': stats.get('ci_97.5', np.nan),
                    'CI_Width': stats.get('ci_width', np.nan),
                    'Robustness_Label': stats.get('robustness_label', 'unknown')
                })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_file = output_path / 'prob3_monte_carlo_robustness.csv'
        df_summary.to_csv(summary_file, index=False)
        exported_files['summary'] = summary_file
        print(f"✅ Exported MC summary: {summary_file}")
    
    # Export simulation metadata
    if mc_results['model_metadata']:
        df_metadata = pd.DataFrame(mc_results['model_metadata'])
        metadata_file = output_path / 'prob3_monte_carlo_metadata.csv'
        df_metadata.to_csv(metadata_file, index=False)
        exported_files['metadata'] = metadata_file
        print(f"✅ Exported MC metadata: {metadata_file}")
    
    return exported_files
