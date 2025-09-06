"""
Problem 3: Multi-Covariate AFT Extension
Extended AFT analysis with multiple covariates, collinearity control, and enhanced group-wise reporting.
"""

from .data_preprocessing import *
from .survival_analysis import *  
from .bmi_grouping import *
from .monte_carlo import *
from .validation import *

__all__ = [
    # Extended data preprocessing
    'comprehensive_data_preprocessing',
    'apply_extended_qc_filters',
    'parse_gestational_weeks',
    'remove_outliers_iqr',
    'calculate_vif',
    'standardize_covariates_extended',
    'assess_multicollinearity',
    'comprehensive_vif_assessment',
    'construct_intervals_extended',
    'prepare_extended_feature_matrix',
    'validate_feature_matrix_completeness',
    'handle_missing_covariates',
    'create_spline_basis',
    
    # Extended survival analysis
    'ExtendedAFTAnalyzer',
    'fit_aft_model_extended',
    'comprehensive_aft_model_fitting',
    'assess_nonlinearity',
    'compare_covariate_specifications',
    'compute_time_ratios',
    'validate_aft_assumptions_extended',
    'assess_model_fit_extended',
    
    # Enhanced BMI grouping with contrasts
    'create_enhanced_bmi_groups',
    'compute_group_survival_extended',
    'calculate_group_optimal_weeks',
    'compute_group_contrasts',
    'create_group_contrast_table',
    'assess_clinical_significance',
    'perform_enhanced_group_analysis',
    'create_group_survival_plots',
    
    # 300-run Monte Carlo robustness
    'run_enhanced_monte_carlo',
    'summarize_monte_carlo_per_group',
    'assess_robustness',
    'create_robustness_distribution_plots',
    'analyze_monte_carlo_convergence',
    'export_monte_carlo_results',
    
    # Comprehensive validation
    'perform_covariate_sensitivity_analysis',
    'validate_baseline_distribution_choice',
    'create_final_policy_table_extended',
    'perform_patient_level_cv',
    'generate_clinical_interpretation',
    'export_comprehensive_results',
]
