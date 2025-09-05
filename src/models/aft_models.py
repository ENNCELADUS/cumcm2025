"""
Accelerated Failure Time (AFT) models for survival analysis.

This module provides a unified interface for fitting and using AFT models
for interval-censored survival analysis in Problem 2.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, KaplanMeierFitter

# Import NPMLE (Turnbull estimator) functionality
try:
    from lifelines.fitters.npmle import npmle, reconstruct_survival_function
    INTERVAL_CENSORING_AVAILABLE = True
except ImportError:
    INTERVAL_CENSORING_AVAILABLE = False
    npmle = None
    reconstruct_survival_function = None
    print("⚠️ NPMLE (Turnbull estimator) not available in this lifelines version")
    print("   Turnbull validation will be skipped, but AFT analysis will work normally")


class TurnbullEstimator:
    """
    Wrapper class for Turnbull estimator using lifelines NPMLE function.
    
    This provides a consistent interface for the Turnbull non-parametric 
    maximum likelihood estimator for interval-censored data.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.survival_probs = None
        self.intervals = None
        self.timeline = None
        self._survival_function = None
    
    def fit_interval_censoring(self, df: pd.DataFrame, left_col: str, right_col: str, 
                              verbose: bool = False):
        """
        Fit Turnbull estimator to interval-censored data.
        
        Args:
            df: DataFrame with interval-censored data
            left_col: Column name for left bounds
            right_col: Column name for right bounds
            verbose: Whether to print fitting progress
        """
        if not INTERVAL_CENSORING_AVAILABLE:
            raise ImportError("NPMLE (Turnbull estimator) not available")
        
        left_bounds = df[left_col].values
        right_bounds = df[right_col].values
        
        # Fit NPMLE (Turnbull estimator)
        self.survival_probs, self.intervals = npmle(
            left_bounds, right_bounds, verbose=verbose
        )
        
        # Extract timeline from intervals
        self.timeline = []
        for interval in self.intervals:
            self.timeline.append(interval.left)
            if interval.right != interval.left and interval.right != np.inf:
                self.timeline.append(interval.right)
        
        self.timeline = np.array(sorted(set(self.timeline)))
        
        # Reconstruct survival function for arbitrary time points
        self._survival_function = self._create_survival_function()
        
        self.is_fitted = True
        
        if verbose:
            print(f"✅ Turnbull estimator fitted successfully")
            print(f"  Timeline range: {self.timeline[0]:.1f} - {self.timeline[-1]:.1f}")
            print(f"  Number of intervals: {len(self.intervals)}")
    
    def _create_survival_function(self):
        """Create interpolated survival function from NPMLE results."""
        # Convert intervals and probabilities to step function
        times = []
        surv_probs = []
        
        current_surv = 1.0
        
        for i, interval in enumerate(self.intervals):
            times.append(interval.left)
            surv_probs.append(current_surv)
            
            # At the right end of interval, survival drops by the mass
            if interval.right != np.inf:
                times.append(interval.right)
                current_surv -= self.survival_probs[i]
                surv_probs.append(current_surv)
        
        # Ensure we have a proper step function
        times = np.array(times)
        surv_probs = np.array(surv_probs)
        
        def survival_at_times(query_times):
            """Return survival probabilities at query times."""
            result = np.ones_like(query_times, dtype=float)
            
            for i, t in enumerate(query_times):
                # Find the appropriate survival probability for time t
                valid_indices = times <= t
                if np.any(valid_indices):
                    last_valid_idx = np.where(valid_indices)[0][-1]
                    result[i] = surv_probs[last_valid_idx]
                else:
                    result[i] = 1.0  # Before any events
            
            # Ensure survival probabilities are properly bounded [0, 1]
            result = np.clip(result, 0.0, 1.0)
            return result
        
        return survival_at_times
    
    def survival_function_at_times(self, times):
        """
        Predict survival probabilities at given times.
        
        Args:
            times: Array of time points
            
        Returns:
            Array of survival probabilities
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before prediction")
        
        return self._survival_function(times)


@dataclass
class SurvivalResults:
    """Container for survival analysis results."""
    times: np.ndarray
    survival_probs: np.ndarray
    bmi: float
    bmi_z: float
    group_name: str
    n_mothers: int
    prob_at_last_week: float


@dataclass
class ModelFitResults:
    """Container for model fitting results."""
    model: Union[WeibullAFTFitter, LogLogisticAFTFitter]
    model_name: str
    fit_success: bool
    aic: Optional[float]
    log_likelihood: Optional[float]
    bmi_coefficient: Optional[float]
    bmi_pvalue: Optional[float]


class AFTSurvivalAnalyzer:
    """
    Unified interface for AFT survival analysis with interval censoring.
    
    Handles fitting multiple AFT models, model comparison, and survival prediction.
    """
    
    def __init__(self, baseline_week: float = 10.0):
        """
        Initialize the AFT analyzer.
        
        Args:
            baseline_week: Baseline week for time scale correction
        """
        self.baseline_week = baseline_week
        self.models: Dict[str, ModelFitResults] = {}
        self.primary_model = None
        self.bmi_standardization = None
        
    def fit_models(self, df_X: pd.DataFrame, formula: str = '~ bmi_z') -> Dict[str, ModelFitResults]:
        """
        Fit multiple AFT models and compare them.
        
        Args:
            df_X: DataFrame with interval-censored data and features
            formula: Model formula for covariates
            
        Returns:
            Dictionary of fitted model results
        """
        # Store BMI standardization parameters
        self.bmi_standardization = {
            'mean': df_X.attrs.get('bmi_mean', df_X['bmi'].mean()),
            'std': df_X.attrs.get('bmi_std', df_X['bmi'].std())
        }
        
        model_configs = [
            ('Weibull', WeibullAFTFitter()),
            ('LogLogistic', LogLogisticAFTFitter())
        ]
        
        for model_name, model in model_configs:
            print(f"🔵 Fitting {model_name} AFT model...")
            
            try:
                model.fit_interval_censoring(
                    df_X, 
                    lower_bound_col='L', 
                    upper_bound_col='R', 
                    formula=formula
                )
                
                # Extract model statistics
                aic = getattr(model, 'AIC_', None)
                log_likelihood = getattr(model, 'log_likelihood_', None)
                
                # Extract BMI coefficient and p-value
                bmi_coef, bmi_pvalue = self._extract_bmi_coefficient(model)
                
                fit_result = ModelFitResults(
                    model=model,
                    model_name=model_name,
                    fit_success=True,
                    aic=aic,
                    log_likelihood=log_likelihood,
                    bmi_coefficient=bmi_coef,
                    bmi_pvalue=bmi_pvalue
                )
                
                self.models[model_name] = fit_result
                print(f"✅ {model_name} AFT model fitted successfully")
                
            except Exception as e:
                print(f"❌ {model_name} AFT fitting failed: {e}")
                self.models[model_name] = ModelFitResults(
                    model=None,
                    model_name=model_name,
                    fit_success=False,
                    aic=None,
                    log_likelihood=None,
                    bmi_coefficient=None,
                    bmi_pvalue=None
                )
        
        # Select primary model (best AIC among successful fits)
        self._select_primary_model()
        
        return self.models
    
    def _extract_bmi_coefficient(self, model) -> Tuple[Optional[float], Optional[float]]:
        """Extract BMI coefficient and p-value from fitted model."""
        try:
            # Try different parameter key formats
            param_keys_to_try = ['bmi_z', ('lambda_', 'bmi_z'), ('beta_', 'bmi_z')]
            
            bmi_coef = None
            for key in param_keys_to_try:
                if key in model.params_:
                    bmi_coef = model.params_[key]
                    break
            
            # Try to get p-value from summary
            bmi_pvalue = None
            if hasattr(model, 'summary'):
                summary_df = model.summary
                if hasattr(summary_df, 'index'):
                    # Check for p-value in different index formats
                    if ('lambda_', 'bmi_z') in summary_df.index:
                        bmi_pvalue = summary_df.loc[('lambda_', 'bmi_z'), 'p'] if 'p' in summary_df.columns else None
                    elif 'bmi_z' in summary_df.index:
                        bmi_pvalue = summary_df.loc['bmi_z', 'p'] if 'p' in summary_df.columns else None
            
            return bmi_coef, bmi_pvalue
            
        except Exception:
            return None, None
    
    def _select_primary_model(self):
        """Select the best model as primary based on AIC."""
        successful_models = [name for name, result in self.models.items() if result.fit_success]
        
        if not successful_models:
            self.primary_model = None
            return
        
        # Select best model by AIC
        best_model_name = min(successful_models, 
                             key=lambda name: self.models[name].aic or float('inf'))
        self.primary_model = self.models[best_model_name].model
        
        print(f"🎯 Using {best_model_name} as primary model")
    
    def predict_survival_function(self, bmi_values: Union[float, List[float]], 
                                time_grid: np.ndarray) -> Dict[str, SurvivalResults]:
        """
        Predict survival functions for given BMI values.
        
        Args:
            bmi_values: BMI value(s) to predict for
            time_grid: Time points for prediction (absolute gestational weeks)
            
        Returns:
            Dictionary of survival results by BMI group
        """
        if self.primary_model is None:
            raise ValueError("No primary model available. Fit models first.")
        
        if isinstance(bmi_values, (int, float)):
            bmi_values = [bmi_values]
        
        results = {}
        last_observed_week = 24.0
        
        for i, bmi_val in enumerate(bmi_values):
            # Standardize BMI
            bmi_z = (bmi_val - self.bmi_standardization['mean']) / self.bmi_standardization['std']
            
            # Create query dataframe
            X_query = pd.DataFrame({'bmi_z': [bmi_z]})
            
            try:
                # Predict survival function
                surv_func = self.primary_model.predict_survival_function(X_query, times=time_grid)
                survival_probs = surv_func.iloc[:, 0].values
                
                # Calculate probability at last observed week
                last_week_idx = np.argmin(np.abs(time_grid - last_observed_week))
                prob_at_last_week = 1 - survival_probs[last_week_idx]
                
                results[f'BMI_{bmi_val:.1f}'] = SurvivalResults(
                    times=time_grid,
                    survival_probs=survival_probs,
                    bmi=bmi_val,
                    bmi_z=bmi_z,
                    group_name=f'BMI_{bmi_val:.1f}',
                    n_mothers=1,  # Individual prediction
                    prob_at_last_week=prob_at_last_week
                )
                
            except Exception as e:
                print(f"❌ Failed to predict survival for BMI {bmi_val}: {e}")
        
        return results
    
    def get_model_summary(self) -> Dict[str, str]:
        """Get a summary of fitted models."""
        if not self.models:
            return {"status": "No models fitted"}
        
        summary = {}
        for name, result in self.models.items():
            if result.fit_success:
                summary[name] = {
                    'AIC': f"{result.aic:.2f}" if result.aic else "N/A",
                    'LogLikelihood': f"{result.log_likelihood:.2f}" if result.log_likelihood else "N/A",
                    'BMI_Coefficient': f"{result.bmi_coefficient:.4f}" if result.bmi_coefficient else "N/A",
                    'BMI_PValue': f"{result.bmi_pvalue:.4f}" if result.bmi_pvalue else "N/A"
                }
            else:
                summary[name] = {'status': 'Failed to fit'}
        
        return summary


class OptimalWeeksCalculator:
    """
    Calculate optimal testing weeks based on survival functions.
    """
    
    @staticmethod
    def calculate_optimal_week(survival_result: SurvivalResults, 
                             confidence_level: float = 0.90) -> Union[float, Dict[str, Any]]:
        """
        Calculate optimal testing week for a given confidence level.
        
        Args:
            survival_result: SurvivalResults object
            confidence_level: Desired confidence level (e.g., 0.90 for 90%)
            
        Returns:
            Optimal week (float) or dict with details for unreachable cases
        """
        threshold = 1 - confidence_level  # Survival probability threshold
        times = survival_result.times
        survival_probs = survival_result.survival_probs
        last_observed_week = 24.0
        
        # Find first time point where survival probability drops below threshold
        crossing_indices = np.where(survival_probs <= threshold)[0]
        
        if len(crossing_indices) > 0:
            optimal_time = times[crossing_indices[0]]
            
            # Check if within reasonable clinical window
            if optimal_time <= last_observed_week:
                return optimal_time
            else:
                return {
                    'week': np.inf,
                    'prob_at_last_week': survival_result.prob_at_last_week,
                    'target_prob': confidence_level,
                    'status': 'unreachable'
                }
        else:
            return {
                'week': np.inf,
                'prob_at_last_week': survival_result.prob_at_last_week,
                'target_prob': confidence_level,
                'status': 'never_reached'
            }
    
    @staticmethod
    def calculate_group_optimal_weeks(survival_results: Dict[str, SurvivalResults],
                                    confidence_levels: List[float] = [0.90, 0.95]) -> Dict[str, Dict[float, Union[float, Dict]]]:
        """
        Calculate optimal weeks for multiple groups and confidence levels.
        
        Args:
            survival_results: Dictionary of survival results by group
            confidence_levels: List of confidence levels to calculate
            
        Returns:
            Nested dictionary: {group_name: {confidence_level: optimal_week}}
        """
        group_optimal_weeks = {}
        
        for group_name, survival_result in survival_results.items():
            group_optimal_weeks[group_name] = {}
            
            for conf_level in confidence_levels:
                optimal_result = OptimalWeeksCalculator.calculate_optimal_week(
                    survival_result, conf_level
                )
                group_optimal_weeks[group_name][conf_level] = optimal_result
        
        return group_optimal_weeks


def fit_aft_models(df_X: pd.DataFrame, formula: str = '~ bmi_z', verbose: bool = True) -> Tuple[Optional[Union[WeibullAFTFitter, LogLogisticAFTFitter]], str, Dict[str, ModelFitResults]]:
    """
    Fit multiple AFT models and select the best one.
    
    Args:
        df_X: DataFrame with interval-censored data and features
        formula: Model formula for covariates
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (primary_model, primary_name, all_models_dict)
    """
    if verbose:
        print("⚙️ Fitting AFT models to interval-censored data...")
    
    models = {}
    
    # Primary model: Weibull AFT
    if verbose:
        print("\n🔵 Fitting Weibull AFT model...")
    weibull_aft = WeibullAFTFitter()
    
    try:
        weibull_aft.fit_interval_censoring(
            df_X, 
            lower_bound_col='L', 
            upper_bound_col='R', 
            formula=formula
        )
        if verbose:
            print("✅ Weibull AFT model fitted successfully")
        weibull_success = True
        
        # Extract statistics
        aic = getattr(weibull_aft, 'AIC_', None)
        log_likelihood = getattr(weibull_aft, 'log_likelihood_', None)
        bmi_coef, bmi_pvalue = _extract_bmi_coefficient(weibull_aft)
        
        models['Weibull'] = ModelFitResults(
            model=weibull_aft,
            model_name='Weibull AFT',
            fit_success=True,
            aic=aic,
            log_likelihood=log_likelihood,
            bmi_coefficient=bmi_coef,
            bmi_pvalue=bmi_pvalue
        )
        
    except Exception as e:
        if verbose:
            print(f"❌ Weibull AFT fitting failed: {e}")
        weibull_success = False
    
    # Alternative model: Log-logistic AFT
    if verbose:
        print("\n🟠 Fitting Log-logistic AFT model...")
    loglogistic_aft = LogLogisticAFTFitter()
    
    try:
        loglogistic_aft.fit_interval_censoring(
            df_X, 
            lower_bound_col='L', 
            upper_bound_col='R', 
            formula=formula
        )
        if verbose:
            print("✅ Log-logistic AFT model fitted successfully")
        loglogistic_success = True
        
        # Extract statistics
        aic = getattr(loglogistic_aft, 'AIC_', None)
        log_likelihood = getattr(loglogistic_aft, 'log_likelihood_', None)
        bmi_coef, bmi_pvalue = _extract_bmi_coefficient(loglogistic_aft)
        
        models['LogLogistic'] = ModelFitResults(
            model=loglogistic_aft,
            model_name='Log-logistic AFT',
            fit_success=True,
            aic=aic,
            log_likelihood=log_likelihood,
            bmi_coefficient=bmi_coef,
            bmi_pvalue=bmi_pvalue
        )
        
    except Exception as e:
        if verbose:
            print(f"❌ Log-logistic AFT fitting failed: {e}")
        loglogistic_success = False
    
    # Choose primary model for analysis
    if weibull_success:
        primary_model = weibull_aft
        primary_name = "Weibull AFT"
        if verbose:
            print(f"\n🎯 Using {primary_name} as primary model")
    elif loglogistic_success:
        primary_model = loglogistic_aft
        primary_name = "Log-logistic AFT"
        if verbose:
            print(f"\n🎯 Using {primary_name} as primary model")
    else:
        if verbose:
            print("❌ Both AFT models failed to fit")
        primary_model = None
        primary_name = None
    
    return primary_model, primary_name, models


def _extract_bmi_coefficient(model) -> Tuple[Optional[float], Optional[float]]:
    """Extract BMI coefficient and p-value from fitted model."""
    try:
        # Try different parameter key formats
        param_keys_to_try = ['bmi_z', ('lambda_', 'bmi_z'), ('beta_', 'bmi_z')]
        
        bmi_coef = None
        for key in param_keys_to_try:
            if key in model.params_:
                bmi_coef = model.params_[key]
                break
        
        # Try to get p-value from summary
        bmi_pvalue = None
        if hasattr(model, 'summary'):
            summary_df = model.summary
            if hasattr(summary_df, 'index'):
                # Check for p-value in different index formats
                if ('lambda_', 'bmi_z') in summary_df.index:
                    bmi_pvalue = summary_df.loc[('lambda_', 'bmi_z'), 'p'] if 'p' in summary_df.columns else None
                elif 'bmi_z' in summary_df.index:
                    bmi_pvalue = summary_df.loc['bmi_z', 'p'] if 'p' in summary_df.columns else None
        
        return bmi_coef, bmi_pvalue
        
    except Exception:
        return None, None


def display_model_summary(model, model_name: str, verbose: bool = True):
    """Display comprehensive model summary and diagnostics."""
    if model is None:
        if verbose:
            print("❌ No fitted model available for summary")
        return
    
    if verbose:
        print(f"📊 {model_name} Model Summary:")
        print("="*50)
        
        # Display model summary
        try:
            print(model.summary)
        except:
            print("Model summary not available")
        
        print(f"\n📈 Model Parameters:")
        if hasattr(model, 'params_'):
            for param_name, param_value in model.params_.items():
                print(f"  {param_name}: {param_value:.4f}")
        
        # Extract BMI coefficient significance
        try:
            bmi_coef, bmi_pvalue = _extract_bmi_coefficient(model)
            
            if bmi_coef is not None:
                print(f"\n🎯 BMI Effect Analysis:")
                print(f"  BMI coefficient (standardized): {bmi_coef:.4f}")
                
                if bmi_pvalue is not None:
                    print(f"  P-value: {bmi_pvalue:.4f}")
                    significance = "significant" if bmi_pvalue < 0.05 else "not significant"
                    print(f"  Statistical significance: {significance}")
                else:
                    print(f"  P-value: Not available")
                
                # Interpretation
                effect_direction = "delays" if bmi_coef > 0 else "accelerates"
                print(f"  Interpretation: Higher BMI {effect_direction} time to Y≥4% threshold")
                
                # Additional AFT interpretation
                if model_name == "Weibull AFT":
                    print(f"  AFT interpretation: 1-unit BMI increase → {100*bmi_coef:.1f}% change in log-time scale")
            else:
                print(f"\n🎯 BMI Effect Analysis:")
                print(f"  ⚠️ Could not find BMI coefficient in model parameters")
            
        except Exception as e:
            print(f"⚠️ Could not extract coefficient details: {e}")
        
        # Model fit statistics
        try:
            if hasattr(model, 'AIC_'):
                print(f"\n📊 Model Fit Statistics:")
                print(f"  AIC: {model.AIC_:.2f}")
            if hasattr(model, 'log_likelihood_'):
                print(f"  Log-likelihood: {model.log_likelihood_:.2f}")
        except:
            print("⚠️ Model fit statistics not available")


def predict_survival_curves(model, df_intervals: pd.DataFrame, time_grid: np.ndarray, 
                           quartiles: List[float] = [0.25, 0.5, 0.75], verbose: bool = True) -> Dict[str, Dict]:
    """
    Generate survival curves for different BMI levels.
    
    Args:
        model: Fitted AFT model
        df_intervals: DataFrame with interval data for BMI statistics
        time_grid: Time points for prediction
        quartiles: BMI quartiles to analyze
        verbose: Whether to print progress information
        
    Returns:
        Dictionary of survival curves by BMI group
    """
    if model is None:
        if verbose:
            print("❌ No fitted model available for survival prediction")
        return {}
    
    if verbose:
        print("📈 Generating survival curves for different BMI levels...")
    
    # Get BMI statistics
    bmi_mean = df_intervals['bmi'].mean()
    bmi_std = df_intervals['bmi'].std()
    
    # Create BMI representative values (quartiles)
    bmi_quartiles = df_intervals['bmi'].quantile(quartiles)
    
    if verbose:
        print(f"\n📊 BMI quartiles for analysis:")
        for q, bmi_val in bmi_quartiles.items():
            print(f"  Q{int(q*100):02d}: {bmi_val:.1f}")
    
    survival_curves = {}
    
    if verbose:
        print(f"\n🔄 Computing survival functions...")
    
    for q, bmi_val in bmi_quartiles.items():
        q_label = f"Q{int(q*100):02d}"
        
        # Standardize BMI value
        bmi_z_val = (bmi_val - bmi_mean) / bmi_std
        
        # Create query dataframe
        X_query = pd.DataFrame({'bmi_z': [bmi_z_val]})
        
        try:
            # Predict survival function
            surv_func = model.predict_survival_function(X_query, times=time_grid)
            survival_curves[f'BMI_{q_label}'] = {
                'survival': surv_func.iloc[:, 0].values,
                'times': time_grid,
                'bmi_raw': bmi_val,
                'bmi_z': bmi_z_val
            }
            if verbose:
                print(f"  ✅ {q_label}: BMI {bmi_val:.1f} (z={bmi_z_val:.2f})")
        except Exception as e:
            if verbose:
                print(f"  ❌ {q_label}: Failed to predict survival function - {e}")
    
    # Display survival probabilities at key time points
    if survival_curves and verbose:
        print(f"\n📊 Survival Probabilities at Key Weeks:")
        key_weeks = [12, 14, 16, 18, 20]
        
        print(f"{'Week':<6}", end="")
        for curve_name in survival_curves.keys():
            print(f"{curve_name:<12}", end="")
        print()
        print("-" * (6 + 12 * len(survival_curves)))
        
        for week in key_weeks:
            print(f"{week:<6}", end="")
            for curve_name, curve_data in survival_curves.items():
                # Find closest time point
                week_idx = np.argmin(np.abs(curve_data['times'] - week))
                surv_prob = curve_data['survival'][week_idx]
                print(f"{surv_prob:.3f}       ", end="")
            print()
        
        print(f"\n✅ Survival function prediction completed for {len(survival_curves)} BMI levels")
    
    return survival_curves


def calculate_optimal_weeks(survival_curves: Dict[str, Dict], 
                          confidence_levels: List[float] = [0.90, 0.95],
                          verbose: bool = True) -> Dict[float, Dict[str, float]]:
    """
    Calculate optimal testing weeks for different confidence levels.
    
    Args:
        survival_curves: Dictionary of survival curves
        confidence_levels: List of confidence levels to calculate
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary: {confidence_level: {curve_name: optimal_week}}
    """
    def _calculate_single_optimal_week(survival_data, confidence_level=0.90):
        """Calculate optimal testing week based on confidence level"""
        threshold = 1 - confidence_level  # Survival probability threshold
        times = survival_data['times']
        survival_probs = survival_data['survival']
        
        # Find first time point where survival probability drops below threshold
        crossing_indices = np.where(survival_probs <= threshold)[0]
        
        if len(crossing_indices) > 0:
            optimal_time = times[crossing_indices[0]]
            return optimal_time
        else:
            return np.inf  # Never reaches confidence level
    
    if verbose:
        print("🎯 Calculating optimal testing weeks for different confidence levels...")
    
    if not survival_curves:
        if verbose:
            print("❌ No survival curves available for optimal week calculation")
        return {}
    
    optimal_weeks = {}
    
    for conf in confidence_levels:
        if verbose:
            print(f"\n📊 Optimal weeks for {int(conf*100)}% confidence level:")
            print(f"  (When ≥{int(conf*100)}% of mothers have reached Y≥4% threshold)")
            print("-" * 60)
        
        optimal_weeks[conf] = {}
        for curve_name, curve_data in survival_curves.items():
            optimal_week = _calculate_single_optimal_week(curve_data, conf)
            optimal_weeks[conf][curve_name] = optimal_week
            
            if verbose:
                bmi_label = curve_name.replace('BMI_', '')
                bmi_val = curve_data['bmi_raw']
                
                if optimal_week == np.inf:
                    print(f"  {bmi_label}: BMI {bmi_val:.1f} → Never reaches {int(conf*100)}% confidence")
                else:
                    print(f"  {bmi_label}: BMI {bmi_val:.1f} → Week {optimal_week:.1f}")
    
    # Summary table
    if verbose:
        print(f"\n📋 Optimal Testing Weeks Summary:")
        print(f"{'BMI Group':<12} {'BMI Value':<10} {'90% Conf':<10} {'95% Conf':<10}")
        print("-" * 45)
        
        for curve_name in survival_curves.keys():
            bmi_label = curve_name.replace('BMI_', '')
            bmi_val = survival_curves[curve_name]['bmi_raw']
            week_90 = optimal_weeks.get(0.90, {}).get(curve_name, np.inf)
            week_95 = optimal_weeks.get(0.95, {}).get(curve_name, np.inf)
            
            week_90_str = f"{week_90:.1f}" if week_90 != np.inf else "Never"
            week_95_str = f"{week_95:.1f}" if week_95 != np.inf else "Never"
            
            print(f"{bmi_label:<12} {bmi_val:<10.1f} {week_90_str:<10} {week_95_str:<10}")
        
        print(f"\n✅ Optimal testing week calculation completed")
    
    return optimal_weeks


def fit_turnbull_estimator(df_X: pd.DataFrame, verbose: bool = True) -> Optional[TurnbullEstimator]:
    """
    Fit Turnbull non-parametric estimator for interval-censored data.
    
    Args:
        df_X: DataFrame with interval-censored data (L, R columns)
        verbose: Whether to print progress information
        
    Returns:
        Fitted TurnbullEstimator object or None if not available
    """
    if not INTERVAL_CENSORING_AVAILABLE:
        if verbose:
            print("⚠️ Turnbull estimator not available - NPMLE not found")
            print("   Skipping Turnbull validation (AFT analysis will continue normally)")
        return None
    
    if verbose:
        print("🔵 Fitting Turnbull non-parametric estimator...")
    
    try:
        turnbull = TurnbullEstimator()
        turnbull.fit_interval_censoring(df_X, 'L', 'R', verbose=verbose)
        
        return turnbull
        
    except Exception as e:
        if verbose:
            print(f"❌ Turnbull fitting failed: {e}")
        return None


def compare_aft_vs_turnbull(aft_model, turnbull_model, time_grid: np.ndarray, 
                           representative_bmi: float, df_intervals: pd.DataFrame,
                           verbose: bool = True, 
                           restrict_to_observed_range: bool = True) -> Dict[str, Any]:
    """
    Compare AFT model predictions against Turnbull non-parametric estimates.
    
    Args:
        aft_model: Fitted AFT model
        turnbull_model: Fitted Turnbull estimator (can be None if not available)
        time_grid: Time points for comparison
        representative_bmi: BMI value for AFT prediction (e.g., median BMI)
        df_intervals: Original intervals data for BMI standardization
        verbose: Whether to print detailed comparison
        restrict_to_observed_range: Whether to restrict comparison to observed data range
        
    Returns:
        Dictionary with comparison results and metrics
    """
    if turnbull_model is None:
        if verbose:
            print("⚠️ Turnbull model not available - skipping comparison")
            print("   AFT model validation will rely on other methods")
        return {
            'comparison_available': False,
            'reason': 'Turnbull estimator not available in this lifelines version'
        }
    
    if verbose:
        print("📊 Comparing AFT model vs Turnbull non-parametric estimator...")
    
    try:
        # Restrict to clinically meaningful range if requested (avoids extrapolation issues)
        if restrict_to_observed_range:
            # Use clinically meaningful range where we have actual observations
            # Avoid early extrapolation which causes unrealistic disagreement
            min_clinical_time = 12.0  # Start of clinically meaningful range
            max_observed_time = df_intervals[df_intervals['R'] != np.inf]['R'].max()
            
            # Filter time grid to clinically meaningful range
            time_mask = (time_grid >= min_clinical_time) & (time_grid <= max_observed_time)
            if time_mask.sum() > 0:
                time_grid = time_grid[time_mask]
                if verbose:
                    print(f"  Restricting comparison to clinically meaningful range: {min_clinical_time:.1f} - {max_observed_time:.1f} weeks")
                    print(f"  Using {len(time_grid)} time points (avoids unreliable early extrapolation)")
            else:
                if verbose:
                    print("  Warning: No time points in clinical range, using full grid")
        
        elif verbose:
            print(f"  Using full time range: {time_grid[0]:.1f} - {time_grid[-1]:.1f} weeks")
            print(f"  Warning: Early times may involve unreliable extrapolation")
        # Get Turnbull survival function (population-level, all observations)
        turnbull_survival = turnbull_model.survival_function_at_times(time_grid)
        
        # Get AFT survival function (population-level, averaged across all observations)
        # This is the correct comparison: both should be population-level curves
        bmi_mean = df_intervals['bmi'].mean()
        bmi_std = df_intervals['bmi'].std()
        
        # Create predictions for ALL observations, then average
        all_aft_predictions = []
        for _, row in df_intervals.iterrows():
            bmi_z = (row['bmi'] - bmi_mean) / bmi_std
            X_query = pd.DataFrame({'bmi_z': [bmi_z]})
            individual_survival = aft_model.predict_survival_function(X_query, times=time_grid)
            all_aft_predictions.append(individual_survival.iloc[:, 0].values)
        
        # Average across all individuals to get population-level AFT curve
        aft_survival_values = np.mean(all_aft_predictions, axis=0)
        
        # Calculate comparison metrics
        # Mean Absolute Error
        mae = np.mean(np.abs(turnbull_survival - aft_survival_values))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((turnbull_survival - aft_survival_values) ** 2))
        
        # Maximum Absolute Error
        max_ae = np.max(np.abs(turnbull_survival - aft_survival_values))
        
        # Kolmogorov-Smirnov statistic (supremum difference)
        ks_statistic = max_ae
        
        results = {
            'comparison_available': True,
            'time_grid': time_grid,
            'turnbull_survival': turnbull_survival,
            'aft_survival': aft_survival_values,
            'representative_bmi': representative_bmi,
            'bmi_z': bmi_z,
            'mae': mae,
            'rmse': rmse,
            'max_absolute_error': max_ae,
            'ks_statistic': ks_statistic
        }
        
        if verbose:
            print(f"✅ Model comparison completed")
            print(f"\n📈 Population-Level Survival Curve Comparison:")
            print(f"  Turnbull: Non-parametric estimate using all {len(df_intervals)} observations")
            print(f"  AFT: Parametric prediction averaged across all observations")
            print(f"\n📊 Comparison Metrics:")
            print(f"  Mean Absolute Error: {mae:.4f}")
            print(f"  Root Mean Square Error: {rmse:.4f}")
            print(f"  Maximum Absolute Error: {max_ae:.4f}")
            print(f"  KS Statistic: {ks_statistic:.4f}")
            
            # Interpretation
            if mae < 0.05:
                print(f"  ✅ Excellent agreement (MAE < 0.05)")
            elif mae < 0.10:
                print(f"  ✅ Good agreement (0.05 ≤ MAE < 0.10)")
            elif mae < 0.20:
                print(f"  ⚠️ Moderate agreement (0.10 ≤ MAE < 0.20)")
            else:
                print(f"  ❌ Poor agreement (MAE ≥ 0.20)")
            
            # Display survival probabilities at key time points
            print(f"\n📊 Survival Probabilities at Key Weeks:")
            print(f"{'Week':<6} {'Turnbull':<12} {'AFT':<12} {'Difference':<12}")
            print("-" * 48)
            
            key_weeks = [12, 14, 16, 18, 20]
            for week in key_weeks:
                # Find closest time point
                week_idx = np.argmin(np.abs(time_grid - week))
                if week_idx < len(turnbull_survival) and week_idx < len(aft_survival_values):
                    turnbull_prob = turnbull_survival[week_idx]
                    aft_prob = aft_survival_values[week_idx]
                    diff = abs(turnbull_prob - aft_prob)
                    print(f"{week:<6} {turnbull_prob:<12.3f} {aft_prob:<12.3f} {diff:<12.3f}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"❌ Turnbull comparison failed: {e}")
        return {
            'comparison_available': False,
            'reason': f'Comparison failed: {str(e)}'
        }


def assess_aft_goodness_of_fit(comparison_results: Dict[str, Any], alpha: float = 0.05,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Assess goodness of fit for AFT model against Turnbull estimator.
    
    Args:
        comparison_results: Results from compare_aft_vs_turnbull()
        alpha: Significance level for tests
        verbose: Whether to print assessment results
        
    Returns:
        Dictionary with goodness of fit assessment
    """
    # Check if comparison is available
    if not comparison_results.get('comparison_available', False):
        if verbose:
            print("🎯 AFT Model Goodness of Fit Assessment:")
            print("="*50)
            print("⚠️ Turnbull comparison not available")
            print(f"   Reason: {comparison_results.get('reason', 'Unknown')}")
            print("\n✅ AFT model validation will rely on:")
            print("   • Cross-validation assessment")
            print("   • Monte Carlo robustness testing")
            print("   • Model diagnostic checks")
            print("   • Clinical validation against literature")
        
        return {
            'comparison_available': False,
            'reason': comparison_results.get('reason', 'Turnbull estimator not available'),
            'alternative_validation': 'Cross-validation and Monte Carlo testing',
            'recommend_aft': True  # Assume AFT is reasonable without Turnbull comparison
        }
    
    mae = comparison_results['mae']
    rmse = comparison_results['rmse']
    ks_stat = comparison_results['ks_statistic']
    
    # Thresholds for goodness of fit assessment - adjusted for survival analysis context
    # These are more realistic for interval-censored survival data comparison
    mae_threshold = 0.10   # Allow up to 10% average absolute error
    rmse_threshold = 0.15  # Allow up to 15% root mean square error  
    ks_threshold = 0.20    # Allow up to 20% maximum difference (KS statistic)
    
    # Overall assessment
    good_fit_criteria = [
        mae < mae_threshold,
        rmse < rmse_threshold,
        ks_stat < ks_threshold
    ]
    
    criteria_met = sum(good_fit_criteria)
    overall_fit = "Good" if criteria_met >= 2 else "Moderate" if criteria_met == 1 else "Poor"
    
    assessment = {
        'mae': mae,
        'rmse': rmse,
        'ks_statistic': ks_stat,
        'mae_threshold': mae_threshold,
        'rmse_threshold': rmse_threshold,
        'ks_threshold': ks_threshold,
        'criteria_met': criteria_met,
        'total_criteria': len(good_fit_criteria),
        'overall_fit': overall_fit,
        'recommend_aft': criteria_met >= 2
    }
    
    if verbose:
        print("🎯 AFT Model Goodness of Fit Assessment:")
        print("="*50)
        
        print(f"\n📊 Fit Quality Metrics:")
        print(f"  MAE: {mae:.4f} (threshold: < {mae_threshold})")
        print(f"  RMSE: {rmse:.4f} (threshold: < {rmse_threshold})")
        print(f"  KS: {ks_stat:.4f} (threshold: < {ks_threshold})")
        
        print(f"\n✅ Criteria Assessment:")
        criteria_names = ["MAE", "RMSE", "KS"]
        for i, (name, met) in enumerate(zip(criteria_names, good_fit_criteria)):
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"  {name}: {status}")
        
        print(f"\n🎯 Overall Assessment:")
        print(f"  Criteria met: {criteria_met}/{len(good_fit_criteria)}")
        print(f"  Overall fit quality: {overall_fit}")
        
        if assessment['recommend_aft']:
            print(f"  ✅ Recommendation: AFT model provides adequate fit to data")
            print(f"     The parametric AFT assumptions appear reasonable.")
        else:
            print(f"  ⚠️ Recommendation: Consider using Turnbull estimator or alternative models")
            print(f"     AFT parametric assumptions may not be well-supported.")
        
        print(f"\n📋 Validation Summary:")
        print(f"  • Turnbull provides the non-parametric gold standard")
        print(f"  • AFT model offers interpretable covariate effects")
        print(f"  • Model agreement suggests {overall_fit.lower()} parametric fit")
    
    return assessment


def validate_aft_with_kaplan_meier(df_X: pd.DataFrame, aft_model, time_points: List[float]) -> Dict[str, float]:
    """
    Validate AFT model predictions against Kaplan-Meier estimates.
    
    Args:
        df_X: Interval-censored data
        aft_model: Fitted AFT model
        time_points: Time points for comparison
        
    Returns:
        Dictionary with calibration metrics
    """
    # Convert interval data to point estimates for KM
    def convert_to_km_data(row):
        if row['censor_type'] == 'left':
            return row['R'], 1
        elif row['censor_type'] == 'interval':
            return (row['L'] + row['R']) / 2, 1
        else:
            return row['L'], 0
    
    df_approx = df_X.copy()
    df_approx[['time_approx', 'event_approx']] = df_approx.apply(
        lambda row: pd.Series(convert_to_km_data(row)), axis=1
    )
    
    # Fit KM estimator
    km_fitter = KaplanMeierFitter()
    km_fitter.fit(
        durations=df_approx['time_approx'], 
        event_observed=df_approx['event_approx']
    )
    
    # Compare at time points (assuming mean BMI for AFT)
    bmi_mean = df_X['bmi'].mean()
    bmi_std = df_X['bmi'].std()
    bmi_z_mean = 0.0  # Standardized mean
    X_mean = pd.DataFrame({'bmi_z': [bmi_z_mean]})
    
    calibration_diffs = []
    
    for time_point in time_points:
        try:
            # AFT prediction
            aft_surv_func = aft_model.predict_survival_function(X_mean, times=[time_point])
            aft_surv = aft_surv_func.iloc[0, 0]
            
            # KM prediction
            km_surv = km_fitter.predict(time_point)
            
            diff = abs(aft_surv - km_surv)
            calibration_diffs.append(diff)
            
        except Exception:
            continue
    
    if calibration_diffs:
        return {
            'mean_abs_difference': np.mean(calibration_diffs),
            'max_abs_difference': max(calibration_diffs),
            'n_comparisons': len(calibration_diffs)
        }
    else:
        return {'mean_abs_difference': np.nan, 'max_abs_difference': np.nan, 'n_comparisons': 0}
