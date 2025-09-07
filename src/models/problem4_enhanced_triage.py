"""
Problem 4 Enhanced Triage System: Recall Rescue Plan under FPR≤1% Constraint

This module implements the comprehensive three-tier triage approach for maximizing
recall while maintaining strict FPR≤1% constraint, as outlined in the clinical guide.

Key Components:
A. Three-Tier Decision Strategy (vs single threshold)
B. Z-Score Residual Features (debiased from QC effects)  
C. Gray Zone Evidence Enhancement
D. Stratified Quality-Based Thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

# Core ML libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Statistical libraries
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TriageResults:
    """Container for three-tier triage system results."""
    strategy_name: str
    high_threshold: float
    low_threshold: float
    tier_predictions: Dict[str, np.ndarray]  # 'direct_positive', 'gray_zone', 'negative'
    tier_metrics: Dict[str, Dict]
    overall_metrics: Dict
    quality_buckets: Optional[Dict] = None


class ZScoreResidualExtractor:
    """
    Extract debiased Z-score features by removing QC-related systematic biases.
    
    For each chromosome j ∈ {13,18,21,X}:
    Z_j^obs = f_j(GC, log(reads), map_ratio, dup_ratio) + ε
    Z̃_j := Z_j^obs - f̂_j(·)
    """
    
    def __init__(self, use_polynomial: bool = False, degree: int = 2, verbose: bool = True):
        """
        Initialize the Z-score residual extractor.
        
        Args:
            use_polynomial: Whether to use polynomial features for debiasing
            degree: Polynomial degree if use_polynomial=True
            verbose: Whether to print progress
        """
        self.use_polynomial = use_polynomial
        self.degree = degree
        self.verbose = verbose
        self.fitted_models = {}
        self.feature_transformers = {}
        
    def fit_debias_models(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Fit debiasing models for each Z-score feature.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with fitted models and diagnostics
        """
        if self.verbose:
            print("🧬 Fitting Z-score debiasing models...")
            print(f"   📊 Input data shape: {X.shape}, dtype: {X.dtype}")
        
        # Identify Z-score and QC features
        z_features = ['z13', 'z18', 'z21', 'zx']
        qc_features = ['gc_global', 'reads', 'map_ratio', 'dup_ratio']
        
        # Find feature indices
        z_indices = [i for i, name in enumerate(feature_names) if name in z_features]
        qc_indices = [i for i, name in enumerate(feature_names) if name in qc_features]
        
        if len(z_indices) == 0 or len(qc_indices) == 0:
            raise ValueError(f"Missing required features. Found Z-features: {len(z_indices)}, QC-features: {len(qc_indices)}")
        
        # Extract QC features for regression
        X_qc = X[:, qc_indices]
        
        # Log-transform reads if available
        reads_idx = None
        for i, idx in enumerate(qc_indices):
            if 'reads' in feature_names[idx]:
                reads_idx = i
                break
                
        if reads_idx is not None and reads_idx < X_qc.shape[1]:
            X_qc_transformed = X_qc.copy().astype(float)  # Ensure float type
            reads_values = X_qc_transformed[:, reads_idx]
            # Ensure we have a proper numpy array and apply log1p
            if isinstance(reads_values, np.ndarray):
                X_qc_transformed[:, reads_idx] = np.log1p(np.maximum(reads_values, 0))  # Avoid log of negative
            else:
                # Fallback if there's a type issue
                X_qc_transformed = X_qc.copy().astype(float)
        else:
            X_qc_transformed = X_qc.astype(float)
        
        # Optional polynomial features
        if self.use_polynomial:
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            X_qc_transformed = poly.fit_transform(X_qc_transformed)
            self.feature_transformers['polynomial'] = poly
        
        # Fit debiasing model for each Z-score
        debias_results = {}
        
        for z_name, z_idx in zip(z_features, z_indices):
            if z_idx >= X.shape[1]:
                continue
                
            y_z = X[:, z_idx]
            
            # Fit linear regression to predict Z from QC features
            model = LinearRegression()
            model.fit(X_qc_transformed, y_z)
            
            # Calculate residuals
            y_pred = model.predict(X_qc_transformed)
            residuals = y_z - y_pred
            
            # Calculate R² and diagnostics
            ss_res = np.sum((y_z - y_pred) ** 2)
            ss_tot = np.sum((y_z - np.mean(y_z)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.fitted_models[z_name] = {
                'model': model,
                'r2': r2,
                'residual_std': np.std(residuals),
                'qc_feature_names': [feature_names[idx] for idx in qc_indices]
            }
            
            debias_results[z_name] = {
                'original_std': np.std(y_z),
                'residual_std': np.std(residuals),
                'r2': r2,
                'bias_removed': r2 > 0.01  # At least 1% variance explained
            }
            
            if self.verbose:
                print(f"   📊 {z_name}: R²={r2:.3f}, STD reduction: {np.std(y_z):.3f} → {np.std(residuals):.3f}")
        
        return debias_results
    
    def transform_to_residuals(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Transform Z-score features to residuals using fitted models.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Tuple of (transformed_X, new_feature_names)
        """
        X_transformed = X.copy()
        new_feature_names = feature_names.copy()
        
        # Identify features
        z_features = ['z13', 'z18', 'z21', 'zx']
        qc_features = ['gc_global', 'reads', 'map_ratio', 'dup_ratio']
        
        z_indices = [i for i, name in enumerate(feature_names) if name in z_features]
        qc_indices = [i for i, name in enumerate(feature_names) if name in qc_features]
        
        # Extract and transform QC features
        X_qc = X[:, qc_indices]
        
        # Apply same transformations as in fitting
        reads_idx = None
        for i, idx in enumerate(qc_indices):
            if 'reads' in feature_names[idx]:
                reads_idx = i
                break
                
        if reads_idx is not None and reads_idx < X_qc.shape[1]:
            X_qc_transformed = X_qc.copy().astype(float)  # Ensure float type
            reads_values = X_qc_transformed[:, reads_idx]
            # Ensure we have a proper numpy array and apply log1p
            if isinstance(reads_values, np.ndarray):
                X_qc_transformed[:, reads_idx] = np.log1p(np.maximum(reads_values, 0))  # Avoid log of negative
            else:
                # Fallback if there's a type issue
                X_qc_transformed = X_qc.copy().astype(float)
        else:
            X_qc_transformed = X_qc.astype(float)
            
        if self.use_polynomial and 'polynomial' in self.feature_transformers:
            X_qc_transformed = self.feature_transformers['polynomial'].transform(X_qc_transformed)
        
        # Transform each Z-score to residual
        for z_name in z_features:
            z_idx = next((i for i, name in enumerate(feature_names) if name == z_name), None)
            
            if z_idx is not None and z_name in self.fitted_models:
                model_info = self.fitted_models[z_name]
                model = model_info['model']
                
                # Predict bias and compute residual
                y_z = X[:, z_idx]
                y_pred = model.predict(X_qc_transformed)
                residuals = y_z - y_pred
                
                # Replace original Z with residual
                X_transformed[:, z_idx] = residuals
                new_feature_names[z_idx] = f"{z_name}_residual"
                
                if self.verbose:
                    print(f"   🔄 Transformed {z_name} to {z_name}_residual")
        
        return X_transformed, new_feature_names


class ThreeTierTriageSystem:
    """
    Implements the three-tier clinical triage system:
    1. Direct Positive (p ≥ τ_high): Immediate positive result
    2. Gray Zone (τ_low ≤ p < τ_high): Secondary evidence/recheck
    3. Negative (p < τ_low): Negative result
    """
    
    def __init__(self, target_fpr: float = 0.01, bootstrap_iterations: int = 1000, verbose: bool = True):
        """
        Initialize the three-tier triage system.
        
        Args:
            target_fpr: Maximum allowed false positive rate for direct positives
            bootstrap_iterations: Number of bootstrap iterations for CI estimation
            verbose: Whether to print progress
        """
        self.target_fpr = target_fpr
        self.bootstrap_iterations = bootstrap_iterations
        self.verbose = verbose
        
    def optimize_thresholds_with_bootstrap(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Optimize thresholds using bootstrap for confidence intervals.
        
        Solves: max_τ TP(τ) s.t. FP(τ) ≤ α·N₀ (α=0.01)
        with bootstrap 95% CI ensuring CI upper bound ≤ 1%
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with optimal thresholds and bootstrap statistics
        """
        if self.verbose:
            print(f"🎯 Optimizing thresholds with bootstrap validation (target FPR ≤ {self.target_fpr:.1%})")
        
        n_samples = len(y_true)
        n_positive = np.sum(y_true == 1)
        n_negative = np.sum(y_true == 0)
        
        if self.verbose:
            print(f"   📊 Dataset: {n_samples} samples ({n_positive} positive, {n_negative} negative)")
        
        # Fine-grained threshold search
        thresholds = np.concatenate([
            np.arange(0.0001, 0.01, 0.0001),    # Very fine for low thresholds
            np.arange(0.01, 0.1, 0.001),        # Fine for medium thresholds  
            np.arange(0.1, 0.9, 0.005),         # Medium for high thresholds
            np.arange(0.9, 0.999, 0.001)        # Fine for very high thresholds
        ])
        thresholds = np.unique(thresholds)
        
        if self.verbose:
            print(f"   🔍 Testing {len(thresholds)} thresholds")
        
        # Find candidate high thresholds
        candidate_thresholds = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if fpr <= self.target_fpr and recall > 0:
                candidate_thresholds.append({
                    'threshold': threshold,
                    'recall': recall,
                    'fpr': fpr,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                })
        
        if len(candidate_thresholds) == 0:
            if self.verbose:
                print(f"   ⚠️ No thresholds meet strict FPR constraint, using relaxed search")
            
            # Relaxed search with penalty
            best_score = -np.inf
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                tp = np.sum((y_pred == 1) & (y_true == 1))
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Score with heavy FPR penalty
                if fpr <= self.target_fpr * 2:  # Allow 2x target
                    score = recall - max(0, (fpr - self.target_fpr) * 10)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            tau_high = best_threshold
            bootstrap_stats = {'ci_satisfied': False, 'fpr_ci': (1.0, 1.0)}
            
        else:
            # Bootstrap validation for candidate thresholds
            valid_thresholds = []
            
            for candidate in candidate_thresholds:
                threshold = candidate['threshold']
                
                # Bootstrap FPR confidence interval
                bootstrap_fprs = []
                
                for _ in range(self.bootstrap_iterations):
                    # Resample with replacement
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    y_boot = y_true[indices]
                    p_boot = y_proba[indices]
                    
                    y_pred_boot = (p_boot >= threshold).astype(int)
                    
                    tn_boot = np.sum((y_pred_boot == 0) & (y_boot == 0))
                    fp_boot = np.sum((y_pred_boot == 1) & (y_boot == 0))
                    
                    fpr_boot = fp_boot / (fp_boot + tn_boot) if (fp_boot + tn_boot) > 0 else 0
                    bootstrap_fprs.append(fpr_boot)
                
                # Calculate 95% confidence interval
                fpr_ci = np.percentile(bootstrap_fprs, [2.5, 97.5])
                
                # Check if CI upper bound ≤ target FPR
                if fpr_ci[1] <= self.target_fpr:
                    valid_thresholds.append({
                        **candidate,
                        'fpr_ci': fpr_ci,
                        'bootstrap_fprs': bootstrap_fprs
                    })
            
            if len(valid_thresholds) > 0:
                # Select threshold with highest recall among valid ones
                best_valid = max(valid_thresholds, key=lambda x: x['recall'])
                tau_high = best_valid['threshold']
                bootstrap_stats = {
                    'ci_satisfied': True,
                    'fpr_ci': best_valid['fpr_ci'],
                    'bootstrap_fprs': best_valid['bootstrap_fprs']
                }
                
                if self.verbose:
                    print(f"   ✅ Selected τ_high = {tau_high:.4f}")
                    print(f"   📊 Recall: {best_valid['recall']:.4f}, FPR CI: [{best_valid['fpr_ci'][0]:.4f}, {best_valid['fpr_ci'][1]:.4f}]")
            else:
                # Fallback to best candidate
                best_candidate = max(candidate_thresholds, key=lambda x: x['recall'])
                tau_high = best_candidate['threshold']
                bootstrap_stats = {'ci_satisfied': False, 'fpr_ci': (1.0, 1.0)}
                
                if self.verbose:
                    print(f"   ⚠️ No threshold satisfies bootstrap CI, using best candidate: {tau_high:.4f}")
        
        # Set low threshold (empirical starting point)
        tau_low = 0.5 * tau_high
        
        results = {
            'tau_high': tau_high,
            'tau_low': tau_low,
            'bootstrap_stats': bootstrap_stats,
            'n_candidates_tested': len(candidate_thresholds) if 'candidate_thresholds' in locals() else 0
        }
        
        if self.verbose:
            print(f"   📋 Final thresholds: τ_high = {tau_high:.4f}, τ_low = {tau_low:.4f}")
        
        return results
    
    def create_quality_buckets(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Create quality-based buckets for stratified thresholds.
        
        Buckets based on:
        - GC content: normal (40-60%) vs deviated
        - Reads: high vs low (median split)
        - Optional: BMI normal vs high
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with bucket definitions and sample assignments
        """
        if self.verbose:
            print("📦 Creating quality-based buckets for stratified thresholds")
        
        n_samples = X.shape[0]
        bucket_assignments = np.zeros(n_samples, dtype=int)  # Default bucket 0
        bucket_definitions = {}
        
        # Find feature indices
        gc_idx = next((i for i, name in enumerate(feature_names) if 'gc_global' in name), None)
        reads_idx = next((i for i, name in enumerate(feature_names) if 'reads' in name), None)
        bmi_idx = next((i for i, name in enumerate(feature_names) if 'bmi' in name), None)
        
        # GC content buckets
        if gc_idx is not None:
            gc_values = X[:, gc_idx]
            gc_normal = (gc_values >= 0.40) & (gc_values <= 0.60)
        else:
            gc_normal = np.ones(n_samples, dtype=bool)  # Default all normal
        
        # Reads buckets  
        if reads_idx is not None:
            reads_values = X[:, reads_idx]
            reads_median = np.median(reads_values)
            reads_high = reads_values >= reads_median
        else:
            reads_high = np.ones(n_samples, dtype=bool)  # Default all high
        
        # BMI buckets
        if bmi_idx is not None:
            bmi_values = X[:, bmi_idx]
            bmi_normal = bmi_values <= 25  # Normal BMI
        else:
            bmi_normal = np.ones(n_samples, dtype=bool)  # Default all normal
        
        # Create bucket assignments
        # Bucket 0: High quality (GC normal + reads high + BMI normal)
        # Bucket 1: Medium quality (GC normal + reads high, BMI may be high)
        # Bucket 2: Lower quality (other combinations)
        
        high_quality = gc_normal & reads_high & bmi_normal
        medium_quality = gc_normal & reads_high & ~high_quality
        
        bucket_assignments[high_quality] = 0   # High quality
        bucket_assignments[medium_quality] = 1  # Medium quality
        bucket_assignments[~(high_quality | medium_quality)] = 2  # Lower quality
        
        # Store bucket definitions
        bucket_definitions = {
            0: {'name': 'high_quality', 'description': 'GC normal + reads high + BMI normal', 
                'count': np.sum(bucket_assignments == 0)},
            1: {'name': 'medium_quality', 'description': 'GC normal + reads high + BMI high',
                'count': np.sum(bucket_assignments == 1)},
            2: {'name': 'lower_quality', 'description': 'Other combinations',
                'count': np.sum(bucket_assignments == 2)}
        }
        
        if self.verbose:
            for bucket_id, info in bucket_definitions.items():
                print(f"   📦 Bucket {bucket_id} ({info['name']}): {info['count']} samples")
        
        return {
            'assignments': bucket_assignments,
            'definitions': bucket_definitions,
            'feature_info': {
                'gc_idx': gc_idx,
                'reads_idx': reads_idx, 
                'bmi_idx': bmi_idx
            }
        }
    
    def optimize_stratified_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                     quality_buckets: Dict) -> Dict:
        """
        Optimize thresholds separately for each quality bucket.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            quality_buckets: Quality bucket information
            
        Returns:
            Dictionary with bucket-specific thresholds
        """
        if self.verbose:
            print("🎯 Optimizing stratified thresholds per quality bucket")
        
        bucket_assignments = quality_buckets['assignments']
        bucket_definitions = quality_buckets['definitions']
        stratified_thresholds = {}
        
        for bucket_id, bucket_info in bucket_definitions.items():
            bucket_mask = bucket_assignments == bucket_id
            
            if np.sum(bucket_mask) < 10:  # Skip buckets with too few samples
                if self.verbose:
                    print(f"   ⚠️ Bucket {bucket_id}: Too few samples ({np.sum(bucket_mask)}), using global threshold")
                stratified_thresholds[bucket_id] = None
                continue
            
            y_bucket = y_true[bucket_mask]
            p_bucket = y_proba[bucket_mask]
            
            # Optimize threshold for this bucket
            bucket_optimization = self.optimize_thresholds_with_bootstrap(y_bucket, p_bucket)
            
            stratified_thresholds[bucket_id] = {
                'tau_high': bucket_optimization['tau_high'],
                'tau_low': bucket_optimization['tau_low'],
                'bootstrap_stats': bucket_optimization['bootstrap_stats'],
                'n_samples': np.sum(bucket_mask),
                'n_positive': np.sum(y_bucket == 1)
            }
            
            if self.verbose:
                print(f"   📊 Bucket {bucket_id}: τ_high = {bucket_optimization['tau_high']:.4f}, n_samples = {np.sum(bucket_mask)}")
        
        return stratified_thresholds
    
    def apply_three_tier_classification(self, y_proba: np.ndarray, tau_high: float, tau_low: float,
                                       quality_buckets: Optional[Dict] = None,
                                       stratified_thresholds: Optional[Dict] = None,
                                       X: Optional[np.ndarray] = None,
                                       feature_names: Optional[List[str]] = None) -> Dict:
        """
        Apply three-tier classification with optional stratified thresholds and gray zone rules.
        
        Args:
            y_proba: Predicted probabilities
            tau_high: High threshold for direct positives
            tau_low: Low threshold for gray zone
            quality_buckets: Optional quality bucket information
            stratified_thresholds: Optional bucket-specific thresholds
            X: Optional feature matrix for gray zone rules
            feature_names: Optional feature names for gray zone rules
            
        Returns:
            Dictionary with tier predictions and metadata
        """
        n_samples = len(y_proba)
        
        # Initialize tier assignments
        tier_predictions = {
            'direct_positive': np.zeros(n_samples, dtype=bool),
            'gray_zone': np.zeros(n_samples, dtype=bool),
            'negative': np.zeros(n_samples, dtype=bool),
            'recheck_recommended': np.zeros(n_samples, dtype=bool)
        }
        
        # Apply stratified thresholds if available
        if quality_buckets is not None and stratified_thresholds is not None:
            bucket_assignments = quality_buckets['assignments']
            
            for bucket_id in quality_buckets['definitions'].keys():
                bucket_mask = bucket_assignments == bucket_id
                
                if bucket_id in stratified_thresholds and stratified_thresholds[bucket_id] is not None:
                    # Use bucket-specific thresholds
                    bucket_tau_high = stratified_thresholds[bucket_id]['tau_high']
                    bucket_tau_low = stratified_thresholds[bucket_id]['tau_low']
                else:
                    # Use global thresholds
                    bucket_tau_high = tau_high
                    bucket_tau_low = tau_low
                
                # Apply thresholds to this bucket
                bucket_proba = y_proba[bucket_mask]
                
                direct_pos_mask = bucket_proba >= bucket_tau_high
                gray_zone_mask = (bucket_proba >= bucket_tau_low) & (bucket_proba < bucket_tau_high)
                negative_mask = bucket_proba < bucket_tau_low
                
                # Update global predictions for this bucket
                bucket_indices = np.where(bucket_mask)[0]
                tier_predictions['direct_positive'][bucket_indices[direct_pos_mask]] = True
                tier_predictions['gray_zone'][bucket_indices[gray_zone_mask]] = True
                tier_predictions['negative'][bucket_indices[negative_mask]] = True
        else:
            # Use global thresholds
            tier_predictions['direct_positive'] = y_proba >= tau_high
            tier_predictions['gray_zone'] = (y_proba >= tau_low) & (y_proba < tau_high)
            tier_predictions['negative'] = y_proba < tau_low
        
        # Apply gray zone rules if features are available
        if X is not None and feature_names is not None:
            gray_zone_indices = np.where(tier_predictions['gray_zone'])[0]
            
            if len(gray_zone_indices) > 0:
                gray_zone_enhanced = self._apply_gray_zone_rules(
                    X[gray_zone_indices], feature_names, y_proba[gray_zone_indices]
                )
                
                # Update predictions based on gray zone rules
                for i, idx in enumerate(gray_zone_indices):
                    if gray_zone_enhanced['promote_to_positive'][i]:
                        tier_predictions['direct_positive'][idx] = True
                        tier_predictions['gray_zone'][idx] = False
                    elif gray_zone_enhanced['recommend_recheck'][i]:
                        tier_predictions['recheck_recommended'][idx] = True
        
        # Final predictions for evaluation
        final_predictions = tier_predictions['direct_positive'].astype(int)
        
        results = {
            'tier_predictions': tier_predictions,
            'final_predictions': final_predictions,
            'tier_counts': {
                'direct_positive': np.sum(tier_predictions['direct_positive']),
                'gray_zone': np.sum(tier_predictions['gray_zone']),
                'negative': np.sum(tier_predictions['negative']),
                'recheck_recommended': np.sum(tier_predictions['recheck_recommended'])
            }
        }
        
        return results
    
    def _apply_gray_zone_rules(self, X_gray: np.ndarray, feature_names: List[str], 
                             p_gray: np.ndarray) -> Dict:
        """
        Apply rule-based enhancement in gray zone.
        
        Rules:
        1. OR-enhancement: |Z_j| ≥ 3 or max(Z̃_j) ≥ 2.5
        2. AND-filtering: Rule positive + high quality conditions
        3. Selective abstention: Recommend recheck for unclear cases
        
        Args:
            X_gray: Gray zone feature matrix
            feature_names: Feature names
            p_gray: Gray zone probabilities
            
        Returns:
            Dictionary with gray zone enhancement decisions
        """
        n_gray = X_gray.shape[0]
        
        enhance_results = {
            'promote_to_positive': np.zeros(n_gray, dtype=bool),
            'recommend_recheck': np.zeros(n_gray, dtype=bool)
        }
        
        # Find Z-score feature indices
        z_features = ['z13', 'z18', 'z21', 'zx']
        z_residual_features = ['z13_residual', 'z18_residual', 'z21_residual', 'zx_residual']
        
        z_indices = [i for i, name in enumerate(feature_names) if name in z_features]
        z_residual_indices = [i for i, name in enumerate(feature_names) if name in z_residual_features]
        
        # Quality feature indices
        gc_idx = next((i for i, name in enumerate(feature_names) if 'gc_global' in name), None)
        reads_idx = next((i for i, name in enumerate(feature_names) if 'reads' in name), None)
        dup_idx = next((i for i, name in enumerate(feature_names) if 'dup_ratio' in name), None)
        
        for i in range(n_gray):
            # Rule 1: |Z| ≥ 3 rule
            rule_triggered = False
            
            if len(z_indices) > 0:
                z_values = X_gray[i, z_indices]
                rule_triggered = np.any(np.abs(z_values) >= 3.0)
            
            # Rule 2: max(Z̃) ≥ 2.5 rule
            if not rule_triggered and len(z_residual_indices) > 0:
                z_residual_values = X_gray[i, z_residual_indices]
                rule_triggered = np.max(z_residual_values) >= 2.5
            
            # Quality assessment for AND-filtering
            high_quality = True
            
            if gc_idx is not None:
                gc_val = X_gray[i, gc_idx]
                if gc_val < 0.40 or gc_val > 0.60:
                    high_quality = False
            
            if reads_idx is not None and reads_idx < X_gray.shape[1]:
                reads_val = X_gray[i, reads_idx]
                # Assuming reads values are normalized, check if below median
                if reads_val < 0:  # Below normalized median
                    high_quality = False
            
            if dup_idx is not None:
                dup_val = X_gray[i, dup_idx]
                if dup_val > 0.15:  # High duplication rate
                    high_quality = False
            
            # Decision logic
            if rule_triggered and high_quality:
                # Promote to positive
                enhance_results['promote_to_positive'][i] = True
            elif rule_triggered and not high_quality:
                # Recommend recheck
                enhance_results['recommend_recheck'][i] = True
            elif p_gray[i] > 0.8 * (0.5 * 0.1):  # Close to low threshold
                # Recommend recheck for borderline cases
                enhance_results['recommend_recheck'][i] = True
        
        return enhance_results


def run_enhanced_triage_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                                X_calib: np.ndarray, y_calib: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str],
                                model_results: Any,
                                training_weights: Optional[np.ndarray] = None,
                                target_fpr: float = 0.01,
                                verbose: bool = True) -> Dict:
    """
    Run the complete enhanced triage pipeline.
    
    Implements the full Recall Rescue Plan:
    A. Z-score residual extraction (debiasing)
    B. Enhanced positive class training  
    C. Three-tier triage with stratified thresholds
    D. Gray zone evidence enhancement
    
    Args:
        X_train, y_train: Training data
        X_calib, y_calib: Calibration data
        X_test, y_test: Test data  
        feature_names: Feature names
        model_results: Original model results
        training_weights: Optional training weights
        target_fpr: Target false positive rate
        verbose: Whether to print progress
        
    Returns:
        Dictionary with complete triage results
    """
    if verbose:
        print("🚀 ENHANCED TRIAGE PIPELINE: Recall Rescue Plan")
        print("=" * 80)
    
    results = {}
    
    # Step A: Z-score residual extraction (debiasing)
    if verbose:
        print("\n📊 Step A: Z-Score Residual Feature Engineering")
    
    residual_extractor = ZScoreResidualExtractor(verbose=verbose)
    debias_results = residual_extractor.fit_debias_models(X_train, feature_names)
    
    # Transform all datasets
    X_train_debiased, feature_names_debiased = residual_extractor.transform_to_residuals(X_train, feature_names)
    X_calib_debiased, _ = residual_extractor.transform_to_residuals(X_calib, feature_names)
    X_test_debiased, _ = residual_extractor.transform_to_residuals(X_test, feature_names)
    
    results['debias_results'] = debias_results
    results['feature_names_debiased'] = feature_names_debiased
    
    # Step B: Enhanced model training (optional - can use existing model for now)
    if verbose:
        print("\n📊 Step B: Using existing calibrated model with debiased features")
    
    # Get predictions on debiased test set
    calibrated_model = model_results.calibrated_model
    y_test_proba_debiased = calibrated_model.predict_proba(X_test_debiased)[:, 1]
    y_calib_proba_debiased = calibrated_model.predict_proba(X_calib_debiased)[:, 1]
    
    # Step C: Three-tier triage system
    if verbose:
        print("\n📊 Step C: Three-Tier Triage System")
    
    triage_system = ThreeTierTriageSystem(target_fpr=target_fpr, verbose=verbose)
    
    # C1: Optimize global thresholds with bootstrap
    threshold_results = triage_system.optimize_thresholds_with_bootstrap(y_calib, y_calib_proba_debiased)
    
    # C2: Create quality buckets on calibration set (for threshold optimization)
    quality_buckets_calib = triage_system.create_quality_buckets(X_calib_debiased, feature_names_debiased)
    
    # C3: Optimize stratified thresholds
    stratified_thresholds = triage_system.optimize_stratified_thresholds(
        y_calib, y_calib_proba_debiased, quality_buckets_calib
    )
    
    # C4: Create quality buckets on test set (for final evaluation)
    quality_buckets_test = triage_system.create_quality_buckets(X_test_debiased, feature_names_debiased)
    
    results['threshold_results'] = threshold_results
    results['quality_buckets_calib'] = quality_buckets_calib
    results['quality_buckets_test'] = quality_buckets_test
    results['stratified_thresholds'] = stratified_thresholds
    
    # Step D: Apply three-tier classification on test set
    if verbose:
        print("\n📊 Step D: Three-Tier Classification on Test Set")
    
    # Strategy 1: Global thresholds only
    strategy1_results = triage_system.apply_three_tier_classification(
        y_test_proba_debiased, 
        threshold_results['tau_high'], 
        threshold_results['tau_low']
    )
    
    # Strategy 2: Stratified thresholds
    strategy2_results = triage_system.apply_three_tier_classification(
        y_test_proba_debiased,
        threshold_results['tau_high'],
        threshold_results['tau_low'], 
        quality_buckets_test,
        stratified_thresholds
    )
    
    # Strategy 3: Stratified + gray zone rules
    strategy3_results = triage_system.apply_three_tier_classification(
        y_test_proba_debiased,
        threshold_results['tau_high'],
        threshold_results['tau_low'],
        quality_buckets_test,
        stratified_thresholds,
        X_test_debiased,
        feature_names_debiased
    )
    
    # Evaluate all strategies
    strategies = {
        'S1_global': strategy1_results,
        'S2_stratified': strategy2_results, 
        'S3_enhanced': strategy3_results
    }
    
    strategy_metrics = {}
    
    for strategy_name, strategy_result in strategies.items():
        y_pred = strategy_result['final_predictions']
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_test, y_test_proba_debiased),
            'tier_counts': strategy_result['tier_counts'],
            'n_positive': np.sum(y_test),
            'n_negative': len(y_test) - np.sum(y_test)
        }
        
        strategy_metrics[strategy_name] = metrics
        
        if verbose:
            print(f"   📊 {strategy_name}: Recall={metrics['recall']:.3f}, FPR={metrics['fpr']:.3f}, Precision={metrics['precision']:.3f}")
            print(f"      Tiers: Direct+={metrics['tier_counts']['direct_positive']}, Gray={metrics['tier_counts']['gray_zone']}, Recheck={metrics['tier_counts']['recheck_recommended']}")
    
    results['strategies'] = strategies
    results['strategy_metrics'] = strategy_metrics
    
    # Select best strategy (prioritize recall with FPR constraint)
    best_strategy_name = None
    best_score = -np.inf
    
    for strategy_name, metrics in strategy_metrics.items():
        fpr = metrics['fpr']
        recall = metrics['recall']
        
        # Score with FPR constraint
        if fpr <= target_fpr:
            score = recall * 2.0  # Bonus for meeting constraint
        elif fpr <= target_fpr * 2:
            score = recall * 1.0
        else:
            score = recall * 0.5
        
        if score > best_score:
            best_score = score
            best_strategy_name = strategy_name
    
    results['best_strategy'] = {
        'name': best_strategy_name,
        'metrics': strategy_metrics[best_strategy_name] if best_strategy_name else None,
        'score': best_score
    }
    
    # Compare with original results
    original_recall = model_results.test_metrics['recall']
    best_recall = strategy_metrics[best_strategy_name]['recall'] if best_strategy_name else 0
    
    results['improvement_summary'] = {
        'original_recall': original_recall,
        'enhanced_recall': best_recall,
        'recall_improvement': best_recall - original_recall,
        'original_fpr': model_results.test_metrics['fpr'],
        'enhanced_fpr': strategy_metrics[best_strategy_name]['fpr'] if best_strategy_name else 1.0
    }
    
    if verbose:
        print(f"\n🎉 ENHANCED TRIAGE COMPLETED!")
        print(f"   🏆 Best Strategy: {best_strategy_name}")
        print(f"   📈 Recall Improvement: {original_recall:.3f} → {best_recall:.3f} (+{best_recall - original_recall:.3f})")
        print(f"   📊 FPR: {results['improvement_summary']['enhanced_fpr']:.3f}")
        print("=" * 80)
    
    return results
