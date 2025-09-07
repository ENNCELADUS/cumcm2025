"""
Data preprocessing transformers to prevent data leakage.
These transformers implement proper fit/transform pattern for Problem 4.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import spearmanr


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Winsorizer that fits quantiles on training data and applies to new data."""
    
    def __init__(self, columns: List[str], quantiles: Tuple[float, float] = (0.01, 0.99)):
        self.columns = columns
        self.quantiles = quantiles
        self.bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit winsorization bounds on training data."""
        for col in self.columns:
            if col in X.columns:
                lower_bound = X[col].quantile(self.quantiles[0])
                upper_bound = X[col].quantile(self.quantiles[1])
                self.bounds_[col] = (lower_bound, upper_bound)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization bounds."""
        X_transformed = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_transformed.columns:
                X_transformed[col] = np.clip(X_transformed[col], lower, upper)
        return X_transformed


class StandardizerTransformer(BaseEstimator, TransformerMixin):
    """Standardizer that fits mean/std on training data and applies to new data."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.stats_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit standardization parameters on training data."""
        for col in self.columns:
            if col in X.columns and X[col].notna().sum() > 0:
                mean_val = X[col].mean()
                std_val = X[col].std()
                if std_val > 0:
                    self.stats_[col] = {'mean': mean_val, 'std': std_val}
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization."""
        X_transformed = X.copy()
        for col, stats in self.stats_.items():
            if col in X_transformed.columns:
                col_std = f"{col}_std"
                X_transformed[col_std] = (X_transformed[col] - stats['mean']) / stats['std']
        return X_transformed


class CollinearityRemover(BaseEstimator, TransformerMixin):
    """Remove collinear features based on training data correlations."""
    
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.features_to_remove_ = []
        self.correlation_info_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Identify collinear features on training data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr(method='spearman').abs()
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((corr_matrix > self.correlation_threshold) & upper_triangle)
        
        features_to_remove = []
        correlation_info = {}
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            corr_val = corr_matrix.iloc[i, j]
            
            # Prioritize keeping certain features
            if self._should_keep_col1_over_col2(col1, col2):
                remove_col = col2
                keep_col = col1
            else:
                remove_col = col1
                keep_col = col2
                
            if remove_col not in features_to_remove:
                features_to_remove.append(remove_col)
                correlation_info[remove_col] = {
                    'correlated_with': keep_col,
                    'correlation': corr_val,
                    'reason': f'High correlation (r={corr_val:.3f}) with {keep_col}'
                }
        
        self.features_to_remove_ = features_to_remove
        self.correlation_info_ = correlation_info
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove collinear features."""
        cols_to_keep = [col for col in X.columns if col not in self.features_to_remove_]
        return X[cols_to_keep].copy()
    
    def _should_keep_col1_over_col2(self, col1: str, col2: str) -> bool:
        """Priority rules for which feature to keep when collinear."""
        # Keep BMI over height/weight
        if col1 == 'bmi' and col2 in ['height', 'weight']:
            return True
        if col2 == 'bmi' and col1 in ['height', 'weight']:
            return False
            
        # Keep map_ratio over uniq_rate  
        if col1 == 'map_ratio' and col2 == 'uniq_rate':
            return True
        if col2 == 'map_ratio' and col1 == 'uniq_rate':
            return False
            
        # Keep gc13 over gc18/gc21 (main GC chromosome)
        if col1 == 'gc13' and col2 in ['gc18', 'gc21']:
            return True
        if col2 == 'gc13' and col1 in ['gc18', 'gc21']:
            return False
            
        # Default: keep first column (arbitrary but consistent)
        return True


def create_qc_filter_thresholds(df: pd.DataFrame) -> Dict[str, Any]:
    """Create QC filter thresholds based on training data only."""
    thresholds = {}
    
    # GC outlier threshold (40-60%)
    thresholds['gc_normal_range'] = (0.40, 0.60)
    
    # Bottom percentile removal for reads
    thresholds['reads_bottom_percentile'] = 0.01
    if 'reads' in df.columns:
        thresholds['reads_min_threshold'] = df['reads'].quantile(0.01)
    
    return thresholds


def apply_qc_filters_with_thresholds(df: pd.DataFrame, thresholds: Dict[str, Any], 
                                   verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply QC filters using pre-fitted thresholds."""
    df_result = df.copy()
    qc_stats = {
        'initial_samples': len(df),
        'steps': []
    }
    
    # Mark GC outliers (but don't remove)
    gc_min, gc_max = thresholds['gc_normal_range']
    df_result['gc_outlier'] = (df_result['gc_global'] < gc_min) | (df_result['gc_global'] > gc_max)
    gc_outliers = df_result['gc_outlier'].sum()
    
    qc_stats['steps'].append({
        'step': 'gc_outlier_marking',
        'removed': 0,
        'remaining': len(df_result),
        'gc_outliers_marked': gc_outliers
    })
    
    # Remove bottom percentile of reads
    if 'reads_min_threshold' in thresholds and 'reads' in df_result.columns:
        reads_threshold = thresholds['reads_min_threshold']
        before_reads_filter = len(df_result)
        df_result = df_result[df_result['reads'] >= reads_threshold]
        removed_low_reads = before_reads_filter - len(df_result)
        
        qc_stats['steps'].append({
            'step': 'low_reads_removal',
            'removed': removed_low_reads,
            'remaining': len(df_result),
            'threshold': reads_threshold
        })
    
    qc_stats['final_samples'] = len(df_result)
    qc_stats['total_removed'] = qc_stats['initial_samples'] - qc_stats['final_samples']
    qc_stats['retention_rate'] = qc_stats['final_samples'] / qc_stats['initial_samples']
    
    if verbose:
        print(f"🔍 Applying QC filters with training-fitted thresholds:")
        print(f"   📊 GC outliers marked: {gc_outliers} ({gc_outliers/len(df)*100:.1f}%)")
        print(f"   🗑️  Low reads removed: {qc_stats['total_removed']} samples")
        print(f"   ✅ Final samples: {qc_stats['final_samples']} (retention: {qc_stats['retention_rate']*100:.1f}%)")
    
    return df_result, qc_stats
