#!/usr/bin/env python3
"""
Problem 4: Complete Pipeline Implementation
女胎异常检测的完整流程 - 从数据预处理到最终评估

Based on plan/prob4.md specifications:
- 数据预处理 (Data Preprocessing)
- 划分与不平衡处理 (Split & Imbalance Handling) 
- 模型训练与交叉验证 (Model Training & Cross-Validation)
- 最终测试评估 (Final Test Evaluation)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Core ML imports
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, precision_recall_curve, brier_score_loss
)
# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - will use only Logistic Regression")
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports (with error handling)
try:
    from data.loader import load_problem4_data
    from analysis.problem4.data_preprocessing import DataPreprocessor
    DATA_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import data modules: {e}")
    print("ℹ️ Will create minimal test data for demonstration")
    DATA_MODULES_AVAILABLE = False

# Utility imports (optional)
try:
    from utils.visualization import save_figure
    from utils.statistics import calculate_confidence_intervals
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
for dir_path in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True)

print("🎯 Problem 4: Complete Pipeline Implementation")
print("=" * 60)
print(f"📁 Results will be saved to: {RESULTS_DIR}")
print(f"📊 Figures will be saved to: {FIGURES_DIR}")


def create_synthetic_problem4_data(n_samples=300, n_features=20, positive_rate=0.1, random_state=42):
    """Create synthetic data for Problem 4 when real data modules are not available"""
    np.random.seed(random_state)
    
    n_positive = int(n_samples * positive_rate)
    n_negative = n_samples - n_positive
    
    # Create feature names matching expected structure
    feature_names = [
        'Z_13', 'Z_18', 'Z_21', 'Z_X',  # Z-scores
        'GC_global', 'GC_13', 'GC_18', 'GC_21',  # GC content
        'reads', 'map_ratio', 'dup_ratio', 'unique_reads',  # QC metrics
        'BMI', 'age', 'weeks',  # Clinical features
        'max_Z', 'Z13_indicator', 'Z18_indicator', 'Z21_indicator', 'uniq_rate'  # Derived features
    ]
    
    # Generate realistic data
    data = np.random.randn(n_samples, len(feature_names))
    
    # Make Z-scores more realistic
    data[:, 0:4] = np.random.normal(0, 2, (n_samples, 4))  # Z-scores
    
    # Add some signal for positive cases
    positive_indices = np.random.choice(n_samples, n_positive, replace=False)
    
    # Make some Z-scores more extreme for positive cases
    for i in positive_indices:
        extreme_z_idx = np.random.choice(3)  # Pick one of Z13, Z18, Z21
        data[i, extreme_z_idx] = np.random.choice([-1, 1]) * np.random.uniform(3, 5)
    
    # Make other features realistic
    data[:, 4:8] = np.random.uniform(0.4, 0.6, (n_samples, 4))  # GC content
    data[:, 8] = np.random.lognormal(10, 1, n_samples)  # reads
    data[:, 9:11] = np.random.uniform(0.7, 0.95, (n_samples, 2))  # map_ratio, dup_ratio
    data[:, 11] = data[:, 8] * np.random.uniform(0.8, 0.95, n_samples)  # unique_reads
    data[:, 12] = np.random.normal(25, 5, n_samples)  # BMI
    data[:, 13] = np.random.uniform(20, 40, n_samples)  # age
    data[:, 14] = np.random.uniform(10, 25, n_samples)  # weeks
    
    # Derived features
    data[:, 15] = np.max(np.abs(data[:, 0:3]), axis=1)  # max_Z
    data[:, 16:19] = (np.abs(data[:, 0:3]) >= 3).astype(int)  # Z indicators
    data[:, 19] = data[:, 11] / data[:, 8]  # uniq_rate
    
    # Create labels
    y = np.zeros(n_samples)
    y[positive_indices] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    return {
        'raw_data': df,
        'y': y
    }


def simple_preprocess_data(raw_data, y, test_size=0.2, random_state=42):
    """Simple preprocessing when DataPreprocessor is not available"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Basic train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.values, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create sample weights (simple version)
    n_samples = len(y_train)
    n_positive = np.sum(y_train)
    n_negative = n_samples - n_positive
    
    sample_weights = np.ones(n_samples)
    sample_weights[y_train == 1] = n_negative / n_positive
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test},
        'feature_columns': list(raw_data.columns),
        'sample_weights': sample_weights
    }


class Problem4Pipeline:
    """Complete Problem 4 pipeline implementation"""
    
    def __init__(self, random_state=42, output_dir="output"):
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.figures_dir = self.output_dir / "figures"
        
        # Initialize results storage
        self.results = {}
        self.preprocessor = None
        self.best_model = None
        self.calibrator = None
        self.optimal_threshold = None
        self.feature_names = None
        
        # Set random seeds
        np.random.seed(random_state)
        
    def load_and_preprocess_data(self):
        """Step 1: 数据预处理 (Data Preprocessing)"""
        print("\n🔧 Step 1: Data Preprocessing")
        print("-" * 40)
        
        # Load raw data (with fallback)
        if DATA_MODULES_AVAILABLE:
            print("📁 Loading real Problem 4 data...")
            data_dict = load_problem4_data()
            raw_data = data_dict['raw_data']
            y = data_dict['y']
            
            print(f"✅ Loaded {raw_data.shape[0]} samples, {raw_data.shape[1]} features")
            print(f"📊 Positive cases: {np.sum(y)} ({np.mean(y):.1%})")
            
            # Apply comprehensive preprocessing according to plan
            self.preprocessor = DataPreprocessor()
            processed_results = self.preprocessor.preprocess_for_modeling(
                raw_data, y, test_size=0.2, random_state=self.random_state,
                apply_winsorization=True,
                apply_feature_engineering=True,
                apply_correlation_filtering=True
            )
        else:
            print("📁 Creating synthetic data for demonstration...")
            data_dict = create_synthetic_problem4_data(
                n_samples=300, positive_rate=0.1, random_state=self.random_state
            )
            raw_data = data_dict['raw_data']
            y = data_dict['y']
            
            print(f"✅ Created {raw_data.shape[0]} samples, {raw_data.shape[1]} features")
            print(f"📊 Positive cases: {np.sum(y)} ({np.mean(y):.1%})")
            
            # Apply simple preprocessing
            processed_results = simple_preprocess_data(
                raw_data, y, test_size=0.2, random_state=self.random_state
            )
        
        # Store preprocessed data
        self.train_dict = processed_results['train']
        self.test_dict = processed_results['test'] 
        self.feature_names = processed_results['feature_columns']
        self.sample_weights = processed_results['sample_weights']
        
        # Create calibration split from training data
        from sklearn.model_selection import train_test_split
        
        # Get group info for stratified split
        if 'groups' in processed_results:
            train_groups = processed_results['groups'][:len(self.train_dict['y'])]
        else:
            train_groups = np.arange(len(self.train_dict['y']))  # Fallback
            
        # Split training into train/calibration (90/10)
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=self.random_state)
            train_idx, calib_idx = next(sgkf.split(
                self.train_dict['X'], self.train_dict['y'], train_groups
            ))
        except:
            # Fallback to regular stratified split
            train_idx, calib_idx = train_test_split(
                np.arange(len(self.train_dict['y'])), 
                test_size=0.1, 
                random_state=self.random_state,
                stratify=self.train_dict['y']
            )
        
        # Create train/calibration splits
        self.train_split = {
            'X': self.train_dict['X'][train_idx],
            'y': self.train_dict['y'][train_idx]
        }
        self.calib_split = {
            'X': self.train_dict['X'][calib_idx], 
            'y': self.train_dict['y'][calib_idx]
        }
        
        print(f"✅ Data splits created:")
        print(f"   Training: {len(self.train_split['y'])} ({np.sum(self.train_split['y'])} positive)")
        print(f"   Calibration: {len(self.calib_split['y'])} ({np.sum(self.calib_split['y'])} positive)")
        print(f"   Test: {len(self.test_dict['y'])} ({np.sum(self.test_dict['y'])} positive)")
        print(f"   Features: {len(self.feature_names)}")
        
        # Store preprocessing results
        self.results['preprocessing'] = {
            'raw_samples': raw_data.shape[0],
            'raw_features': raw_data.shape[1],
            'processed_features': len(self.feature_names),
            'train_samples': len(self.train_split['y']),
            'calib_samples': len(self.calib_split['y']), 
            'test_samples': len(self.test_dict['y']),
            'positive_rate': {
                'train': np.mean(self.train_split['y']),
                'calib': np.mean(self.calib_split['y']),
                'test': np.mean(self.test_dict['y'])
            },
            'feature_names': self.feature_names
        }
        
        return self
    
    def train_models_with_cv(self):
        """Step 2-3: 模型训练与交叉验证 (Model Training & Cross-Validation)"""
        print("\n🎯 Step 2-3: Model Training & Cross-Validation")
        print("-" * 40)
        
        # Prepare for cross-validation
        X_train = self.train_split['X']
        y_train = self.train_split['y']
        
        # Create groups for GroupKFold (fallback to indices if no group info)
        try:
            if hasattr(self.preprocessor, 'groups'):
                groups = self.preprocessor.groups[:len(y_train)]
            else:
                groups = np.arange(len(y_train)) % 20  # Create artificial groups
        except:
            groups = np.arange(len(y_train)) % 20
        
        # Cross-validation setup
        cv = GroupKFold(n_splits=5)
        
        # Calculate class weights for imbalance handling
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1.0, 1: n_neg / n_pos}
        scale_pos_weight = n_neg / n_pos
        
        print(f"📊 Class imbalance: {n_pos}/{len(y_train)} positive ({np.mean(y_train):.1%})")
        print(f"⚖️ Class weight ratio: {class_weight[1]:.2f}")
        
        # 1. Elastic-Net Logistic Regression
        print("\n🎯 Training Elastic-Net Logistic Regression...")
        
        # Standardize features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Parameter grid for Logistic Regression (from plan)
        logistic_param_grid = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            'class_weight': [class_weight]
        }
        
        logistic_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=5000,
            random_state=self.random_state
        )
        
        logistic_search = RandomizedSearchCV(
            logistic_model,
            logistic_param_grid,
            n_iter=60,
            cv=cv,
            scoring='average_precision',  # PR-AUC as primary metric
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        logistic_search.fit(X_train_scaled, y_train, groups=groups)
        best_logistic = logistic_search.best_estimator_
        
        print(f"✅ Best Logistic params: {logistic_search.best_params_}")
        print(f"📊 Best CV score (PR-AUC): {logistic_search.best_score_:.4f}")
        
        # 2. XGBoost (if available)
        xgb_search = None
        if XGBOOST_AVAILABLE:
            print("\n🎯 Training XGBoost...")
            
            # Parameter grid for XGBoost (from plan)
            xgb_param_grid = {
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'n_estimators': [200, 400, 800, 1200],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.5, 1, 2],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0.5, 1.0, 5.0],
                'scale_pos_weight': [scale_pos_weight]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            xgb_search = RandomizedSearchCV(
                xgb_model,
                xgb_param_grid,
                n_iter=100,
                cv=cv,
                scoring='average_precision',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            xgb_search.fit(X_train, y_train, groups=groups)
            best_xgb = xgb_search.best_estimator_
            
            print(f"✅ Best XGBoost params: {xgb_search.best_params_}")
            print(f"📊 Best CV score (PR-AUC): {xgb_search.best_score_:.4f}")
        else:
            print("\n⚠️ Skipping XGBoost - not available")
        
        # Model selection: Choose best based on PR-AUC
        if xgb_search is None or logistic_search.best_score_ >= xgb_search.best_score_:
            self.best_model = best_logistic
            self.model_type = 'logistic'
            self.X_transform = lambda X: scaler.transform(X)
            best_score = logistic_search.best_score_
            print(f"\n🏆 Selected Model: Elastic-Net Logistic (PR-AUC: {best_score:.4f})")
        else:
            self.best_model = best_xgb
            self.model_type = 'xgboost'
            self.X_transform = lambda X: X  # No scaling needed
            best_score = xgb_search.best_score_
            print(f"\n🏆 Selected Model: XGBoost (PR-AUC: {best_score:.4f})")
        
        # Store model training results
        model_training_results = {
            'selected_model': self.model_type,
            'best_cv_score': best_score,
            'logistic_results': {
                'best_params': logistic_search.best_params_,
                'best_score': logistic_search.best_score_,
                'cv_results': logistic_search.cv_results_
            }
        }
        
        # Add XGBoost results if available
        if xgb_search is not None:
            model_training_results['xgboost_results'] = {
                'best_params': xgb_search.best_params_,
                'best_score': xgb_search.best_score_,
                'cv_results': xgb_search.cv_results_
            }
        else:
            model_training_results['xgboost_results'] = {'status': 'not_available'}
        
        self.results['model_training'] = model_training_results
        
        # Store scalers and transformers
        self.scaler = scaler
        
        return self
    
    def calibrate_and_optimize_threshold(self):
        """Step 4: 概率校准与阈值优化 (Calibration & Threshold Optimization)"""
        print("\n📊 Step 4: Probability Calibration & Threshold Optimization")
        print("-" * 40)
        
        # Retrain best model on full training data
        X_train_full = self.train_dict['X']
        y_train_full = self.train_dict['y']
        X_train_transformed = self.X_transform(X_train_full)
        
        print(f"🔧 Retraining {self.model_type} on full training data...")
        self.best_model.fit(X_train_transformed, y_train_full)
        
        # Probability calibration using calibration set
        X_calib = self.calib_split['X'] 
        y_calib = self.calib_split['y']
        X_calib_transformed = self.X_transform(X_calib)
        
        print("🎯 Applying Platt scaling calibration...")
        
        # Handle XGBoost calibration issues
        if self.model_type == 'xgboost':
            # For XGBoost, use the model directly without calibration due to compatibility issues
            print("⚠️ Using XGBoost without additional calibration (already outputs probabilities)")
            self.calibrator = self.best_model
        else:
            # For other models, apply standard calibration
            self.calibrator = CalibratedClassifierCV(
                self.best_model, 
                method='sigmoid',  # Platt scaling
                cv='prefit'  # Use pre-fitted model
            )
            self.calibrator.fit(X_calib_transformed, y_calib)
        
        # Threshold optimization: maximize recall subject to FPR ≤ 1%
        print("🔍 Optimizing threshold for FPR ≤ 1%...")
        
        # Get calibrated probabilities  
        if self.model_type == 'xgboost':
            calib_probs = self.calibrator.predict_proba(X_calib_transformed)[:, 1]
        else:
            calib_probs = self.calibrator.predict_proba(X_calib_transformed)[:, 1]
        
        # Find optimal threshold
        thresholds = np.linspace(0.01, 0.99, 100)
        best_recall = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            pred_calib = (calib_probs >= threshold).astype(int)
            
            # Calculate FPR and Recall
            tn = np.sum((y_calib == 0) & (pred_calib == 0))
            fp = np.sum((y_calib == 0) & (pred_calib == 1))
            fn = np.sum((y_calib == 1) & (pred_calib == 0))
            tp = np.sum((y_calib == 1) & (pred_calib == 1))
            
            if fp + tn > 0:
                fpr = fp / (fp + tn)
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    
                    # Check FPR constraint and update best threshold
                    if fpr <= 0.01 and recall > best_recall:
                        best_recall = recall
                        best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        print(f"✅ Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"📊 Expected recall @ FPR≤1%: {best_recall:.1%}")
        
        # Store calibration results
        self.results['calibration'] = {
            'optimal_threshold': self.optimal_threshold,
            'calibration_recall': best_recall,
            'calibration_samples': len(y_calib),
            'calibration_positive_rate': np.mean(y_calib)
        }
        
        return self
    
    def evaluate_on_test_set(self):
        """Step 5: 最终测试评估 (Final Test Evaluation)"""
        print("\n🏆 Step 5: Final Test Set Evaluation")
        print("-" * 40)
        
        # Transform test data
        X_test = self.test_dict['X']
        y_test = self.test_dict['y']
        X_test_transformed = self.X_transform(X_test)
        
        # Get calibrated probabilities and predictions
        if self.model_type == 'xgboost':
            test_probs = self.calibrator.predict_proba(X_test_transformed)[:, 1]
        else:
            test_probs = self.calibrator.predict_proba(X_test_transformed)[:, 1]
        test_pred = (test_probs >= self.optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        
        metrics = {
            'confusion_matrix': [[tn, fp], [fn, tp]],
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'roc_auc': roc_auc_score(y_test, test_probs),
            'pr_auc': average_precision_score(y_test, test_probs),
            'brier_score': brier_score_loss(y_test, test_probs),
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Same as precision
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
        
        # Baseline comparison: |Z| ≥ 3 rule
        print("📊 Calculating baseline |Z|≥3 rule performance...")
        
        # Extract Z-scores for baseline (assume first 3 features are Z13, Z18, Z21)
        if len(self.feature_names) >= 3:
            z_features = X_test[:, :3]  # Z13, Z18, Z21
            max_abs_z = np.max(np.abs(z_features), axis=1)
            baseline_pred = (max_abs_z >= 3).astype(int)
            
            # Calculate baseline metrics
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, baseline_pred).ravel()
            baseline_metrics = {
                'confusion_matrix': [[int(tn_b), int(fp_b)], [int(fn_b), int(tp_b)]],
                'precision': tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0,
                'recall': tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0,
                'fpr': fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0,
                'f1_score': 2 * tp_b / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0
            }
        else:
            baseline_metrics = {'error': 'Insufficient Z-score features for baseline'}
        
        # Display results
        print("\n🎯 FINAL TEST RESULTS")
        print("=" * 50)
        print(f"📊 Selected Model: {self.model_type.upper()}")
        print(f"📊 Test Samples: {len(y_test)} ({np.sum(y_test)} positive)")
        print(f"📊 Optimal Threshold: {self.optimal_threshold:.4f}")
        
        print(f"\n🏆 MAIN RESULTS:")
        print(f"   Recall: {metrics['recall']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   FPR: {metrics['fpr']:.1%}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"   PR-AUC: {metrics['pr_auc']:.3f}")
        
        print(f"\n📊 CONFUSION MATRIX: [[TN, FP], [FN, TP]]")
        print(f"   {metrics['confusion_matrix']}")
        
        print(f"\n🏥 CLINICAL METRICS:")
        print(f"   PPV (Positive Predictive Value): {metrics['ppv']:.1%}")
        print(f"   NPV (Negative Predictive Value): {metrics['npv']:.1%}")
        print(f"   Detected Cases: {tp}/{tp+fn} ({metrics['recall']:.1%})")
        
        if 'error' not in baseline_metrics:
            print(f"\n📊 BASELINE COMPARISON (|Z|≥3 rule):")
            print(f"   Baseline Recall: {baseline_metrics['recall']:.1%}")
            print(f"   Baseline Precision: {baseline_metrics['precision']:.1%}")
            print(f"   Baseline FPR: {baseline_metrics['fpr']:.1%}")
            print(f"   Improvement: {metrics['recall']-baseline_metrics['recall']:+.1%} recall")
        
        # Store final results
        self.results['final_test'] = {
            'model_type': self.model_type,
            'threshold': self.optimal_threshold,
            'test_samples': len(y_test),
            'test_positive_rate': np.mean(y_test),
            'metrics': metrics,
            'baseline_metrics': baseline_metrics,
            'predictions': {
                'probabilities': test_probs.tolist(),
                'predictions': test_pred.tolist(),
                'true_labels': y_test.tolist()
            }
        }
        
        return self
    
    def generate_model_interpretation(self):
        """Generate model interpretation and feature importance"""
        print("\n🔍 Generating Model Interpretation")
        print("-" * 40)
        
        interpretation = {}
        
        if self.model_type == 'logistic':
            # Logistic regression coefficients
            coefficients = self.best_model.coef_[0]
            intercept = self.best_model.intercept_[0]
            
            # Calculate odds ratios and confidence intervals
            odds_ratios = np.exp(coefficients)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'odds_ratio': odds_ratios,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            interpretation = {
                'model_type': 'logistic',
                'intercept': float(intercept),
                'feature_importance': feature_importance.to_dict('records'),
                'top_features': feature_importance.head(10).to_dict('records')
            }
            
            print("📊 Top 10 Most Important Features (Logistic Regression):")
            print(feature_importance.head(10)[['feature', 'coefficient', 'odds_ratio']].to_string(index=False))
            
        elif self.model_type == 'xgboost':
            # XGBoost feature importance
            feature_importance_gain = self.best_model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance_gain
            }).sort_values('importance', ascending=False)
            
            interpretation = {
                'model_type': 'xgboost', 
                'feature_importance': feature_importance.to_dict('records'),
                'top_features': feature_importance.head(10).to_dict('records')
            }
            
            print("📊 Top 10 Most Important Features (XGBoost):")
            print(feature_importance.head(10).to_string(index=False))
        
        self.results['interpretation'] = interpretation
        return self
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating Visualizations")
        print("-" * 40)
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Problem 4: Model Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Get test predictions for visualization
        X_test_transformed = self.X_transform(self.test_dict['X'])
        if self.model_type == 'xgboost':
            test_probs = self.calibrator.predict_proba(X_test_transformed)[:, 1]
        else:
            test_probs = self.calibrator.predict_proba(X_test_transformed)[:, 1]
        y_test = self.test_dict['y']
        
        # Confusion Matrix
        test_pred = (test_probs >= self.optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        auc_score = roc_auc_score(y_test, test_probs)
        axes[0,1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc_score:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'r--')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # Precision-Recall Curve  
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)
        axes[1,0].plot(recall, precision, 'g-', label=f'PR (AUC = {pr_auc:.3f})')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        
        # Probability Distribution
        pos_probs = test_probs[y_test == 1]
        neg_probs = test_probs[y_test == 0]
        axes[1,1].hist(neg_probs, bins=30, alpha=0.7, label='Negative', color='blue')
        axes[1,1].hist(pos_probs, bins=30, alpha=0.7, label='Positive', color='red')
        axes[1,1].axvline(self.optimal_threshold, color='black', linestyle='--', label='Threshold')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Probability Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        dashboard_path = self.figures_dir / "p4_final_model_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model dashboard saved: {dashboard_path}")
        
        # 2. Feature Importance Plot
        if 'interpretation' in self.results:
            plt.figure(figsize=(12, 8))
            
            if self.model_type == 'logistic':
                # Plot top features by absolute coefficient
                top_features = pd.DataFrame(self.results['interpretation']['top_features'])
                
                plt.barh(range(len(top_features)), top_features['abs_coefficient'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Absolute Coefficient')
                plt.title('Top 10 Most Important Features (Logistic Regression)')
                
            elif self.model_type == 'xgboost':
                # Plot top features by importance
                top_features = pd.DataFrame(self.results['interpretation']['top_features'])
                
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Most Important Features (XGBoost)')
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
            importance_path = self.figures_dir / "p4_feature_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Feature importance plot saved: {importance_path}")
        
        # 3. Calibration Curve
        if len(np.unique(y_test)) > 1:  # Only if we have both classes
            try:
                from sklearn.calibration import calibration_curve
                plt.figure(figsize=(10, 8))
                
                # Create calibration plot
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, test_probs, n_bins=10
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
                plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title("Calibration Plot (Reliability Diagram)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                calibration_path = self.figures_dir / "p4_calibration_curve.png"
                plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Calibration plot saved: {calibration_path}")
            except ImportError:
                print("⚠️ Calibration curve not available in this sklearn version")
        
        return self
    
    def save_results(self):
        """Save all results to output directory"""
        print("\n💾 Saving Results")
        print("-" * 40)
        
        # Save comprehensive results as JSON
        results_file = self.results_dir / "problem4_final_results.json"
        
        def convert_numpy(obj):
            """Convert numpy types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"✅ Results saved: {results_file}")
        
        # Save trained model and pipeline components
        model_file = self.results_dir / "problem4_trained_model.pkl"
        model_components = {
            'best_model': self.best_model,
            'calibrator': self.calibrator,
            'scaler': getattr(self, 'scaler', None),
            'optimal_threshold': self.optimal_threshold,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'X_transform': self.X_transform.__code__.co_code  # Save transform function signature
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_components, f)
        print(f"✅ Model saved: {model_file}")
        
        # Save detailed metrics CSV
        metrics_df = pd.DataFrame([{
            'model_type': self.model_type,
            'threshold': self.optimal_threshold,
            'test_samples': len(self.test_dict['y']),
            **self.results['final_test']['metrics']
        }])
        
        metrics_file = self.results_dir / "problem4_final_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Metrics saved: {metrics_file}")
        
        # Save feature importance
        if 'interpretation' in self.results:
            importance_df = pd.DataFrame(self.results['interpretation']['feature_importance'])
            importance_file = self.results_dir / "problem4_feature_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            print(f"✅ Feature importance saved: {importance_file}")
        
        return self
    
    def run_complete_pipeline(self):
        """Run the complete Problem 4 pipeline"""
        print("🚀 Starting Complete Problem 4 Pipeline")
        print("=" * 60)
        
        try:
            # Execute pipeline steps
            (self
             .load_and_preprocess_data()
             .train_models_with_cv()
             .calibrate_and_optimize_threshold()
             .evaluate_on_test_set()
             .generate_model_interpretation()
             .create_visualizations()
             .save_results())
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Final summary
            final_metrics = self.results['final_test']['metrics']
            print(f"\n🏆 FINAL SUMMARY:")
            print(f"   Selected Model: {self.model_type.upper()}")
            print(f"   Test Recall: {final_metrics['recall']:.1%}")
            print(f"   Test Precision: {final_metrics['precision']:.1%}")
            print(f"   Test FPR: {final_metrics['fpr']:.1%}")
            print(f"   ROC-AUC: {final_metrics['roc_auc']:.3f}")
            print(f"   PR-AUC: {final_metrics['pr_auc']:.3f}")
            print(f"\n📁 All results saved to: {self.results_dir}")
            print(f"📊 Visualizations saved to: {self.figures_dir}")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = Problem4Pipeline(random_state=RANDOM_STATE, output_dir=OUTPUT_DIR)
    results = pipeline.run_complete_pipeline()
    
    if results:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Check {RESULTS_DIR} for detailed results")
        print(f"📈 Check {FIGURES_DIR} for visualizations")
    else:
        print(f"\n❌ Pipeline failed - check error messages above")
        sys.exit(1)
