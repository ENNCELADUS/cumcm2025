#!/usr/bin/env python3
"""
Example script showing how to use the trained Problem 4 model for predictions.

This script demonstrates how to load the trained model and use it to make predictions
on new data, following the same preprocessing pipeline.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Configuration
MODEL_PATH = "output/results/problem4_trained_model.pkl"
RESULTS_PATH = "output/results/problem4_final_results.json"

def load_trained_model():
    """Load the trained model and pipeline components"""
    print("📁 Loading trained model...")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_components = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {model_components['model_type']}")
        print(f"   Features: {len(model_components['feature_names'])}")
        print(f"   Optimal threshold: {model_components['optimal_threshold']:.4f}")
        
        return model_components
    
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("🔧 Please run 'python run_problem4_pipeline.py' first")
        return None

def load_results_metadata():
    """Load results metadata for context"""
    try:
        with open(RESULTS_PATH, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"⚠️ Results file not found: {RESULTS_PATH}")
        return None

def create_example_data(feature_names, n_samples=5):
    """Create example data for demonstration"""
    print(f"📊 Creating example data with {len(feature_names)} features...")
    
    # Create realistic-looking data
    np.random.seed(123)
    n_features = len(feature_names)
    
    # Initialize with random data
    data = np.random.randn(n_samples, n_features)
    
    # Make specific features more realistic based on names
    for i, name in enumerate(feature_names):
        if 'Z_' in name or 'max_Z' in name:
            # Z-scores: mean=0, some extreme values
            data[:, i] = np.random.normal(0, 2, n_samples)
            if 'max_Z' in name:
                data[:, i] = np.abs(data[:, i])  # max_Z should be positive
        elif 'GC' in name:
            # GC content: realistic range 40-60%
            data[:, i] = np.random.uniform(0.4, 0.6, n_samples)
        elif name in ['map_ratio', 'dup_ratio']:
            # Ratios: 0.7-0.95 range
            data[:, i] = np.random.uniform(0.7, 0.95, n_samples)
        elif name == 'reads':
            # Read counts: log-normal distribution
            data[:, i] = np.random.lognormal(10, 1, n_samples)
        elif name == 'unique_reads':
            # Unique reads: slightly less than total reads
            reads_idx = feature_names.index('reads') if 'reads' in feature_names else i
            data[:, i] = data[:, reads_idx] * np.random.uniform(0.8, 0.95, n_samples)
        elif name == 'BMI':
            # BMI: realistic range
            data[:, i] = np.random.normal(25, 5, n_samples)
        elif name == 'age':
            # Age: 20-40 range
            data[:, i] = np.random.uniform(20, 40, n_samples)
        elif name == 'weeks':
            # Pregnancy weeks: 10-25 range
            data[:, i] = np.random.uniform(10, 25, n_samples)
        elif 'indicator' in name:
            # Binary indicators
            data[:, i] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        elif name == 'uniq_rate':
            # Unique rate: ratio
            data[:, i] = np.random.uniform(0.8, 0.95, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    # Add some extreme cases for demonstration
    if n_samples >= 2:
        # Make one sample high-risk (extreme Z-score)
        z_cols = [col for col in feature_names if col.startswith('Z_') and not 'indicator' in col]
        if z_cols:
            df.loc[0, z_cols[0]] = 4.5  # Extreme Z-score
            if 'max_Z' in feature_names:
                df.loc[0, 'max_Z'] = 4.5
            # Update indicators
            indicator_cols = [col for col in feature_names if 'indicator' in col]
            if indicator_cols:
                df.loc[0, indicator_cols[0]] = 1
    
    return df

def make_predictions(model_components, X_data):
    """Make predictions using the trained model"""
    print(f"🎯 Making predictions on {len(X_data)} samples...")
    
    # Extract components
    model = model_components['calibrator']  # This is the calibrated model
    model_type = model_components['model_type']
    threshold = model_components['optimal_threshold']
    feature_names = model_components['feature_names']
    
    # Apply the same transformation as during training
    if model_type == 'logistic' and 'scaler' in model_components:
        scaler = model_components['scaler']
        X_transformed = scaler.transform(X_data.values)
    else:
        X_transformed = X_data.values
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_transformed)[:, 1]
    else:
        # Fallback for some model types
        probabilities = model.predict(X_transformed)
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'sample_id': range(len(X_data)),
        'probability': probabilities,
        'prediction': predictions,
        'risk_level': ['High Risk' if p == 1 else 'Low Risk' for p in predictions],
        'confidence': ['High' if abs(prob - 0.5) > 0.3 else 'Moderate' if abs(prob - 0.5) > 0.1 else 'Low' 
                      for prob in probabilities]
    })
    
    return results_df, probabilities, predictions

def display_predictions(X_data, results_df, model_components):
    """Display prediction results in a user-friendly format"""
    print("\n🏥 PREDICTION RESULTS")
    print("=" * 60)
    
    feature_names = model_components['feature_names']
    threshold = model_components['optimal_threshold']
    
    print(f"📊 Model: {model_components['model_type'].upper()}")
    print(f"📊 Threshold: {threshold:.4f}")
    print(f"📊 Total samples: {len(X_data)}")
    print(f"📊 High-risk predictions: {np.sum(results_df['prediction'])}")
    
    print(f"\n📋 INDIVIDUAL PREDICTIONS:")
    print("-" * 60)
    
    for idx, row in results_df.iterrows():
        print(f"Sample {row['sample_id']+1:2d}: {row['risk_level']:9s} "
              f"(prob: {row['probability']:.3f}, confidence: {row['confidence']})")
        
        # Show key features for high-risk cases
        if row['prediction'] == 1:
            z_features = [col for col in feature_names if col.startswith('Z_')]
            z_values = [f"{col}={X_data.iloc[idx][col]:.2f}" for col in z_features[:3]]
            print(f"          Key features: {', '.join(z_values)}")
        print()
    
    # Summary statistics
    high_risk_count = np.sum(results_df['prediction'])
    avg_prob = np.mean(results_df['probability'])
    
    print(f"📊 SUMMARY:")
    print(f"   High-risk rate: {high_risk_count}/{len(X_data)} ({high_risk_count/len(X_data):.1%})")
    print(f"   Average probability: {avg_prob:.3f}")
    print(f"   Confidence distribution: {results_df['confidence'].value_counts().to_dict()}")

def main():
    """Main function demonstrating model usage"""
    print("🎯 Problem 4: Using Trained Model for Predictions")
    print("=" * 60)
    
    # Load trained model
    model_components = load_trained_model()
    if model_components is None:
        return
    
    # Load results for context
    results = load_results_metadata()
    if results:
        final_metrics = results.get('final_test', {}).get('metrics', {})
        print(f"\n📊 Model Performance (from training):")
        print(f"   Test Recall: {final_metrics.get('recall', 0):.1%}")
        print(f"   Test Precision: {final_metrics.get('precision', 0):.1%}")
        print(f"   Test FPR: {final_metrics.get('fpr', 0):.1%}")
        print(f"   ROC-AUC: {final_metrics.get('roc_auc', 0):.3f}")
    
    # Create example data
    feature_names = model_components['feature_names']
    example_data = create_example_data(feature_names, n_samples=5)
    
    print(f"\n📊 Example input data:")
    print(example_data.round(3))
    
    # Make predictions
    results_df, probabilities, predictions = make_predictions(model_components, example_data)
    
    # Display results
    display_predictions(example_data, results_df, model_components)
    
    print(f"\n💡 USAGE NOTES:")
    print(f"   📋 Input data must have exactly {len(feature_names)} features")
    print(f"   📋 Feature order must match: {feature_names[:3]}... (first 3 shown)")
    print(f"   📋 Predictions are binary: 0=Low Risk, 1=High Risk")
    print(f"   📋 Probability values range from 0.0 to 1.0")
    print(f"   📋 Threshold {model_components['optimal_threshold']:.4f} optimized for recall with FPR≤1%")
    
    return results_df

if __name__ == "__main__":
    results = main()
