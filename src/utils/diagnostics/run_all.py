"""
Run All Diagnostics

Convenience script to run all data quality diagnostics at once and provide
a comprehensive summary report.
"""

import pandas as pd
from pathlib import Path
import sys
import argparse

# Import functions directly to avoid relative import issues
from .filter_impact import test_filter_impact
from .selection_bias import test_selection_bias
from .gc_correlation import test_gc_correlations

def run_all_diagnostics(data_file_path="src/data/data.xlsx", save_outputs=True, verbose=True):
    """
    Run all data quality diagnostics and provide summary report.
    
    Args:
        data_file_path: Path to the Excel data file
        save_outputs: Whether to save plots and CSV results
        verbose: Whether to print detailed results
    
    Returns:
        dict: Summary results from all tests
    """
    
    if verbose:
        print("🔬 COMPREHENSIVE DATA QUALITY DIAGNOSTICS")
        print("=" * 80)
        print("Running all tests to assess impact of filtering on statistical relationships...")
        print()
    
    results = {}
    
    # Test 1: Filter Impact Analysis
    if verbose:
        print("🔧 TEST 1: FILTER IMPACT ANALYSIS")
        print("-" * 50)
    
    try:
        filter_results, raw_data, clean_data = test_filter_impact(
            data_file_path=data_file_path, 
            verbose=verbose
        )
        results['filter_impact'] = {
            'results_df': filter_results,
            'raw_data': raw_data,
            'clean_data': clean_data
        }
        
        if save_outputs:
            output_path = Path("output/results/filter_impact_results.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            filter_results.to_csv(output_path, index=False)
            if verbose:
                print(f"   Results saved to: {output_path}")
                
    except Exception as e:
        if verbose:
            print(f"   ❌ Error in filter impact analysis: {e}")
        results['filter_impact'] = {'error': str(e)}
    
    if verbose:
        print("\n" + "="*80 + "\n")
    
    # Test 2: Selection Bias Analysis  
    if verbose:
        print("🎯 TEST 2: SELECTION BIAS ANALYSIS")
        print("-" * 50)
    
    try:
        original_data, final_data, bias_detected = test_selection_bias(
            data_file_path=data_file_path,
            save_plots=save_outputs,
            verbose=verbose
        )
        results['selection_bias'] = {
            'original_data': original_data,
            'final_data': final_data,
            'bias_detected': bias_detected
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Error in selection bias analysis: {e}")
        results['selection_bias'] = {'error': str(e)}
    
    if verbose:
        print("\n" + "="*80 + "\n")
    
    # Test 3: GC Correlation Analysis
    if verbose:
        print("🧬 TEST 3: GC CORRELATION ANALYSIS")
        print("-" * 50)
    
    try:
        corr_df, analysis_data, gc_bias_detected = test_gc_correlations(
            data_file_path=data_file_path,
            save_plots=save_outputs,
            verbose=verbose
        )
        results['gc_correlation'] = {
            'correlation_df': corr_df,
            'analysis_data': analysis_data,
            'bias_detected': gc_bias_detected
        }
        
        if save_outputs:
            output_path = Path("output/results/gc_correlation_results.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            corr_df.to_csv(output_path, index=False)
            if verbose:
                print(f"   Results saved to: {output_path}")
                
    except Exception as e:
        if verbose:
            print(f"   ❌ Error in GC correlation analysis: {e}")
        results['gc_correlation'] = {'error': str(e)}
    
    # Generate comprehensive summary
    if verbose:
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE SUMMARY")
        print("="*80)
        
        # Summary from filter impact
        if 'filter_impact' in results and 'results_df' in results['filter_impact']:
            filter_df = results['filter_impact']['results_df']
            if len(filter_df) > 0:
                initial_r = filter_df.iloc[0]['r_weeks']
                final_r = filter_df.iloc[-1]['r_weeks']
                change_pct = ((final_r - initial_r) / abs(initial_r)) * 100 if initial_r != 0 else 0
                retention = filter_df.iloc[-1]['N'] / filter_df.iloc[0]['N'] * 100
                
                print(f"🔧 Filter Impact:")
                print(f"   • Correlation change: {initial_r:.4f} → {final_r:.4f} ({change_pct:+.1f}%)")
                print(f"   • Sample retention: {retention:.1f}%")
                
                if abs(change_pct) > 20:
                    print(f"   • ⚠️  SUBSTANTIAL correlation change!")
                elif abs(change_pct) > 10:
                    print(f"   • ⚠️  Moderate correlation change")
                else:
                    print(f"   • ✅ Minimal correlation change")
        
        # Summary from selection bias
        if 'selection_bias' in results and 'bias_detected' in results['selection_bias']:
            bias_detected = results['selection_bias']['bias_detected']
            print(f"\n🎯 Selection Bias:")
            if bias_detected:
                print(f"   • ⚠️  BIAS DETECTED in GC filtering")
                print(f"   • Removed samples differ systematically from kept samples")
            else:
                print(f"   • ✅ NO significant selection bias detected")
                print(f"   • GC filtering appears random with respect to key variables")
        
        # Summary from GC correlation
        if 'gc_correlation' in results and 'bias_detected' in results['gc_correlation']:
            gc_bias = results['gc_correlation']['bias_detected']
            print(f"\n🧬 GC Content Correlations:")
            if gc_bias:
                print(f"   • ⚠️  GC correlates with key variables")
                print(f"   • Filtering on GC may introduce bias")
            else:
                print(f"   • ✅ GC doesn't correlate with key variables")
                print(f"   • GC filtering unlikely to cause bias")
        
        # Overall recommendation
        print(f"\n💡 OVERALL RECOMMENDATION:")
        
        all_bias_indicators = []
        if 'selection_bias' in results and 'bias_detected' in results['selection_bias']:
            all_bias_indicators.append(results['selection_bias']['bias_detected'])
        if 'gc_correlation' in results and 'bias_detected' in results['gc_correlation']:
            all_bias_indicators.append(results['gc_correlation']['bias_detected'])
        
        if any(all_bias_indicators):
            print("   ⚠️  CAUTION: Some bias indicators detected")
            print("   • Consider reporting both filtered and unfiltered results")
            print("   • Use inverse probability weighting if bias is substantial")
            print("   • Consider relaxing GC thresholds")
        else:
            print("   ✅ FILTERING APPEARS SOUND")
            print("   • No major bias indicators detected")
            print("   • Strict quality filters likely improve rather than harm analysis")
            print("   • Weak correlations likely reflect true biological relationships")
        
        print("="*80)
    
    return results

def main():
    """Command line interface for running diagnostics."""
    parser = argparse.ArgumentParser(description="Run comprehensive data quality diagnostics")
    parser.add_argument("--data-file", default="src/data/data.xlsx", 
                       help="Path to Excel data file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save plots and CSV outputs")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")
    
    args = parser.parse_args()
    
    results = run_all_diagnostics(
        data_file_path=args.data_file,
        save_outputs=not args.no_save,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print("\n🎯 Diagnostics completed!")

if __name__ == "__main__":
    main()
