"""
Main entry point for NIPT analysis project.

This script provides a command-line interface to run different analyses
for the NIPT project problems.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data import NIPTDataLoader
from analysis.problem1 import YChromosomeCorrelationAnalyzer
from utils import NIPTVisualizer, StatisticalAnalyzer


def run_data_exploration():
    """Run basic data exploration."""
    print("=== NIPT Data Exploration ===")
    
    # Load and preprocess data
    loader = NIPTDataLoader()
    data = loader.preprocess_data()
    
    # Get summary statistics
    summary = loader.get_summary_statistics()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Split by fetal sex
    male_data, female_data = loader.split_by_fetal_sex()
    
    return data, male_data, female_data


def run_problem1_analysis(data):
    """Run Problem 1 analysis."""
    print("\n=== Problem 1: Y Chromosome Correlation Analysis ===")
    
    analyzer = YChromosomeCorrelationAnalyzer()
    
    # Generate comprehensive report
    summary = analyzer.generate_summary_report(data)
    
    print(f"\nSample Size: {summary['sample_size']}")
    print(f"Significant Correlations: {summary['key_findings']['significant_correlations']}")
    print(f"Model R² Score: {summary['key_findings']['model_performance']['r2_score']:.3f}")
    
    # Create visualizations
    analyzer.create_visualizations(data, save_figures=True)
    print("Visualizations saved to output/figures/")
    
    return summary


def run_problem2_analysis(data):
    """Run Problem 2 analysis (placeholder)."""
    print("\n=== Problem 2: BMI Grouping for Optimal NIPT Timing ===")
    print("Analysis module under development...")
    # TODO: Implement Problem 2 analysis
    return {}


def run_problem3_analysis(data):
    """Run Problem 3 analysis (placeholder)."""
    print("\n=== Problem 3: Multi-factor NIPT Timing Optimization ===")
    print("Analysis module under development...")
    # TODO: Implement Problem 3 analysis
    return {}


def run_problem4_analysis(data):
    """Run Problem 4 analysis (placeholder)."""
    print("\n=== Problem 4: Female Fetus Abnormality Detection ===")
    print("Analysis module under development...")
    # TODO: Implement Problem 4 analysis
    return {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='NIPT Analysis Project')
    parser.add_argument(
        '--problem', 
        type=int, 
        choices=[1, 2, 3, 4], 
        help='Run specific problem analysis (1-4)'
    )
    parser.add_argument(
        '--explore', 
        action='store_true', 
        help='Run data exploration'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Run all analyses'
    )
    
    args = parser.parse_args()
    
    if not any([args.problem, args.explore, args.all]):
        parser.print_help()
        return
    
    try:
        # Load data
        data, male_data, female_data = run_data_exploration()
        
        if args.explore:
            return
        
        # Run specific problem or all problems
        if args.problem == 1 or args.all:
            run_problem1_analysis(data)
        
        if args.problem == 2 or args.all:
            run_problem2_analysis(data)
        
        if args.problem == 3 or args.all:
            run_problem3_analysis(data)
        
        if args.problem == 4 or args.all:
            run_problem4_analysis(data)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
