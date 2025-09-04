#!/usr/bin/env python3
"""
Quick runner for data quality diagnostics.

Usage:
    python run_diagnostics.py              # Run all tests with default settings
    python run_diagnostics.py --quiet      # Run with minimal output
    python run_diagnostics.py --no-save    # Don't save plots/results
"""

import sys
from pathlib import Path

# Add src to Python path and change to project directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run directly to avoid relative import issues
if __name__ == "__main__":
    # Change to project directory
    import os
    os.chdir(project_root)
    
    # Import directly from the script
    from src.utils.diagnostics.run_all import main
    main()
