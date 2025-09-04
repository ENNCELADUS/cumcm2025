"""
Visualization utilities for NIPT analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

from ..config.settings import FIGURES_DIR

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


class NIPTVisualizer:
    """
    Visualization utilities for NIPT analysis.
    
    Provides methods for creating various plots and charts for data exploration
    and results presentation.
    """
    
    def __init__(self, save_dir: Optional[Path] = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save figures. If None, uses default from config.
            figsize: Default figure size for plots.
        """
        self.save_dir = save_dir or FIGURES_DIR
        self.figsize = figsize
        
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                    hue: Optional[str] = None, title: Optional[str] = None,
                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a scatter plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            hue: Column name for color coding
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, alpha=0.7)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def correlation_heatmap(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                           title: str = "Correlation Matrix", save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            data: DataFrame containing the data
            columns: List of columns to include. If None, uses all numeric columns.
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if columns is None:
            # Select numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[columns]
        
        correlation = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def distribution_plot(self, data: pd.DataFrame, column: str, 
                         by_group: Optional[str] = None, title: Optional[str] = None,
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a distribution plot (histogram + KDE).
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot distribution for
            by_group: Column name to group by (creates separate distributions)
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if by_group:
            for group in data[by_group].unique():
                if pd.notna(group):
                    subset = data[data[by_group] == group][column].dropna()
                    sns.histplot(subset, kde=True, alpha=0.6, label=str(group), ax=ax)
            ax.legend()
        else:
            sns.histplot(data[column].dropna(), kde=True, ax=ax)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Distribution of {column.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
        
        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def box_plot(self, data: pd.DataFrame, x: str, y: str,
                title: Optional[str] = None, save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a box plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis (categorical)
            y: Column name for y-axis (numerical)
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.boxplot(data=data, x=x, y=y, ax=ax)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
        
        # Rotate x-axis labels if they're long
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def line_plot(self, data: pd.DataFrame, x: str, y: str, 
                 hue: Optional[str] = None, title: Optional[str] = None,
                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a line plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            hue: Column name for color coding
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, marker='o')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def regression_plot(self, data: pd.DataFrame, x: str, y: str,
                       title: Optional[str] = None, save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a regression plot with confidence interval.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.regplot(data=data, x=x, y=y, ax=ax, scatter_kws={'alpha': 0.6})
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def create_subplots(self, nrows: int, ncols: int, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a figure with subplots.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size. If None, uses default.
            
        Returns:
            Tuple of (figure, axes array)
        """
        if figsize is None:
            figsize = (self.figsize[0] * ncols, self.figsize[1] * nrows)
            
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        return fig, axes
    
    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save figure to file.
        
        Args:
            fig: matplotlib Figure object
            filename: Name of the file (without extension)
            dpi: Resolution for saved figure
        """
        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Add .png extension if not present
        if not filename.endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg')):
            filename += '.png'
            
        filepath = self.save_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
        
    def set_style(self, style: str = 'whitegrid', palette: str = 'husl'):
        """
        Set the plotting style.
        
        Args:
            style: Seaborn style name
            palette: Color palette name
        """
        sns.set_style(style)
        sns.set_palette(palette)
        
    @staticmethod
    def show_all():
        """Show all open figures."""
        plt.show()
        
    @staticmethod
    def close_all():
        """Close all figures to free memory."""
        plt.close('all')
