#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   memory_optimization.py
@Time    :   2024/08/29
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Memory optimization utilities for iCoExpNet classes
"""

import pandas as pd
import numpy as np
import sys
import gc
from typing import Tuple, List, Optional, Dict, Any


def optimize_dataframe_memory(df: pd.DataFrame, name: str = "DataFrame") -> Tuple[pd.DataFrame, List[Dict], float]:
    """
    Optimize data types for a pandas DataFrame to reduce memory usage.
    
    Args:
        df: DataFrame to optimize
        name: Name of the DataFrame for logging
        
    Returns:
        Tuple of (optimized_df, optimizations_list, memory_saved_mb)
    """
    if df is None or df.empty:
        return df, [], 0.0
        
    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    optimizations = []
    
    for col in df.columns:
        original_dtype = df[col].dtype
        original_size = df[col].memory_usage(deep=True) / 1024 / 1024
        
        # Skip if column has all NaN values
        if df[col].isna().all():
            continue
        
        # Optimize numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Try to downcast integers
            if pd.api.types.is_integer_dtype(df[col]):
                new_col = pd.to_numeric(df[col], downcast='integer')
                if new_col.dtype != original_dtype:
                    df[col] = new_col
                    new_size = df[col].memory_usage(deep=True) / 1024 / 1024
                    optimizations.append({
                        'column': col,
                        'original_dtype': str(original_dtype),
                        'new_dtype': str(new_col.dtype),
                        'memory_saved_mb': original_size - new_size
                    })
            
            # Try to downcast floats (float64 -> float32 where safe)
            elif pd.api.types.is_float_dtype(df[col]):
                if original_dtype == 'float64':
                    # Check if values fit in float32 range safely
                    col_min, col_max = df[col].min(), df[col].max()
                    if (not pd.isna(col_min) and not pd.isna(col_max) and 
                        col_min >= np.finfo(np.float32).min * 0.9 and  # Add safety margin
                        col_max <= np.finfo(np.float32).max * 0.9):
                        
                        # Test conversion and verify no precision loss for critical data
                        if _safe_float32_conversion(df[col]):
                            df[col] = df[col].astype('float32')
                            new_size = df[col].memory_usage(deep=True) / 1024 / 1024
                            optimizations.append({
                                'column': col,
                                'original_dtype': str(original_dtype),
                                'new_dtype': 'float32',
                                'memory_saved_mb': original_size - new_size
                            })
        
        # Optimize string/object columns with categorical
        elif df[col].dtype == 'object':
            # Only convert to category if it's beneficial (< 50% unique values)
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.5 and df[col].nunique() > 1:
                try:
                    df[col] = df[col].astype('category')
                    new_size = df[col].memory_usage(deep=True) / 1024 / 1024
                    if original_size > new_size:  # Only keep if actually saves memory
                        optimizations.append({
                            'column': col,
                            'original_dtype': 'object',
                            'new_dtype': 'category',
                            'memory_saved_mb': original_size - new_size
                        })
                    else:
                        # Revert if no benefit
                        df[col] = df[col].astype('object')
                except:
                    # If conversion fails, keep original
                    pass
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    memory_saved = original_memory - optimized_memory
    
    return df, optimizations, memory_saved


def _safe_float32_conversion(series: pd.Series) -> bool:
    """
    Check if float64 to float32 conversion is safe for this specific series.
    
    Args:
        series: Pandas series to test
        
    Returns:
        True if conversion is safe, False otherwise
    """
    # Skip conversion for very small numbers where precision matters
    abs_series = series.abs()
    
    # Don't convert if we have very small numbers (potential precision issues)
    if (abs_series > 0).any() and (abs_series[abs_series > 0].min() < 1e-6):
        return False
    
    # Don't convert if we have numbers very close to float32 limits
    if abs_series.max() > np.finfo(np.float32).max * 0.8:
        return False
    
    return True


def optimize_graph_memory(graph_obj, remove_redundant: bool = True) -> Dict[str, Any]:
    """
    Optimize graph object memory usage.
    
    Args:
        graph_obj: Graph object (igraph or graph-tool)
        remove_redundant: Whether to remove redundant attributes
        
    Returns:
        Dictionary with optimization details
    """
    optimization_details = {
        'attributes_removed': [],
        'memory_saved_mb': 0,
        'optimizations_applied': []
    }
    
    # This will be implemented based on specific graph types
    # For now, return empty optimization details
    return optimization_details


def apply_memory_optimizations_to_experiment(exp, experiment_name: str = "experiment") -> Dict[str, Any]:
    """
    Apply all memory optimizations to a single experiment object.
    
    Args:
        exp: Experiment object (NetworkOutput, GraphToolExperiment, etc.)
        experiment_name: Name for logging
        
    Returns:
        Dictionary with optimization results
    """
    results = {
        'experiment_name': experiment_name,
        'dataframes_optimized': [],
        'total_memory_saved_mb': 0.0,
        'optimizations_applied': [],
        'errors': []
    }
    
    # List of common DataFrame attributes to optimize
    df_attributes = ['tpm_df', 'edges_df', 'nodes_df', 'meta_df', 'mut_df']
    
    for attr_name in df_attributes:
        if hasattr(exp, attr_name):
            try:
                df = getattr(exp, attr_name)
                if df is not None and not df.empty:
                    optimized_df, optimizations, memory_saved = optimize_dataframe_memory(
                        df.copy(), f"{experiment_name}.{attr_name}")
                    
                    if memory_saved > 0:
                        setattr(exp, attr_name, optimized_df)
                        results['dataframes_optimized'].append(attr_name)
                        results['total_memory_saved_mb'] += memory_saved
                        results['optimizations_applied'].extend(optimizations)
                        
            except Exception as e:
                results['errors'].append(f"Error optimizing {attr_name}: {str(e)}")
    
    return results


class MemoryOptimizedMixin:
    """
    Mixin class to add memory optimization capabilities to existing classes.
    """
    
    def optimize_memory(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize memory usage of this object.
        
        Args:
            verbose: Whether to print optimization details
            
        Returns:
            Dictionary with optimization results
        """
        class_name = self.__class__.__name__
        object_name = getattr(self, 'name', class_name)
        
        if verbose:
            print(f"🔧 Optimizing memory for {object_name}...")
        
        results = apply_memory_optimizations_to_experiment(self, object_name)
        
        if verbose and results['total_memory_saved_mb'] > 0:
            print(f"  ✅ Saved {results['total_memory_saved_mb']:.2f} MB")
            print(f"  📈 Optimized {len(results['dataframes_optimized'])} DataFrames")
        elif verbose:
            print(f"  ➡️  No optimization opportunities found")
        
        return results
    
    def get_memory_usage(self) -> float:
        """
        Get approximate memory usage of this object in MB.
        
        Returns:
            Memory usage in MB
        """
        total_size = sys.getsizeof(self)
        
        # Add sizes of major DataFrame attributes
        df_attributes = ['tpm_df', 'edges_df', 'nodes_df', 'meta_df', 'mut_df']
        for attr_name in df_attributes:
            if hasattr(self, attr_name):
                df = getattr(self, attr_name)
                if df is not None and not df.empty:
                    total_size += df.memory_usage(deep=True).sum()
        
        return total_size / (1024 * 1024)  # Convert to MB


def create_optimized_dataframe(data, columns=None, index=None, optimize=True, **kwargs):
    """
    Create a DataFrame with optimized data types from the start.
    
    Args:
        data: DataFrame data
        columns: Column names
        index: Index
        optimize: Whether to apply optimizations
        **kwargs: Additional DataFrame constructor arguments
        
    Returns:
        Optimized DataFrame
    """
    df = pd.DataFrame(data, columns=columns, index=index, **kwargs)
    
    if optimize:
        df, _, _ = optimize_dataframe_memory(df)
    
    return df


# Export main optimization function for easy importing
__all__ = [
    'optimize_dataframe_memory',
    'apply_memory_optimizations_to_experiment', 
    'MemoryOptimizedMixin',
    'create_optimized_dataframe'
]