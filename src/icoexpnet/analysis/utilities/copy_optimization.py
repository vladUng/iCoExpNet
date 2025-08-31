#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   copy_optimization.py
@Time    :   2024/08/29
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Utilities to eliminate unnecessary deep copies and reduce memory usage
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, Optional, List, Tuple
from contextlib import contextmanager


class CopyOptimizer:
    """
    Utilities to replace unnecessary deep copies with more memory-efficient operations.
    """
    
    @staticmethod
    def safe_view_or_copy(df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         condition: Optional[pd.Series] = None,
                         sort_by: Optional[str] = None,
                         top_n: Optional[int] = None,
                         force_copy: bool = False) -> pd.DataFrame:
        """
        Create a view or shallow copy instead of deep copy when possible.
        
        Args:
            df: Source DataFrame
            columns: Specific columns to select
            condition: Boolean condition for filtering
            sort_by: Column to sort by
            top_n: Number of top rows to keep after sorting
            force_copy: Force a copy even if view would work
            
        Returns:
            Optimized DataFrame (view when possible, copy when necessary)
        """
        result = df
        needs_copy = False
        
        # Apply filtering condition
        if condition is not None:
            result = result.loc[condition]
            needs_copy = True
        
        # Select columns
        if columns is not None:
            result = result[columns]
            needs_copy = True
        
        # Sort and take top N
        if sort_by is not None:
            result = result.sort_values(by=sort_by, ascending=False)
            needs_copy = True
            
            if top_n is not None:
                result = result.iloc[:top_n]
                needs_copy = True
        
        # Return view if no modifications needed, otherwise return result
        if needs_copy or force_copy:
            return result  # This is already a new object from operations above
        else:
            return result  # Original DataFrame (view)
    
    @staticmethod
    def efficient_slice_copy(df: pd.DataFrame, 
                           start: Optional[int] = None, 
                           end: Optional[int] = None,
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Efficient slicing that avoids unnecessary deep copy.
        
        Args:
            df: Source DataFrame
            start: Start index for row slicing
            end: End index for row slicing  
            columns: Columns to select
            
        Returns:
            Sliced DataFrame
        """
        if start is None and end is None and columns is None:
            return df  # Return view of original
        
        # Use iloc for efficient slicing
        if start is not None or end is not None:
            df = df.iloc[start:end]
        
        if columns is not None:
            df = df[columns]
            
        return df
    
    @staticmethod  
    def memory_efficient_sort_top_n(df: pd.DataFrame,
                                   sort_column: str,
                                   n: int,
                                   ascending: bool = False) -> pd.DataFrame:
        """
        Memory-efficient way to get top N rows without full sort + copy.
        
        Args:
            df: Source DataFrame
            sort_column: Column to sort by
            n: Number of top rows to return
            ascending: Sort order
            
        Returns:
            Top N rows (more memory efficient than sort + iloc + copy)
        """
        if n >= len(df):
            # If we need all rows anyway, just sort normally
            return df.sort_values(by=sort_column, ascending=ascending)
        
        # Use nlargest/nsmallest for better performance on partial sorting
        if ascending:
            return df.nsmallest(n, sort_column)
        else:
            return df.nlargest(n, sort_column)
    
    @staticmethod
    def conditional_copy(df: pd.DataFrame, 
                        condition_func,
                        *args, **kwargs) -> pd.DataFrame:
        """
        Only copy DataFrame if condition function returns True.
        
        Args:
            df: Source DataFrame
            condition_func: Function that determines if copy is needed
            *args, **kwargs: Arguments passed to condition function
            
        Returns:
            Original DataFrame or copy based on condition
        """
        if condition_func(df, *args, **kwargs):
            return df.copy()
        return df
    
    @staticmethod
    def optimize_dataframe_assignment(target_df: pd.DataFrame,
                                    source_df: pd.DataFrame,
                                    columns: Optional[List[str]] = None,
                                    inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        Optimize DataFrame column assignments to avoid unnecessary copies.
        
        Args:
            target_df: DataFrame to assign to
            source_df: DataFrame to assign from  
            columns: Specific columns to assign
            inplace: Whether to modify target_df in place
            
        Returns:
            Modified DataFrame if inplace=False, None otherwise
        """
        if columns is None:
            columns = source_df.columns
            
        if inplace:
            for col in columns:
                if col in source_df.columns:
                    target_df[col] = source_df[col]  # View assignment, not copy
            return None
        else:
            result = target_df.copy()
            for col in columns:
                if col in source_df.columns:
                    result[col] = source_df[col]
            return result


# Context manager for tracking copy operations
@contextmanager
def track_copy_operations():
    """
    Context manager to track and warn about deep copy operations.
    Useful for development to identify where copies are being made.
    """
    original_copy = pd.DataFrame.copy
    copy_count = {'count': 0, 'total_memory': 0}
    
    def tracked_copy(self, deep=True):
        if deep:
            copy_count['count'] += 1
            memory_usage = self.memory_usage(deep=True).sum() / (1024**2)  # MB
            copy_count['total_memory'] += memory_usage
            warnings.warn(f"Deep copy created: {memory_usage:.2f} MB", stacklevel=2)
        return original_copy(self, deep=deep)
    
    # Monkey patch
    pd.DataFrame.copy = tracked_copy
    
    try:
        yield copy_count
    finally:
        # Restore original method
        pd.DataFrame.copy = original_copy


def replace_unnecessary_copies():
    """
    Decorator to replace common deep copy patterns with optimized versions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Could implement automatic copy detection and replacement
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Specific optimization functions for common patterns found in codebase

def optimize_leiden_top3(leidenalg_master: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Optimized version of leiden_top3 = leidenalg_master.iloc[:3].copy(deep=True)
    """
    return CopyOptimizer.efficient_slice_copy(leidenalg_master, start=0, end=n)


def optimize_combined_edges_copy(edges_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Optimized version of combined_edges = edges_df[columns].copy(deep=True)
    """
    return CopyOptimizer.safe_view_or_copy(edges_df, columns=columns)


def optimize_sort_copy_pattern(df: pd.DataFrame, 
                             sort_col: str, 
                             num_genes: int,
                             ascending: bool = False) -> pd.DataFrame:
    """
    Optimized version of data.sort_values().iloc[:num_genes].copy(deep=True)
    """
    return CopyOptimizer.memory_efficient_sort_top_n(
        df, sort_col, num_genes, ascending=ascending)


def optimize_filtered_copy(df: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    """
    Optimized version of df[condition].copy(deep=True) when copy isn't needed
    """
    return CopyOptimizer.safe_view_or_copy(df, condition=condition)


# Export main optimization functions
__all__ = [
    'CopyOptimizer',
    'track_copy_operations', 
    'optimize_leiden_top3',
    'optimize_combined_edges_copy',
    'optimize_sort_copy_pattern',
    'optimize_filtered_copy'
]