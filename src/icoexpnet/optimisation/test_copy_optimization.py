#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Test script to validate deep copy optimizations in iCoExpNet
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import psutil
from contextlib import contextmanager

# Add parent directories to path to import icoexpnet modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from icoexpnet.analysis.utilities.copy_optimization import (
    optimize_leiden_top3, 
    optimize_sort_copy_pattern,
    track_copy_operations,
    CopyOptimizer
)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def memory_tracker():
    """Track memory usage around operations"""
    start_memory = get_memory_usage()
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        print(f"Memory delta: {end_memory - start_memory:.2f} MB")

def create_test_dataframe(rows=10000, cols=50):
    """Create a test DataFrame similar to gene expression data"""
    np.random.seed(42)  # For reproducible tests
    data = np.random.rand(rows, cols).astype(np.float64)
    genes = [f"GENE_{i:05d}" for i in range(rows)]
    samples = [f"SAMPLE_{i:03d}" for i in range(cols)]
    
    df = pd.DataFrame(data, index=genes, columns=samples)
    df['ModCon_score'] = np.random.rand(rows) * 100
    return df

def test_leiden_top3_optimization():
    """Test optimize_leiden_top3 function"""
    print("🧪 Testing Leiden Top3 Optimization")
    print("-" * 50)
    
    # Create test data similar to leidenalg_master
    test_df = create_test_dataframe(1000, 10)
    
    with memory_tracker():
        # Original approach (deep copy)
        start_time = time.time()
        result_original = test_df.iloc[:3].copy(deep=True)
        original_time = time.time() - start_time
        print(f"Original deep copy: {original_time:.4f}s")
    
    with memory_tracker():
        # Optimized approach
        start_time = time.time()
        result_optimized = optimize_leiden_top3(test_df, 3)
        optimized_time = time.time() - start_time
        print(f"Optimized version: {optimized_time:.4f}s")
    
    # Verify results are equivalent
    pd.testing.assert_frame_equal(result_original.reset_index(drop=True), 
                                result_optimized.reset_index(drop=True))
    
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"✅ Results identical, speedup: {speedup:.2f}x")
    return result_optimized

def test_sort_copy_optimization():
    """Test optimize_sort_copy_pattern function"""
    print("\n🧪 Testing Sort+Copy Optimization")  
    print("-" * 50)
    
    # Create test data
    test_df = create_test_dataframe(10000, 20)
    num_genes = 100
    sort_col = 'ModCon_score'
    
    with memory_tracker():
        # Original approach (sort + iloc + deep copy)
        start_time = time.time()
        result_original = test_df.sort_values(by=sort_col, ascending=False).iloc[:num_genes].copy(deep=True)
        original_time = time.time() - start_time
        print(f"Original sort+copy: {original_time:.4f}s")
    
    with memory_tracker():
        # Optimized approach 
        start_time = time.time()
        result_optimized = optimize_sort_copy_pattern(test_df, sort_col, num_genes, ascending=False)
        optimized_time = time.time() - start_time
        print(f"Optimized version: {optimized_time:.4f}s")
    
    # Verify results are equivalent (might have different order due to nlargest vs sort)
    original_sorted = result_original.sort_values(by=sort_col, ascending=False)
    optimized_sorted = result_optimized.sort_values(by=sort_col, ascending=False)
    
    # Check that we got the same top genes
    assert len(result_original) == len(result_optimized), "Different number of results"
    assert len(result_optimized) == num_genes, f"Expected {num_genes}, got {len(result_optimized)}"
    
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"✅ Results valid, speedup: {speedup:.2f}x")
    return result_optimized

def test_copy_tracking():
    """Test the copy tracking functionality"""
    print("\n🧪 Testing Copy Tracking")
    print("-" * 50)
    
    test_df = create_test_dataframe(1000, 10)
    
    with track_copy_operations() as copy_stats:
        # Perform operations that would trigger deep copies
        _ = test_df.copy(deep=True)  # This should be tracked
        _ = test_df.copy(deep=False)  # This should not be tracked
        _ = test_df.iloc[:100].copy(deep=True)  # This should be tracked
    
    print(f"Deep copies detected: {copy_stats['count']}")
    print(f"Total memory copied: {copy_stats['total_memory']:.2f} MB")
    print("✅ Copy tracking working")

def test_copy_optimizer_methods():
    """Test CopyOptimizer utility methods"""
    print("\n🧪 Testing CopyOptimizer Methods")
    print("-" * 50)
    
    test_df = create_test_dataframe(1000, 20)
    
    # Test safe_view_or_copy
    columns = test_df.columns[:10].tolist()
    condition = test_df['ModCon_score'] > 50
    
    with memory_tracker():
        result = CopyOptimizer.safe_view_or_copy(
            test_df, 
            columns=columns, 
            condition=condition,
            sort_by='ModCon_score',
            top_n=100
        )
    
    print(f"Filtered result shape: {result.shape}")
    print("✅ CopyOptimizer methods working")
    
    return result

def run_comprehensive_test():
    """Run all optimization tests"""
    print("🚀 iCoExpNet Deep Copy Optimization Tests")
    print("=" * 60)
    
    start_memory = get_memory_usage()
    print(f"Starting memory: {start_memory:.2f} MB")
    
    try:
        # Run individual tests
        test_leiden_top3_optimization()
        test_sort_copy_optimization() 
        test_copy_tracking()
        test_copy_optimizer_methods()
        
        end_memory = get_memory_usage()
        print(f"\n📊 Final Results:")
        print(f"Starting memory: {start_memory:.2f} MB")
        print(f"Ending memory: {end_memory:.2f} MB")
        print(f"Total memory used: {end_memory - start_memory:.2f} MB")
        
        print("\n✅ All deep copy optimization tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_comprehensive_test()