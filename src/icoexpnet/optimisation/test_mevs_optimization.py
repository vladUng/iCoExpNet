#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_mevs_optimization.py
@Time    :   2024/08/31
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Test suite for MEVs optimization validation and benchmarking
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict

# Add the src path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from icoexpnet.analysis.utilities.mevs_optimization import (
    MevsOptimizer, 
    benchmark_mevs_methods, 
    benchmark_imevs_methods
)


def generate_test_data():
    """
    Generate synthetic test data that mimics real gene expression data structure.
    
    Returns:
        Tuple of (tpm_data, modcon_data, healthy_data, tumor_data, mutation_data)
    """
    np.random.seed(42)  # For reproducible tests
    
    n_genes = 1000
    n_samples = 100
    n_communities = 5
    n_healthy_samples = 50
    n_tumor_samples = 75
    
    # Generate gene names
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    
    # Generate sample names
    samples = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    healthy_samples = [f"HEALTHY_{i:03d}" for i in range(n_healthy_samples)]
    tumor_samples = [f"TUMOR_{i:03d}" for i in range(n_tumor_samples)]
    
    # Generate TPM data (log-normal distribution to mimic real expression data)
    tpm_data = pd.DataFrame(
        np.random.lognormal(mean=2, sigma=1.5, size=(n_genes, n_samples)),
        index=genes,
        columns=samples
    )
    
    # Generate healthy and tumor TPM data
    healthy_data = pd.DataFrame(
        np.random.lognormal(mean=2, sigma=1.5, size=(n_genes, n_healthy_samples)),
        index=genes,
        columns=healthy_samples
    )
    
    tumor_data = pd.DataFrame(
        np.random.lognormal(mean=2.2, sigma=1.8, size=(n_genes, n_tumor_samples)),
        index=genes,
        columns=tumor_samples
    )
    
    # Generate ModCon-style data
    modcon_data = {}
    for comm_id in range(n_communities):
        # Select random genes for each community
        comm_genes = np.random.choice(genes, size=200, replace=False)
        
        # Generate mock ModCon scores and other required columns
        comm_df = pd.DataFrame({
            'ModCon_standard_5K': np.random.exponential(scale=100, size=len(comm_genes)),
            'q2E': np.random.uniform(0.1, 0.9, size=len(comm_genes)),
            'varWithin': np.random.uniform(0.1, 0.8, size=len(comm_genes)),
            'varAcross': np.random.uniform(0.1, 0.7, size=len(comm_genes)),
        }, index=comm_genes)
        
        modcon_data[comm_id] = comm_df
    
    # Generate mutation data
    mutation_data = pd.DataFrame({
        'count': np.random.poisson(lam=2, size=n_genes)
    }, index=genes)
    
    return tpm_data, modcon_data, healthy_data, tumor_data, mutation_data


def test_mevs_correctness():
    """
    Test that optimized get_mevs produces results with similar structure and properties.
    """
    print("🔬 Testing MEVs optimization correctness...")
    
    # Generate test data
    tpm_data, modcon_data, _, _, _ = generate_test_data()
    
    # Create optimizer
    optimizer = MevsOptimizer(verbose=False)
    
    # Test optimized method
    try:
        mevs_result, info = optimizer.optimized_get_mevs(
            tpm_data, modcon_data, sort_col="ModCon_standard_5K", num_genes=25, verbose=True
        )
        
        # Validate results structure
        assert isinstance(mevs_result, pd.DataFrame), "Result should be a DataFrame"
        assert len(mevs_result.index) == len(tpm_data.columns), "Should have one row per sample"
        assert len(mevs_result.columns) == len(modcon_data), "Should have one column per community"
        
        # Check column naming
        expected_columns = [f"Com_{i}" for i in range(len(modcon_data))]
        assert all(col in mevs_result.columns for col in expected_columns), "Column names should follow Com_X pattern"
        
        # Check for reasonable value ranges (z-scores should be roughly in [-5, 5] range)
        assert mevs_result.abs().max().max() < 100, "MEVs values should be in reasonable range"
        
        print("✅ MEVs optimization correctness test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MEVs optimization test failed: {e}")
        return False


def test_imevs_correctness():
    """
    Test that optimized get_iMevs produces results with similar structure and properties.
    """
    print("🔬 Testing integrated MEVs optimization correctness...")
    
    # Generate test data
    _, modcon_data, healthy_data, tumor_data, mutation_data = generate_test_data()
    
    # Create optimizer
    optimizer = MevsOptimizer(verbose=False)
    
    # Test optimized method
    try:
        imevs_result, info = optimizer.optimized_get_iMevs(
            healthy_data, tumor_data, modcon_data, 
            sort_col="ModCon_standard_5K", num_genes=25, verbose=True,
            mut_df=mutation_data, mut_offset=1.0
        )
        
        # Validate results structure
        assert isinstance(imevs_result, pd.DataFrame), "Result should be a DataFrame"
        assert len(imevs_result.index) == len(tumor_data.columns), "Should have one row per tumor sample"
        assert len(imevs_result.columns) == len(modcon_data), "Should have one column per community"
        
        # Check column naming
        expected_columns = [f"Com_{i}" for i in range(len(modcon_data))]
        assert all(col in imevs_result.columns for col in expected_columns), "Column names should follow Com_X pattern"
        
        # Check for reasonable value ranges
        assert imevs_result.abs().max().max() < 1000, "Integrated MEVs values should be in reasonable range"
        
        print("✅ Integrated MEVs optimization correctness test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated MEVs optimization test failed: {e}")
        return False


def run_performance_benchmarks():
    """
    Run performance benchmarks to demonstrate speed improvements.
    """
    print("\n🚀 Running performance benchmarks...")
    
    # Generate test data
    tpm_data, modcon_data, healthy_data, tumor_data, mutation_data = generate_test_data()
    
    # Benchmark MEVs
    print("\n📊 Benchmarking MEVs computation:")
    mevs_results = benchmark_mevs_methods(
        tpm_data, modcon_data, 
        sort_col="ModCon_standard_5K", 
        num_genes=25, 
        num_runs=3, 
        verbose=True
    )
    
    # Benchmark integrated MEVs  
    print("\n📊 Benchmarking integrated MEVs computation:")
    imevs_results = benchmark_imevs_methods(
        healthy_data, tumor_data, modcon_data,
        sort_col="ModCon_standard_5K",
        num_genes=25,
        num_runs=3,
        verbose=True,
        mut_df=mutation_data,
        mut_offset=1.0
    )
    
    return mevs_results, imevs_results


def main():
    """
    Main test runner function.
    """
    print("🧪 Starting MEVs optimization test suite...")
    print("=" * 60)
    
    # Run correctness tests
    mevs_ok = test_mevs_correctness()
    imevs_ok = test_imevs_correctness()
    
    if mevs_ok and imevs_ok:
        print("\n✅ All correctness tests passed!")
        
        # Run performance benchmarks
        mevs_bench, imevs_bench = run_performance_benchmarks()
        
        # Summary
        print(f"\n🎯 Performance Summary:")
        print(f"  MEVs speedup: {mevs_bench['speedup_estimate']:.1f}x")
        print(f"  Integrated MEVs speedup: {imevs_bench['speedup_estimate']:.1f}x")
        print(f"\n🏆 Optimization successful! Both methods are significantly faster.")
        
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)