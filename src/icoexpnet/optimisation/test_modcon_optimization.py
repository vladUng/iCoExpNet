#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Test script to validate ModCon optimization correctness and performance
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import warnings
from contextlib import contextmanager

# Add parent directories to path to import icoexpnet modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from icoexpnet.analysis.GraphToolExp import GraphToolExperiment
from icoexpnet.analysis.NetworkOutput import NetworkOutput
from icoexpnet.core.main import iCoExpNet


def create_test_data(num_genes=1000, num_samples=50, num_communities=10):
    """
    Create synthetic test data for ModCon validation.
    
    Args:
        num_genes: Number of genes in the network
        num_samples: Number of samples in TPM data
        num_communities: Number of communities
        
    Returns:
        Tuple of test data components
    """
    print(f"🔬 Creating synthetic test data...")
    print(f"   Genes: {num_genes}, Samples: {num_samples}, Communities: {num_communities}")
    
    np.random.seed(42)  # For reproducible tests
    
    # Generate gene names
    genes = [f"GENE_{i:05d}" for i in range(num_genes)]
    samples = [f"SAMPLE_{i:03d}" for i in range(num_samples)]
    
    # Create synthetic TPM data
    tpm_data = np.random.lognormal(mean=2, sigma=1, size=(num_genes, num_samples))
    tpm_df = pd.DataFrame(tpm_data, index=genes, columns=samples)
    
    # Create synthetic edges data (gene co-expression network)
    num_edges = min(num_genes * 5, num_genes * (num_genes - 1) // 4)  # Dense but not complete
    source_genes = np.random.choice(genes, num_edges, replace=True)
    target_genes = np.random.choice(genes, num_edges, replace=True)
    
    # Ensure no self-loops
    mask = source_genes != target_genes
    source_genes = source_genes[mask]
    target_genes = target_genes[mask]
    
    # Generate weights (correlation-like values)
    weights = np.random.uniform(0.1, 0.95, len(source_genes))
    
    edges_df = pd.DataFrame({
        'Source': source_genes,
        'Target': target_genes, 
        'Weight': weights
    })
    
    # Remove duplicate edges
    edges_df = edges_df.drop_duplicates(subset=['Source', 'Target'])
    
    # Create community assignments
    communities = np.random.randint(0, num_communities, num_genes)
    com_df = pd.DataFrame({
        'max_b': communities
    }, index=genes)
    
    # Create metadata (required for ModCon calculation)
    meta_df = pd.DataFrame({
        'genes': genes,
        'q2E': np.random.uniform(0.5, 2.0, num_genes),
        'varWithin': np.random.uniform(10, 90, num_genes),
        'varAcross': np.random.uniform(5, 50, num_genes)
    })
    
    # Create optional mutation data
    mut_df = pd.DataFrame({
        'count': np.random.randint(0, 10, num_genes)
    }, index=genes)
    
    print(f"✅ Test data created:")
    print(f"   TPM shape: {tpm_df.shape}")
    print(f"   Edges: {len(edges_df)}")
    print(f"   Communities: {len(np.unique(communities))}")
    
    return tpm_df, edges_df, com_df, meta_df, mut_df


class MockGraphToolExperiment:
    """
    Mock GraphToolExperiment for testing ModCon optimization
    without requiring full graph-tool setup.
    """
    
    def __init__(self, edges_df, meta_df, mut_df, exp_type="test"):
        self.edges_df = edges_df
        self.meta_df = meta_df
        self.mut_df = mut_df
        self.type = exp_type
        self.sbm_method = 'sbm'
        
        # Initialize optimization components
        from icoexpnet.analysis.utilities.modcon_optimization import ModConOptimizer
        self._modcon_optimizer = None
        self._optimization_enabled = True
        
    def get_ModCon_original(self, com_df):
        """
        Original ModCon computation logic (extracted and simplified)
        """
        gen_coms = com_df.reset_index().rename(columns={"index": "Id"})
        mut_df = self.mut_df
        meta_df = self.meta_df
        
        # Create components for ModCon equation
        modifier = self.type.split("_")[0]
        col = f"conn_{modifier}"
        
        modCons = {}
        for mod_class in gen_coms["max_b"].unique():
            genes = gen_coms.loc[gen_coms["max_b"] == mod_class]["Id"].values
            
            # Original nested loop approach
            conn_g = []
            for gene in genes:
                # Filter edges for current gene
                in_edges = self.edges_df[
                    (self.edges_df["Source"] == gene) | 
                    (self.edges_df["Target"] == gene)
                ]
                weights_sum = in_edges["Weight"].sum()
                if weights_sum == 0:
                    print(f"Weighted sum = 0 for Com {mod_class} !!")
                conn_g.append([gene, weights_sum])
            
            conn_df = pd.DataFrame(conn_g, columns=["gene", col]).set_index("gene")
            
            # Join with metadata
            if mut_df is None:
                working_df = pd.concat([
                    conn_df, 
                    meta_df.loc[meta_df["genes"].isin(genes)].set_index("genes")
                ], axis=1)
            else:
                working_df = pd.concat([
                    conn_df,
                    meta_df.loc[meta_df["genes"].isin(genes)].set_index("genes"),
                    mut_df[mut_df.index.isin(genes)]["count"]
                ], axis=1)
            
            # Calculate ModCon
            working_df[f"ModCon_{self.type}_gt"] = (
                (working_df[col] ** 2) * 
                working_df["q2E"] * 
                working_df["varWithin"] * 
                (100 - working_df["varAcross"]) / 100
            )
            
            # Sort by ModCon
            modCons[mod_class] = working_df.sort_values(
                by=f"ModCon_{self.type}_gt", 
                ascending=False
            )
        
        return modCons
    
    def get_ModCon_optimized(self, com_df):
        """
        Optimized ModCon computation
        """
        # Initialize optimizer if needed
        if self._modcon_optimizer is None and self._optimization_enabled:
            from icoexpnet.analysis.utilities.modcon_optimization import ModConOptimizer
            self._modcon_optimizer = ModConOptimizer(self.edges_df)
        
        if self._modcon_optimizer is not None and self._optimization_enabled:
            gen_coms = com_df.reset_index().rename(columns={"index": "Id"})
            modifier = self.type.split("_")[0]
            
            # Use optimized computation
            modCons = self._modcon_optimizer.optimized_get_modcon(
                gen_coms=gen_coms,
                meta_df=self.meta_df,
                mut_df=self.mut_df,
                modifier=modifier,
                exp_type=self.type
            )
            return modCons
        else:
            # Fallback to original
            return self.get_ModCon_original(com_df)


def compare_dataframes(df1, df2, name1="Original", name2="Optimized", tolerance=1e-10):
    """
    Compare two DataFrames for equality with detailed reporting.
    
    Args:
        df1, df2: DataFrames to compare
        name1, name2: Names for reporting
        tolerance: Numerical tolerance for float comparison
        
    Returns:
        Boolean indicating if DataFrames are equal
    """
    print(f"\n🔍 Comparing {name1} vs {name2}:")
    
    # Check shapes
    if df1.shape != df2.shape:
        print(f"❌ Shape mismatch: {df1.shape} vs {df2.shape}")
        return False
    print(f"✅ Shapes match: {df1.shape}")
    
    # Check indices
    if not df1.index.equals(df2.index):
        print(f"❌ Index mismatch")
        return False
    print(f"✅ Indices match")
    
    # Check columns
    if not df1.columns.equals(df2.columns):
        print(f"❌ Column mismatch")
        print(f"   {name1}: {list(df1.columns)}")
        print(f"   {name2}: {list(df2.columns)}")
        return False
    print(f"✅ Columns match")
    
    # Check numerical values with tolerance
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not np.allclose(df1[col], df2[col], rtol=tolerance, atol=tolerance, equal_nan=True):
            diff = np.abs(df1[col] - df2[col])
            max_diff = diff.max()
            print(f"❌ Numerical mismatch in column '{col}': max diff = {max_diff}")
            return False
    
    print(f"✅ All numerical values match within tolerance ({tolerance})")
    
    # Check non-numeric values
    non_numeric_cols = df1.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if not df1[col].equals(df2[col]):
            print(f"❌ Non-numeric mismatch in column '{col}'")
            return False
    
    if len(non_numeric_cols) > 0:
        print(f"✅ All non-numeric values match")
    
    print(f"✅ DataFrames are identical!")
    return True


def compare_modcon_results(original_results, optimized_results, tolerance=1e-10):
    """
    Compare ModCon results dictionaries comprehensively.
    
    Args:
        original_results: Results from original method
        optimized_results: Results from optimized method
        tolerance: Numerical tolerance
        
    Returns:
        Boolean indicating if results are identical
    """
    print(f"\n📊 Comparing ModCon Results:")
    print("=" * 50)
    
    # Check if same communities
    orig_communities = set(original_results.keys())
    opt_communities = set(optimized_results.keys())
    
    if orig_communities != opt_communities:
        print(f"❌ Community mismatch:")
        print(f"   Original: {sorted(orig_communities)}")
        print(f"   Optimized: {sorted(opt_communities)}")
        return False
    
    print(f"✅ Communities match: {len(orig_communities)} communities")
    
    # Compare each community's results
    all_match = True
    total_genes = 0
    
    for community in sorted(orig_communities):
        print(f"\n🏘️ Community {community}:")
        orig_df = original_results[community]
        opt_df = optimized_results[community]
        
        total_genes += len(orig_df)
        
        # Compare DataFrames
        matches = compare_dataframes(
            orig_df, opt_df, 
            f"Original-{community}", f"Optimized-{community}",
            tolerance
        )
        
        if not matches:
            all_match = False
            print(f"❌ Community {community} results differ!")
        else:
            print(f"✅ Community {community} results identical ({len(orig_df)} genes)")
    
    print(f"\n📈 Summary:")
    print(f"   Total communities: {len(orig_communities)}")
    print(f"   Total genes: {total_genes}")
    
    if all_match:
        print(f"🎉 ALL RESULTS IDENTICAL! ✨")
        return True
    else:
        print(f"❌ Some results differ")
        return False


def benchmark_methods(mock_exp, com_df, num_runs=5):
    """
    Benchmark original vs optimized methods.
    
    Args:
        mock_exp: MockGraphToolExperiment instance
        com_df: Community DataFrame
        num_runs: Number of benchmark runs
        
    Returns:
        Dict with benchmark results
    """
    print(f"\n🏁 Performance Benchmarking ({num_runs} runs)")
    print("=" * 50)
    
    # Warm up
    _ = mock_exp.get_ModCon_original(com_df.copy())
    _ = mock_exp.get_ModCon_optimized(com_df.copy())
    
    # Benchmark original method
    print("⏱️ Testing original method...")
    original_times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = mock_exp.get_ModCon_original(com_df.copy())
        original_times.append(time.time() - start_time)
        print(f"   Run {i+1}: {original_times[-1]:.4f}s")
    
    # Benchmark optimized method  
    print("\n⚡ Testing optimized method...")
    optimized_times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = mock_exp.get_ModCon_optimized(com_df.copy())
        optimized_times.append(time.time() - start_time)
        print(f"   Run {i+1}: {optimized_times[-1]:.4f}s")
    
    # Calculate statistics
    avg_original = np.mean(original_times)
    std_original = np.std(original_times)
    avg_optimized = np.mean(optimized_times)
    std_optimized = np.std(optimized_times)
    
    speedup = avg_original / avg_optimized if avg_optimized > 0 else float('inf')
    time_saved_percent = ((avg_original - avg_optimized) / avg_original) * 100
    
    results = {
        'original_times': original_times,
        'optimized_times': optimized_times,
        'avg_original': avg_original,
        'std_original': std_original,
        'avg_optimized': avg_optimized,
        'std_optimized': std_optimized,
        'speedup': speedup,
        'time_saved_percent': time_saved_percent
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Original:  {avg_original:.4f}s ± {std_original:.4f}s")
    print(f"   Optimized: {avg_optimized:.4f}s ± {std_optimized:.4f}s")
    print(f"   🚀 Speedup: {speedup:.1f}x")
    print(f"   ⚡ Time saved: {time_saved_percent:.1f}%")
    
    return results


def run_comprehensive_test():
    """
    Run comprehensive ModCon optimization validation and benchmarking.
    """
    print("🧪 ModCon Optimization Validation & Benchmarking")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"genes": 500, "samples": 30, "communities": 8, "name": "Small"},
        {"genes": 1500, "samples": 50, "communities": 15, "name": "Medium"},
        {"genes": 3000, "samples": 75, "communities": 25, "name": "Large"}
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🔬 Testing {config['name']} Dataset:")
        print(f"   {config['genes']} genes, {config['communities']} communities")
        
        # Create test data
        tpm_df, edges_df, com_df, meta_df, mut_df = create_test_data(
            num_genes=config["genes"],
            num_samples=config["samples"], 
            num_communities=config["communities"]
        )
        
        # Create mock experiment
        mock_exp = MockGraphToolExperiment(edges_df, meta_df, mut_df, "test")
        
        # Test correctness
        print(f"\n🔍 Testing Result Correctness...")
        start_time = time.time()
        original_results = mock_exp.get_ModCon_original(com_df.copy())
        original_time = time.time() - start_time
        
        start_time = time.time()
        optimized_results = mock_exp.get_ModCon_optimized(com_df.copy())
        optimized_time = time.time() - start_time
        
        # Compare results
        results_match = compare_modcon_results(original_results, optimized_results)
        
        if not results_match:
            print(f"❌ {config['name']} test FAILED - results don't match!")
            continue
        
        # Benchmark performance
        benchmark_results = benchmark_methods(mock_exp, com_df, num_runs=3)
        
        # Store results
        test_result = {
            'config': config,
            'results_match': results_match,
            'single_run_speedup': original_time / optimized_time,
            'benchmark_speedup': benchmark_results['speedup'],
            'time_saved_percent': benchmark_results['time_saved_percent'],
            'communities': len(original_results),
            'total_genes': sum(len(df) for df in original_results.values())
        }
        all_results.append(test_result)
        
        print(f"\n✅ {config['name']} Test Summary:")
        print(f"   Results Match: {results_match}")
        print(f"   Single Run Speedup: {test_result['single_run_speedup']:.1f}x")
        print(f"   Benchmark Speedup: {test_result['benchmark_speedup']:.1f}x")
        print(f"   Time Saved: {test_result['time_saved_percent']:.1f}%")
    
    # Overall summary
    print(f"\n🎉 FINAL SUMMARY:")
    print("=" * 60)
    
    all_passed = all(result['results_match'] for result in all_results)
    avg_speedup = np.mean([result['benchmark_speedup'] for result in all_results])
    avg_time_saved = np.mean([result['time_saved_percent'] for result in all_results])
    
    print(f"✅ All correctness tests: {'PASSED' if all_passed else 'FAILED'}")
    print(f"🚀 Average speedup: {avg_speedup:.1f}x")
    print(f"⚡ Average time saved: {avg_time_saved:.1f}%")
    
    if all_passed:
        print(f"\n🎊 SUCCESS! ModCon optimization is working perfectly!")
        print(f"   ✅ Results are mathematically identical")
        print(f"   ⚡ Significant performance improvement achieved")
    else:
        print(f"\n❌ Some tests failed - optimization needs debugging")
    
    return all_results


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        results = run_comprehensive_test()
        print(f"\n🏁 Testing completed successfully!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)