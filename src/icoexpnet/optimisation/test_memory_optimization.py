#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Test script to verify memory optimizations in iCoExpNet pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import psutil
import time

# Add parent directories to path to import icoexpnet modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from icoexpnet.core.main import iCoExpNet

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_memory_optimization():
    """Test memory optimization by running a small iCoExpNet experiment"""
    
    # Configuration paths - from src/icoexpnet/optimisation/ to project root
    data_base = "../../../data/"
    results_path = "../../../results/memory_test/"
    
    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)
    
    print("🧪 Testing Memory Optimization in iCoExpNet")
    print("=" * 60)
    
    # Test parameters
    test_configs = [
        {
            "name": "WITHOUT memory tracking", 
            "enable_memory_tracking": False,
            "description": "Baseline run without memory optimization tracking"
        },
        {
            "name": "WITH memory tracking", 
            "enable_memory_tracking": True,
            "description": "Optimized run with memory tracking and optimization"
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n📊 Running test {config['name']}")
        print(f"📝 {config['description']}")
        print("-" * 50)
        
        # Record start memory
        start_memory = get_memory_usage()
        start_time = time.time()
        
        try:
            # Create iCoExpNet instance
            exp = iCoExpNet(
                exp_name="memory_test",
                ge_file="test_data_10000_genes.tsv",
                input_folder=data_base,
                output_folder=results_path,
                gene_subset_file="TF_names_v_1.01.txt",
                mut_file="test_mutation_data.tsv",
                genes_kept=1000,  # Use fewer genes for faster testing
                edges_pg=3,
                edges_sel=6,
                modifier_type="standard",
                enable_memory_tracking=config["enable_memory_tracking"]
            )
            
            # Run just the data loading and correlation computation part
            print("🔄 Loading data...")
            ge_df, sel_ge, mut_df = exp.load_data(
                ge_path=exp.input_ge_file,
                sel_ge_path=exp.sel_ge_file, 
                mut_path=exp.mut_file
            )
            
            data_load_memory = get_memory_usage()
            print(f"📊 Memory after data loading: {data_load_memory:.2f} MB")
            
            print("🔄 Filtering data...")
            ge_df_filtered = exp.filter_data(ge_df, num_genes=exp.genes_kept)
            
            filter_memory = get_memory_usage()
            print(f"📊 Memory after filtering: {filter_memory:.2f} MB")
            
            print("🔄 Computing correlation matrix...")
            corr_df = exp.corr_matrix(df=ge_df_filtered)
            
            corr_memory = get_memory_usage()
            end_time = time.time()
            
            # Record results
            results[config["name"]] = {
                "start_memory": start_memory,
                "data_load_memory": data_load_memory,
                "filter_memory": filter_memory, 
                "final_memory": corr_memory,
                "peak_memory": corr_memory,
                "execution_time": end_time - start_time,
                "memory_increase": corr_memory - start_memory
            }
            
            print(f"📊 Final memory usage: {corr_memory:.2f} MB")
            print(f"📊 Memory increase: {corr_memory - start_memory:.2f} MB")
            print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
            
            # If memory tracking was enabled, print the summary
            if hasattr(exp, 'memory_tracker') and exp.memory_tracker:
                print("\n📈 Memory Tracking Summary:")
                exp.memory_tracker.tracker.print_summary()
                
                # Save memory log for analysis
                memory_summary = exp.memory_tracker.tracker.get_memory_summary()
                if not memory_summary.empty:
                    log_path = f"{results_path}/memory_log_{config['name'].replace(' ', '_').lower()}.tsv"
                    memory_summary.to_csv(log_path, sep='\t', index=False)
                    print(f"📝 Memory log saved to: {log_path}")
            
        except Exception as e:
            print(f"❌ Error in {config['name']}: {str(e)}")
            results[config["name"]] = {
                "error": str(e),
                "start_memory": start_memory,
                "final_memory": get_memory_usage()
            }
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 MEMORY OPTIMIZATION TEST RESULTS")
    print("=" * 60)
    
    if len(results) >= 2:
        without_tracking = None
        with_tracking = None
        
        for name, data in results.items():
            if "error" not in data:
                print(f"\n🔍 {name}:")
                print(f"  • Start memory: {data['start_memory']:.2f} MB")
                print(f"  • Final memory: {data['final_memory']:.2f} MB")
                print(f"  • Memory increase: {data['memory_increase']:.2f} MB")
                print(f"  • Execution time: {data['execution_time']:.2f}s")
                
                if "WITHOUT" in name:
                    without_tracking = data
                elif "WITH" in name:
                    with_tracking = data
            else:
                print(f"\n❌ {name}: {data['error']}")
        
        # Calculate improvement
        if without_tracking and with_tracking and "memory_increase" in without_tracking and "memory_increase" in with_tracking:
            memory_improvement = without_tracking["memory_increase"] - with_tracking["memory_increase"]
            improvement_percent = (memory_improvement / without_tracking["memory_increase"]) * 100 if without_tracking["memory_increase"] > 0 else 0
            
            print(f"\n🎯 OPTIMIZATION RESULTS:")
            print(f"  • Memory improvement: {memory_improvement:.2f} MB")
            print(f"  • Improvement percentage: {improvement_percent:.1f}%")
            
            if memory_improvement > 0:
                print(f"  ✅ Memory optimization is WORKING! 🎉")
            else:
                print(f"  ⚠️  No significant memory improvement detected")
        
        # Check for memory tracking logs
        log_files = [f for f in os.listdir(results_path) if f.startswith("memory_") and f.endswith(".tsv")]
        if log_files:
            print(f"\n📝 Memory tracking logs created:")
            for log_file in log_files:
                print(f"  • {os.path.join(results_path, log_file)}")
                
    print(f"\n📁 Test results saved in: {results_path}")
    return results

if __name__ == "__main__":
    # Check if required data files exist - from src/icoexpnet/optimisation/
    required_files = [
        "../../../data/test_data_10000_genes.tsv",
        "../../../data/TF_names_v_1.01.txt", 
        "../../../data/test_mutation_data.tsv"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"  • {file}")
        print("\nPlease ensure test data files are available before running this test.")
        sys.exit(1)
    
    # Run the test
    try:
        results = test_memory_optimization()
        print("\n✅ Memory optimization test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)