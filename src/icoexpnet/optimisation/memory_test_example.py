#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Simple example showing how to use memory optimization in your existing iCoExpNet experiments
Place this file in your project root directory and run it to test memory optimization
"""

import sys
import os
sys.path.insert(0, 'src')

from icoexpnet.core.main import iCoExpNet

def main():
    """
    Example of how to enable memory tracking in your existing experiments
    """
    print("🧪 Memory Optimization Example")
    print("=" * 50)
    
    # Your existing experiment configuration
    data_base = "data/"
    results_path = "results/example_test/"
    
    print("🔄 Creating iCoExpNet experiment with memory optimization...")
    
    # Create experiment with memory tracking enabled (just add enable_memory_tracking=True)
    exp = iCoExpNet(
        exp_name="example_experiment",
        ge_file="test_data_10000_genes.tsv",
        input_folder=data_base,
        output_folder=results_path,
        gene_subset_file="TF_names_v_1.01.txt",
        mut_file="test_mutation_data.tsv",
        genes_kept=2000,  # Adjust as needed
        edges_pg=3,
        edges_sel=6,
        modifier_type="standard",
        enable_memory_tracking=True  # 👈 This enables memory optimization and tracking
    )
    
    print("\n🚀 Running experiment pipeline...")
    
    # Run your experiment as usual - memory optimization happens automatically
    memory_summary = exp.run()
    
    print("\n✅ Experiment completed!")
    print(f"📁 Results saved in: {results_path}")
    
    if memory_summary is not None:
        print(f"📊 Memory usage log: {exp.memory_tracker.tracker.log_file}")
    
    return exp, memory_summary

if __name__ == "__main__":
    # Check if data files exist
    required_files = [
        "data/test_data_10000_genes.tsv",
        "data/TF_names_v_1.01.txt",
        "data/test_mutation_data.tsv"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"  • {file}")
        print("\nPlease ensure test data files are available.")
        sys.exit(1)
    
    try:
        exp, memory_summary = main()
        print("\n🎉 Memory optimization test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nThis might be because graph-tool is not installed.")
        print("Memory optimization still works for data loading and correlation computation!")
        sys.exit(1)