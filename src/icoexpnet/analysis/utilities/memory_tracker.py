#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   memory_tracker.py
@Time    :   2024/08/29
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Memory tracking system for iCoExpNet pipeline to monitor optimization improvements
"""

import os
import sys
import gc
import psutil
import datetime
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class MemoryTracker:
    """
    Memory tracking system for iCoExpNet pipeline to monitor memory usage 
    and optimization improvements during network analysis.
    """
    
    def __init__(self, log_file: str = "memory_usage_log.tsv", experiment_name: str = "experiment"):
        """
        Initialize memory tracker.
        
        Args:
            log_file: Path to TSV file for logging memory usage
            experiment_name: Name of the current experiment
        """
        self.log_file = log_file
        self.experiment_name = experiment_name
        self.step_counter = 0
        self.baseline_memory = None
        self.current_checkpoint = None
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(log_file):
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create TSV log file with headers."""
        headers = [
            "timestamp", "experiment_name", "step_number", "checkpoint_name", 
            "process_memory_mb", "system_memory_usage_percent", "memory_improvement_mb",
            "memory_improvement_percent", "optimization_details", "notes"
        ]
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, 'w') as f:
            f.write('\t'.join(headers) + '\n')
        
        print(f"📝 Memory tracking log initialized: {self.log_file}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        # Force garbage collection for accurate measurement
        gc.collect()
        
        # Process memory
        process = psutil.Process(os.getpid())
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_usage_percent = system_memory.percent
        
        return {
            'process_memory_mb': process_memory_mb,
            'system_usage_percent': system_usage_percent
        }
    
    def set_baseline(self, notes: str = "Baseline measurement"):
        """Set baseline memory usage for comparison."""
        memory_info = self._get_memory_usage()
        self.baseline_memory = memory_info['process_memory_mb']
        
        self._log_checkpoint("baseline", memory_info, {}, notes)
        print(f"📊 Baseline memory set: {self.baseline_memory:.2f} MB")
    
    def checkpoint(self, name: str, optimization_details: Optional[Dict] = None, notes: str = ""):
        """
        Record a memory checkpoint.
        
        Args:
            name: Name of the checkpoint (e.g., "after_data_loading", "after_correlation")
            optimization_details: Dictionary with details about optimizations applied
            notes: Additional notes about this checkpoint
        """
        memory_info = self._get_memory_usage()
        self.current_checkpoint = name
        
        self._log_checkpoint(name, memory_info, optimization_details or {}, notes)
        
        if self.baseline_memory is not None:
            improvement_mb = self.baseline_memory - memory_info['process_memory_mb']
            improvement_percent = (improvement_mb / self.baseline_memory) * 100 if self.baseline_memory > 0 else 0
            
            print(f"🔍 {name}: {memory_info['process_memory_mb']:.2f} MB "
                  f"({improvement_mb:+.2f} MB / {improvement_percent:+.1f}% vs baseline)")
        else:
            print(f"🔍 {name}: {memory_info['process_memory_mb']:.2f} MB")
    
    def _log_checkpoint(self, checkpoint_name: str, memory_info: Dict[str, float], 
                       optimization_details: Dict, notes: str):
        """Log checkpoint data to file."""
        improvement_mb = 0
        improvement_percent = 0
        
        if self.baseline_memory is not None:
            improvement_mb = self.baseline_memory - memory_info['process_memory_mb']
            improvement_percent = (improvement_mb / self.baseline_memory) * 100 if self.baseline_memory > 0 else 0
        
        row = [
            datetime.datetime.now().isoformat(),
            self.experiment_name,
            self.step_counter,
            checkpoint_name,
            f"{memory_info['process_memory_mb']:.2f}",
            f"{memory_info['system_usage_percent']:.2f}",
            f"{improvement_mb:.2f}",
            f"{improvement_percent:.2f}",
            json.dumps(optimization_details) if optimization_details else "",
            notes
        ]
        
        with open(self.log_file, 'a') as f:
            f.write('\t'.join(str(x) for x in row) + '\n')
        
        self.step_counter += 1
    
    @contextmanager
    def track_operation(self, operation_name: str, optimization_details: Optional[Dict] = None, 
                       notes: str = ""):
        """
        Context manager to track memory usage around an operation.
        
        Usage:
            with tracker.track_operation("data_loading"):
                # Load data
                df = pd.read_csv(...)
        """
        # Before operation
        before_memory = self._get_memory_usage()['process_memory_mb']
        
        try:
            yield
        finally:
            # After operation
            after_memory = self._get_memory_usage()['process_memory_mb']
            memory_delta = after_memory - before_memory
            
            notes_with_delta = f"{notes} | Memory delta: {memory_delta:+.2f} MB" if notes else f"Memory delta: {memory_delta:+.2f} MB"
            
            self.checkpoint(f"after_{operation_name}", optimization_details, notes_with_delta)
    
    def get_memory_summary(self) -> pd.DataFrame:
        """Get summary of memory usage from log file."""
        if not os.path.exists(self.log_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.log_file, sep='\t')
            return df[df['experiment_name'] == self.experiment_name]
        except Exception as e:
            print(f"❌ Error reading memory log: {e}")
            return pd.DataFrame()
    
    def print_summary(self):
        """Print a summary of memory usage improvements."""
        summary_df = self.get_memory_summary()
        
        if summary_df.empty:
            print("📊 No memory tracking data available")
            return
        
        print(f"\n📊 Memory Usage Summary for {self.experiment_name}")
        print("=" * 60)
        
        baseline_row = summary_df[summary_df['checkpoint_name'] == 'baseline']
        if not baseline_row.empty:
            baseline_memory = baseline_row.iloc[0]['process_memory_mb']
            print(f"🔶 Baseline: {baseline_memory:.2f} MB")
        
        # Show latest checkpoint
        latest_row = summary_df.iloc[-1]
        latest_memory = latest_row['process_memory_mb']
        latest_improvement = latest_row['memory_improvement_mb']
        latest_percent = latest_row['memory_improvement_percent']
        
        print(f"📈 Latest ({latest_row['checkpoint_name']}): {latest_memory:.2f} MB")
        print(f"💾 Total improvement: {latest_improvement:.2f} MB ({latest_percent:.1f}%)")
        
        # Show top improvements
        improvements = summary_df[summary_df['memory_improvement_mb'] > 0].sort_values(
            'memory_improvement_mb', ascending=False)
        
        if not improvements.empty:
            print(f"\n🎯 Top Memory Improvements:")
            for _, row in improvements.head(3).iterrows():
                print(f"  • {row['checkpoint_name']}: {row['memory_improvement_mb']:.2f} MB "
                      f"({row['memory_improvement_percent']:.1f}%)")


class PipelineMemoryTracker:
    """
    Specialized memory tracker for iCoExpNet pipeline with predefined checkpoints.
    """
    
    def __init__(self, experiment_name: str, output_folder: str):
        """
        Initialize pipeline memory tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_folder: Output folder for saving logs
        """
        log_path = os.path.join(output_folder, "memory_usage_log.tsv")
        self.tracker = MemoryTracker(log_path, experiment_name)
        self.optimization_results = {}
    
    def start_tracking(self):
        """Start memory tracking with baseline measurement."""
        self.tracker.set_baseline("Pipeline started - baseline memory usage")
    
    def track_data_loading(self, tpm_saved: float = 0, mut_saved: float = 0):
        """Track memory usage after data loading with optimization details."""
        details = {
            'tpm_memory_saved_mb': tpm_saved,
            'mutation_memory_saved_mb': mut_saved,
            'total_data_optimization_mb': tpm_saved + mut_saved
        }
        self.optimization_results['data_loading'] = details
        self.tracker.checkpoint("after_data_loading", details, 
                               f"Data loaded and optimized. Total saved: {tpm_saved + mut_saved:.2f} MB")
    
    def track_preprocessing(self, genes_filtered: int):
        """Track memory usage after data preprocessing."""
        details = {'genes_filtered_to': genes_filtered}
        self.tracker.checkpoint("after_preprocessing", details, 
                               f"Data preprocessed, filtered to {genes_filtered} genes")
    
    def track_correlation_computation(self, corr_saved: float = 0):
        """Track memory usage after correlation computation."""
        details = {'correlation_memory_saved_mb': corr_saved}
        self.optimization_results['correlation'] = details
        self.tracker.checkpoint("after_correlation", details, 
                               f"Correlation computed and optimized. Saved: {corr_saved:.2f} MB")
    
    def track_weight_modification(self):
        """Track memory usage after weight modification."""
        self.tracker.checkpoint("after_weight_modification", {}, "Edge weights modified")
    
    def track_edge_pruning(self):
        """Track memory usage after edge pruning."""
        self.tracker.checkpoint("after_edge_pruning", {}, "Edges pruned")
    
    def track_graph_creation(self):
        """Track memory usage after graph creation."""
        self.tracker.checkpoint("after_graph_creation", {}, "Graph objects created")
    
    def track_community_detection(self, method: str):
        """Track memory usage after community detection."""
        details = {'community_detection_method': method}
        self.tracker.checkpoint("after_community_detection", details, 
                               f"Community detection completed using {method}")
    
    def finish_tracking(self):
        """Finish tracking and provide summary."""
        total_optimization = sum(
            result.get('total_data_optimization_mb', 0) + result.get('correlation_memory_saved_mb', 0)
            for result in self.optimization_results.values()
        )
        
        details = {
            'total_pipeline_optimization_mb': total_optimization,
            'optimization_breakdown': self.optimization_results
        }
        
        self.tracker.checkpoint("pipeline_completed", details, 
                               f"Pipeline completed. Total optimization: {total_optimization:.2f} MB")
        
        self.tracker.print_summary()
        return self.tracker.get_memory_summary()


# Export main classes
__all__ = ['MemoryTracker', 'PipelineMemoryTracker']