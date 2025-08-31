#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_performance.py
@Time    :   2025/08/31
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Performance benchmarking for bioinformatics pipeline optimizations using control experiments
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import datetime
import time
import json
import argparse
from pathlib import Path
import multiprocess as mp
from collections import defaultdict

# iCoExpNet imports
sys.path.append('../..')
from icoexpnet.analysis.GraphToolExp import GraphToolExperiment as GtExp
from icoexpnet.analysis.ExperimentSet import ExperimentSet
from icoexpnet.analysis.utilities.modcon_optimization import benchmark_modcon_methods
from icoexpnet.analysis.utilities.mevs_optimization import benchmark_mevs_methods, benchmark_imevs_methods


def worker(arg):
    """
    Standalone worker function for multiprocessing.
    
    This function needs to be at module level to be picklable for multiprocessing.
    
    Args:
        arg (tuple): Contains (object, method_name) and additional args
        
    Returns:
        object: The processed object after calling the method
    """
    obj, method_name = arg[:2]
    _ = getattr(obj, method_name)()
    return obj


def setup_performance_logging():
    """Set up logging for performance tests with timestamp."""
    # Create logs directory if it doesn't exist
    log_dir = Path("../../../results/optimization_tests/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"performance_test_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("⚡ BIOINFORMATICS OPTIMIZATION PERFORMANCE TESTS")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Test started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger, log_file


class PerformanceTester:
    """
    Performance testing suite for bioinformatics pipeline optimizations.
    
    Uses control experiments for statistical reliability and comprehensive benchmarking.
    """
    
    def __init__(self, logger, fail_fast=False):
        """Initialize performance tester."""
        self.logger = logger
        self.results = defaultdict(dict)
        self.timing_data = defaultdict(list)
        self.fail_fast = fail_fast
        
        # Define paths (same as test_optimisations.py)
        self.results_path = "../../../results/"
        self.data_base = "../../../data/"
        self.base_path = "../../../"
        self.test_cltrs_path = "results/testCtrl/"
        
        self.logger.info(f"Base path: {self.base_path}")
        self.logger.info(f"Data path: {self.data_base}")
        self.logger.info(f"Control experiments path: {self.test_cltrs_path}")
        if fail_fast:
            self.logger.info("⚡ Fail-fast mode enabled - will exit on first error")
    
    def setup_data(self):
        """Load control experiments and mutation data."""
        self.logger.info("📁 Loading test data and control experiments...")
        
        # Load mutation data
        self.mut_df = pd.read_csv(f"{self.data_base}/test_mutation_data.tsv",
                                 sep="\t", index_col="gene")
        self.logger.info(f"Loaded mutation data: {self.mut_df.shape}")
        
        # Load TF list
        tf_path = f"{self.data_base}/TF_names_v_1.01.txt"
        if os.path.exists(tf_path):
            self.tf_list = np.genfromtxt(fname=tf_path, delimiter="\t",
                                        skip_header=1, dtype="str")
            self.logger.info(f"Loaded TF list: {len(self.tf_list)} transcription factors")
        
        # Load control experiments
        folders = next(os.walk(self.base_path + self.test_cltrs_path), (None, None, []))[1]
        self.test_ctrls = {}
        
        for folder in folders:
            hCtrl_path = f"{self.test_cltrs_path}/{folder}/"
            idx = int(folder.split("tctrl_")[-1])
            self.test_ctrls[idx] = ExperimentSet(
                "tCtrl", self.base_path, hCtrl_path, self.mut_df, 
                sel_sets=None, rel_path="../", exp_type="iNet")
            self.test_ctrls[idx].export_to_gephi(save=False)
        
        self.logger.info(f"✅ Loaded {len(self.test_ctrls)} control experiment sets")
        
        # Load control experiments
        self.ctrl_exps = {}
        for key in range(1, len(self.test_ctrls) + 1, 1):
            self.logger.info(f"Loading control experiment set #{key}...")
            exps, entropy = GtExp.load_hsbm_exps(self.test_ctrls[key])
            entropy["Type"] = f"hCtrl{key}"
            self.ctrl_exps[key] = {"entropy": entropy, "exps": exps}
        
        total_exps = sum(len(ctrl_set["exps"]) for ctrl_set in self.ctrl_exps.values())
        self.logger.info(f"✅ Total experiments available for testing: {total_exps}")
    
    def _handle_error(self, error_msg, exception=None):
        """Handle errors with optional fail-fast behavior."""
        if exception:
            self.logger.error(f"{error_msg}: {exception}")
        else:
            self.logger.error(error_msg)
            
        if self.fail_fast:
            self.logger.error("💥 Fail-fast mode: Exiting on first error")
            raise SystemExit(1)
    
    def benchmark_modcon_performance(self, num_runs=5):
        """Benchmark ModCon computation performance."""
        self.logger.info("\n🚀 Benchmarking ModCon performance...")
        
        modcon_results = []
        
        for ctrl_idx, ctrl_data in self.ctrl_exps.items():
            self.logger.info(f"Testing ModCon performance for control set {ctrl_idx}...")
            
            for exp_key, exp in ctrl_data["exps"].items():
                self.logger.info(f"  Benchmarking experiment {exp_key} (TF count: {exp_key})...")
                
                try:
                    # Benchmark ModCon methods
                    start_time = time.time()
                    
                    # Use the benchmark function from modcon_optimization
                    bench_results = benchmark_modcon_methods(
                        edges_df=exp.edges_df,
                        gen_coms=exp.leiden_best.rename(columns={"Id": "Id", "Modularity Class": "max_b"})[["Id", "max_b"]].reset_index(drop=True),
                        meta_df=exp.meta_df,
                        mut_df=exp.mut_df if not exp.mut_df.empty else None,
                        modifier=exp.type.split("_")[0],
                        exp_type=exp.type,
                        num_runs=num_runs,
                        verbose=False
                    )
                    
                    benchmark_time = time.time() - start_time
                    
                    # Store results
                    result = {
                        'ctrl_set': ctrl_idx,
                        'exp_key': exp_key,
                        'tf_count': exp_key,
                        'communities': bench_results['communities_processed'],
                        'total_genes': bench_results['total_genes'],
                        'avg_original_time': bench_results['estimated_original_time'],
                        'avg_optimized_time': bench_results['avg_optimized_time'],
                        'speedup': bench_results['speedup_estimate'],
                        'benchmark_duration': benchmark_time
                    }
                    
                    modcon_results.append(result)
                    self.results[f'modcon_ctrl_{ctrl_idx}_{exp_key}'] = result
                    
                    self.logger.info(f"  ✅ ModCon speedup: {result['speedup']:.1f}x "
                                   f"({result['avg_original_time']:.3f}s → {result['avg_optimized_time']:.3f}s)")
                
                except Exception as e:
                    error_msg = f"ModCon benchmark failed for {exp_key}"
                    self._handle_error(f"  ❌ {error_msg}", e)
                    if not self.fail_fast:
                        continue
        
        # Save ModCon results
        modcon_df = pd.DataFrame(modcon_results)
        if not modcon_df.empty:
            results_dir = Path("../../../results/optimization_tests/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            modcon_file = results_dir / f"modcon_performance_{timestamp}.csv"
            modcon_df.to_csv(modcon_file, index=False)
            
            self.logger.info(f"📊 ModCon results summary:")
            self.logger.info(f"  Average speedup: {modcon_df['speedup'].mean():.1f}x")
            self.logger.info(f"  Median speedup: {modcon_df['speedup'].median():.1f}x")
            self.logger.info(f"  Results saved to: {modcon_file}")
        
        return modcon_df
    
    def benchmark_mevs_performance(self, num_runs=3):
        """Benchmark MEVs computation performance."""
        self.logger.info("\n🧬 Benchmarking MEVs performance...")
        
        mevs_results = []
        
        # First, compute ModCon for all experiments (required for MEVs)
        self.logger.info("Computing ModCon for MEVs benchmarking...")
        for ctrl_idx in self.ctrl_exps.keys():
            try:
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.map(worker, 
                                      ((exp, "get_ModCon") for exp in self.ctrl_exps[ctrl_idx]["exps"].values()))
                    self.ctrl_exps[ctrl_idx]["exps"] = {exp.extract_tf_number(exp.name): exp for exp in results}
            except Exception as e:
                error_msg = f"ModCon computation failed for control set {ctrl_idx}"
                self._handle_error(error_msg, e)
                if not self.fail_fast:
                    continue
        
        # Now benchmark MEVs
        for ctrl_idx, ctrl_data in self.ctrl_exps.items():
            self.logger.info(f"Testing MEVs performance for control set {ctrl_idx}...")
            
            for exp_key, exp in ctrl_data["exps"].items():
                self.logger.info(f"  Benchmarking MEVs for experiment {exp_key}...")
                
                try:
                    sort_col = f"ModCon_{exp.type}_gt"
                    
                    # Benchmark MEVs methods
                    start_time = time.time()
                    
                    bench_results = benchmark_mevs_methods(
                        tpms=exp.tpm_df,
                        modCon=exp.gt_modCon,
                        sort_col=sort_col,
                        num_genes=100,
                        num_runs=num_runs,
                        verbose=False
                    )
                    
                    benchmark_time = time.time() - start_time
                    
                    # Store results
                    result = {
                        'ctrl_set': ctrl_idx,
                        'exp_key': exp_key,
                        'tf_count': exp_key,
                        'communities': bench_results['communities_processed'],
                        'samples': bench_results['total_samples'],
                        'avg_original_time': bench_results['estimated_original_time'],
                        'avg_optimized_time': bench_results['avg_optimized_time'],
                        'speedup': bench_results['speedup_estimate'],
                        'time_saved_percentage': ((bench_results['estimated_original_time'] - bench_results['avg_optimized_time']) / bench_results['estimated_original_time']) * 100,
                        'benchmark_duration': benchmark_time
                    }
                    
                    mevs_results.append(result)
                    self.results[f'mevs_ctrl_{ctrl_idx}_{exp_key}'] = result
                    
                    self.logger.info(f"  ✅ MEVs speedup: {result['speedup']:.1f}x "
                                   f"({result['avg_original_time']:.3f}s → {result['avg_optimized_time']:.3f}s)")
                
                except Exception as e:
                    error_msg = f"MEVs benchmark failed for {exp_key}"
                    self._handle_error(f"  ❌ {error_msg}", e)
                    if not self.fail_fast:
                        continue
        
        # Save MEVs results
        mevs_df = pd.DataFrame(mevs_results)
        if not mevs_df.empty:
            results_dir = Path("../../../results/optimization_tests/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mevs_file = results_dir / f"mevs_performance_{timestamp}.csv"
            mevs_df.to_csv(mevs_file, index=False)
            
            self.logger.info(f"📊 MEVs results summary:")
            self.logger.info(f"  Average speedup: {mevs_df['speedup'].mean():.1f}x")
            self.logger.info(f"  Median speedup: {mevs_df['speedup'].median():.1f}x")
            self.logger.info(f"  Average time saved: {mevs_df['time_saved_percentage'].mean():.1f}%")
            self.logger.info(f"  Results saved to: {mevs_file}")
        
        return mevs_df
    
    def benchmark_imevs_performance(self, num_runs=3):
        """Benchmark integrated MEVs computation performance."""
        self.logger.info("\n🔬 Benchmarking integrated MEVs performance...")
        
        imevs_results = []
        
        # Use a subset of experiments for integrated MEVs testing (it's more computationally intensive)
        test_ctrl_keys = list(self.ctrl_exps.keys())[:]
        
        # First, compute ModCon for the selected control sets (required for integrated MEVs)
        self.logger.info("Computing ModCon for integrated MEVs benchmarking...")
        for ctrl_idx in test_ctrl_keys:
            try:
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.map(worker, 
                                      ((exp, "get_ModCon") for exp in self.ctrl_exps[ctrl_idx]["exps"].values()))
                    self.ctrl_exps[ctrl_idx]["exps"] = {exp.extract_tf_number(exp.name): exp for exp in results}
            except Exception as e:
                error_msg = f"ModCon computation failed for control set {ctrl_idx}"
                self._handle_error(error_msg, e)
                if not self.fail_fast:
                    continue
        
        for ctrl_idx in test_ctrl_keys:
            ctrl_data = self.ctrl_exps[ctrl_idx]
            self.logger.info(f"Testing integrated MEVs performance for control set {ctrl_idx}...")
            
            # Test on a subset of experiments per control set
            exp_keys = list(ctrl_data["exps"].keys())[:]  # Test all experiments

            for exp_key in exp_keys:
                exp = ctrl_data["exps"][exp_key]
                self.logger.info(f"  Benchmarking integrated MEVs for experiment {exp_key}...")
                
                try:
                    sort_col = f"ModCon_{exp.type}_gt"
                    
                    # Use mutation data as tumor data for testing
                    tumor_tpms = self.mut_df.T
                    
                    # Benchmark integrated MEVs methods
                    start_time = time.time()
                    
                    bench_results = benchmark_imevs_methods(
                        h_tpms=exp.tpm_df,
                        tum_tpms=tumor_tpms,
                        modCon=exp.gt_modCon,
                        sort_col=sort_col,
                        num_genes=50,
                        num_runs=num_runs,
                        verbose=False,
                        mut_df=self.mut_df,
                        mut_offset=1.0
                    )
                    
                    benchmark_time = time.time() - start_time
                    
                    # Store results
                    result = {
                        'ctrl_set': ctrl_idx,
                        'exp_key': exp_key,
                        'tf_count': exp_key,
                        'communities': bench_results['communities_processed'],
                        'tumor_samples': bench_results['tumor_samples'],
                        'healthy_samples': bench_results['healthy_samples'],
                        'avg_original_time': bench_results['estimated_original_time'],
                        'avg_optimized_time': bench_results['avg_optimized_time'],
                        'speedup': bench_results['speedup_estimate'],
                        'time_saved_percentage': ((bench_results['estimated_original_time'] - bench_results['avg_optimized_time']) / bench_results['estimated_original_time']) * 100,
                        'benchmark_duration': benchmark_time
                    }
                    
                    imevs_results.append(result)
                    self.results[f'imevs_ctrl_{ctrl_idx}_{exp_key}'] = result
                    
                    self.logger.info(f"  ✅ Integrated MEVs speedup: {result['speedup']:.1f}x "
                                   f"({result['avg_original_time']:.3f}s → {result['avg_optimized_time']:.3f}s)")
                
                except Exception as e:
                    error_msg = f"Integrated MEVs benchmark failed for {exp_key}"
                    self._handle_error(f"  ❌ {error_msg}", e)
                    if not self.fail_fast:
                        continue
        
        # Save integrated MEVs results
        imevs_df = pd.DataFrame(imevs_results)
        if not imevs_df.empty:
            results_dir = Path("../../../results/optimization_tests/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            imevs_file = results_dir / f"imevs_performance_{timestamp}.csv"
            imevs_df.to_csv(imevs_file, index=False)
            
            self.logger.info(f"📊 Integrated MEVs results summary:")
            self.logger.info(f"  Average speedup: {imevs_df['speedup'].mean():.1f}x")
            self.logger.info(f"  Median speedup: {imevs_df['speedup'].median():.1f}x")
            self.logger.info(f"  Average time saved: {imevs_df['time_saved_percentage'].mean():.1f}%")
            self.logger.info(f"  Results saved to: {imevs_file}")
        
        return imevs_df
    
    def benchmark_pipeline_performance(self, num_runs=3):
        """Benchmark complete optimized pipeline performance."""
        self.logger.info("\n🔗 Benchmarking complete optimized pipeline performance...")
        
        pipeline_results = []
        
        # Test on a subset for complete pipeline (most comprehensive test)
        test_ctrl_key = list(self.ctrl_exps.keys())[0]  # Use first control set
        ctrl_data = self.ctrl_exps[test_ctrl_key]
        exp_keys = list(ctrl_data["exps"].keys())[:]  # Test all experiments

        self.logger.info(f"Testing pipeline performance on control set {test_ctrl_key}, experiments: {exp_keys}")
        
        for exp_key in exp_keys:
            exp = ctrl_data["exps"][exp_key]
            self.logger.info(f"  Benchmarking complete pipeline for experiment {exp_key}...")
            
            try:
                # Benchmark complete optimized pipeline
                start_time = time.time()
                
                pipeline_bench_results = exp.benchmark_optimized_pipeline(
                    all_tpms=exp.tpm_df,
                    num_genes=50,
                    is_imev=False,
                    num_runs=num_runs,
                    verbose=False
                )
                
                benchmark_time = time.time() - start_time
                
                # Store results
                result = {
                    'ctrl_set': test_ctrl_key,
                    'exp_key': exp_key,
                    'tf_count': exp_key,
                    'communities': pipeline_bench_results['communities_processed'],
                    'samples': pipeline_bench_results['samples_analyzed'],
                    'avg_original_time': pipeline_bench_results['avg_original_time'],
                    'avg_optimized_time': pipeline_bench_results['avg_optimized_time'],
                    'speedup': pipeline_bench_results['speedup'],
                    'time_saved_percentage': pipeline_bench_results['time_saved_percentage'],
                    'benchmark_duration': benchmark_time,
                    'pipeline_type': pipeline_bench_results['pipeline_type']
                }
                
                pipeline_results.append(result)
                self.results[f'pipeline_ctrl_{test_ctrl_key}_{exp_key}'] = result
                
                self.logger.info(f"  ✅ Complete pipeline speedup: {result['speedup']:.1f}x "
                               f"({result['avg_original_time']:.3f}s → {result['avg_optimized_time']:.3f}s)")
            
            except Exception as e:
                error_msg = f"Complete pipeline benchmark failed for {exp_key}"
                self._handle_error(f"  ❌ {error_msg}", e)
                if not self.fail_fast:
                    continue
        
        # Save pipeline results
        pipeline_df = pd.DataFrame(pipeline_results)
        if not pipeline_df.empty:
            results_dir = Path("../../../results/optimization_tests/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_file = results_dir / f"pipeline_performance_{timestamp}.csv"
            pipeline_df.to_csv(pipeline_file, index=False)
            
            self.logger.info(f"📊 Complete pipeline results summary:")
            self.logger.info(f"  Average speedup: {pipeline_df['speedup'].mean():.1f}x")
            self.logger.info(f"  Median speedup: {pipeline_df['speedup'].median():.1f}x")
            self.logger.info(f"  Results saved to: {pipeline_file}")
        
        return pipeline_df
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("🧹 Cleaned up resources")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for bioinformatics pipeline optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_performance.py                    # Run all performance benchmarks
  python test_performance.py --fail-fast       # Exit on first error
  python test_performance.py --test modcon     # Run only ModCon benchmarks
  python test_performance.py --test mevs       # Run only MEVs benchmarks
        """
    )
    
    parser.add_argument(
        '--fail-fast', 
        action='store_true', 
        help='Exit on first error instead of continuing with other tests'
    )
    
    parser.add_argument(
        '--test', 
        type=str,
        choices=['modcon', 'mevs', 'imevs', 'pipeline'],
        help='Run specific test category only'
    )
    
    parser.add_argument(
        '--runs', 
        type=int, 
        default=3,
        help='Number of benchmark runs per test (default: 3)'
    )
    
    return parser.parse_args()


def run_performance_tests(args=None):
    """
    Run complete performance benchmarking suite.
    """
    if args is None:
        args = parse_arguments()
        
    logger, log_file = setup_performance_logging()
    
    start_time = datetime.datetime.now()
    
    try:
        # Initialize performance tester
        tester = PerformanceTester(logger, fail_fast=args.fail_fast)
        
        # Setup test data
        tester.setup_data()
        
        # Run performance benchmarks
        logger.info("\n🚀 Starting comprehensive performance benchmarking...")
        
        results = {}
        
        # Run specific test if requested
        if args.test:
            logger.info(f"Running specific test category: {args.test}")
            
            if args.test == 'modcon':
                results['modcon'] = tester.benchmark_modcon_performance(num_runs=args.runs)
            elif args.test == 'mevs':
                results['mevs'] = tester.benchmark_mevs_performance(num_runs=args.runs)
            elif args.test == 'imevs':
                results['imevs'] = tester.benchmark_imevs_performance(num_runs=args.runs)
            elif args.test == 'pipeline':
                results['pipeline'] = tester.benchmark_pipeline_performance(num_runs=args.runs)
        else:
            # Run all tests
            # 1. ModCon performance
            results['modcon'] = tester.benchmark_modcon_performance(num_runs=max(5, args.runs))
            
            # 2. MEVs performance
            results['mevs'] = tester.benchmark_mevs_performance(num_runs=args.runs)
            
            # 3. Integrated MEVs performance
            results['imevs'] = tester.benchmark_imevs_performance(num_runs=args.runs)
            
            # 4. Complete pipeline performance
            results['pipeline'] = tester.benchmark_pipeline_performance(num_runs=args.runs)
        
        # Generate overall summary
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total test duration: {duration}")
        
        for test_name, df in results.items():
            if df is not None and not df.empty:
                logger.info(f"{test_name.title()} average speedup: {df['speedup'].mean():.1f}x (range: {df['speedup'].min():.1f}x - {df['speedup'].max():.1f}x)")
        
        # Save comprehensive results
        results_dir = Path("../../../results/optimization_tests/results")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results as JSON
        all_results = {
            'test_info': {
                'timestamp': timestamp,
                'duration_seconds': duration.total_seconds(),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'args': vars(args)
            },
            'results': dict(tester.results)
        }
        
        results_file = results_dir / f"all_performance_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"\n📝 Complete results saved to: {results_file}")
        logger.info(f"📝 Full log saved to: {log_file}")
        logger.info(f"Test completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean up
        tester.cleanup()
        
        return True
    
    except Exception as e:
        logger.error(f"💥 Unexpected error during performance testing: {e}")
        if args and args.fail_fast:
            raise
        return False


if __name__ == "__main__":
    args = parse_arguments()
    success = run_performance_tests(args)
    sys.exit(0 if success else 1)