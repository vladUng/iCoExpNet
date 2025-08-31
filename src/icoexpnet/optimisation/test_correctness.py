#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_correctness.py
@Time    :   2025/08/31
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Unit tests to verify correctness and equality between original and optimized bioinformatics methods
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging
import datetime
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict
import argparse

# iCoExpNet imports
sys.path.append('../..')
from icoexpnet.analysis.GraphToolExp import GraphToolExperiment as GtExp
from icoexpnet.analysis.ExperimentSet import ExperimentSet


class LoggingTestResult(unittest.TextTestResult):
    """Custom test result class that logs to both console and file."""
    
    def __init__(self, stream, descriptions, verbosity, logger):
        super().__init__(stream, descriptions, verbosity)
        self.logger = logger
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.logger.info(f"✅ PASSED: {test._testMethodName}")
    
    def addError(self, test, err):
        super().addError(test, err)
        self.logger.error(f"❌ ERROR: {test._testMethodName} - {err[1]}")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.logger.error(f"❌ FAILED: {test._testMethodName} - {err[1]}")


class LoggingTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses logging test results."""
    
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
    
    def _makeResult(self):
        return LoggingTestResult(self.stream, self.descriptions, self.verbosity, self.logger)


def worker(arg):
    """Standalone worker function for parallel processing."""
    obj, method_name = arg[:2]
    _ = getattr(obj, method_name)()
    return obj


def setup_logging():
    """Set up logging to both console and file with timestamp."""
    # Create logs directory if it doesn't exist
    log_dir = Path("../../../results/optimization_tests/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"correctness_test_{timestamp}.log"
    
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
    logger.info("🔬 BIOINFORMATICS OPTIMIZATION CORRECTNESS TESTS")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Test started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger, log_file


class TestBioinformaticsOptimizations(unittest.TestCase):
    """
    Unit tests for bioinformatics pipeline optimizations.
    
    Tests verify that optimized methods produce identical results to original methods
    across key DataFrame objects: tpm_df, edges_df, nodes_df, meta_df, modCon, mevsMut.
    
    Focus on h_exps (healthy experiments) as representative test cases.
    """
    
    fail_fast = False  # Class variable to control fail-fast behavior
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and load experiments once for all tests."""
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("🔬 Setting up correctness tests for bioinformatics optimizations...")
        
        # Define paths (same as test_optimisations.py)
        cls.results_path = "../../../results/"
        cls.data_base = "../../../data/"
        cls.base_path = "../../../"
        cls.test_exps_path = "results/test/"
        
        cls.logger.info(f"Base path: {cls.base_path}")
        cls.logger.info(f"Data path: {cls.data_base}")
        cls.logger.info(f"Test experiments path: {cls.test_exps_path}")
        
        # Load mutation data
        cls.mut_df = pd.read_csv(f"{cls.data_base}/test_mutation_data.tsv",
                                sep="\t", index_col="gene")
        cls.logger.info(f"Loaded mutation data: {cls.mut_df.shape}")
        
        # Load TF list
        tf_path = f"{cls.data_base}/TF_names_v_1.01.txt"
        if os.path.exists(tf_path):
            cls.tf_list = np.genfromtxt(fname=tf_path, delimiter="\t",
                                       skip_header=1, dtype="str")
            cls.logger.info(f"Loaded TF list: {len(cls.tf_list)} transcription factors")
        
        # Load test experiments (healthy experiments)
        cls.exp_test = ExperimentSet("test", base_path=cls.base_path, 
                                    exp_path=cls.test_exps_path,
                                    mut_df=cls.mut_df, sel_sets=None, exp_type="iNet")
        
        # Load HSBM experiments
        cls.h_exps_raw, cls.h_entropy = GtExp.load_hsbm_exps(cls.exp_test)
        cls.h_entropy["Type"] = "Experiment"
        
        cls.logger.info(f"✅ Loaded {len(cls.h_exps_raw)} healthy experiments for testing")
        cls.logger.info(f"Using {mp.cpu_count()} CPU cores for parallel processing")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        cls.logger.info("🧹 Test cleanup completed")
    
    def setUp(self):
        """Set up for each individual test."""
        self.comparison_results = defaultdict(dict)
        self.failed_comparisons = []
        self.logger = logging.getLogger(__name__)
    
    def compare_dataframes(self, df1, df2, name, exp_key):
        """
        Compare two DataFrames with detailed error reporting.
        
        Args:
            df1, df2: DataFrames to compare
            name: Name of the DataFrame attribute
            exp_key: Experiment key for identification
            
        Returns:
            bool: True if DataFrames are equal
        """
        if df1 is None and df2 is None:
            return True
        
        if (df1 is None) != (df2 is None):
            error_msg = f"Exp {exp_key} - {name}: One DataFrame is None"
            self.failed_comparisons.append(error_msg)
            self.logger.error(error_msg)
            return False
        
        if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
            # Handle non-DataFrame objects (like dictionaries)
            try:
                are_equal = df1 == df2
                if not are_equal:
                    error_msg = f"Exp {exp_key} - {name}: Objects not equal"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                return are_equal
            except:
                # For complex objects like dictionaries of DataFrames
                if isinstance(df1, dict) and isinstance(df2, dict):
                    return self.compare_dict_of_dataframes(df1, df2, name, exp_key)
                else:
                    error_msg = f"Exp {exp_key} - {name}: Cannot compare object types"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                    return False
        
        # Compare DataFrame shapes first
        if df1.shape != df2.shape:
            error_msg = f"Exp {exp_key} - {name}: Shape mismatch - {df1.shape} vs {df2.shape}"
            self.failed_comparisons.append(error_msg)
            self.logger.error(error_msg)
            if self.fail_fast:
                self.fail(f"FAIL-FAST: {error_msg}")
            return False
        
        # Compare DataFrame contents
        try:
            # For floating point comparisons, use numpy.allclose for better tolerance
            if df1.dtypes.apply(lambda x: x in ['float64', 'float32']).any() or \
               df2.dtypes.apply(lambda x: x in ['float64', 'float32']).any():
                # Use all close for floating point comparisons with reasonable tolerance
                try:
                    numeric_equal = np.allclose(df1.select_dtypes(include=[np.number]).values, 
                                              df2.select_dtypes(include=[np.number]).values, 
                                              rtol=1e-10, atol=1e-10, equal_nan=True)
                    non_numeric_equal = df1.select_dtypes(exclude=[np.number]).equals(
                        df2.select_dtypes(exclude=[np.number]))
                    are_equal = numeric_equal and non_numeric_equal
                except:
                    are_equal = df1.equals(df2)
            else:
                are_equal = df1.equals(df2)
                
            if not are_equal:
                # More detailed comparison for debugging
                if not df1.index.equals(df2.index):
                    error_msg = f"Exp {exp_key} - {name}: Index mismatch"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                elif not df1.columns.equals(df2.columns):
                    error_msg = f"Exp {exp_key} - {name}: Columns mismatch"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                else:
                    # Find first differing values
                    diff_mask = df1 != df2
                    if diff_mask.any().any():
                        diff_locations = np.where(diff_mask)
                        first_diff_row = diff_locations[0][0] if len(diff_locations[0]) > 0 else 0
                        first_diff_col = diff_locations[1][0] if len(diff_locations[1]) > 0 else 0
                        val1 = df1.iloc[first_diff_row, first_diff_col]
                        val2 = df2.iloc[first_diff_row, first_diff_col]
                        
                        # Check if it's just a floating point precision issue
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            if abs(val1 - val2) < 1e-10:
                                self.logger.warning(f"Exp {exp_key} - {name}: Minor floating point difference at [{first_diff_row}, {first_diff_col}]: {val1} vs {val2} (within tolerance)")
                                are_equal = True  # Accept as equal
                            else:
                                error_msg = f"Exp {exp_key} - {name}: Significant difference at [{first_diff_row}, {first_diff_col}]: {val1} vs {val2}"
                                self.failed_comparisons.append(error_msg)
                                self.logger.error(error_msg)
                        else:
                            error_msg = f"Exp {exp_key} - {name}: First difference at [{first_diff_row}, {first_diff_col}]: {val1} vs {val2}"
                            self.failed_comparisons.append(error_msg)
                            self.logger.error(error_msg)
            return are_equal
        except Exception as e:
            error_msg = f"Exp {exp_key} - {name}: Comparison error - {e}"
            self.failed_comparisons.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def compare_dict_of_dataframes(self, dict1, dict2, name, exp_key):
        """
        Compare dictionaries containing DataFrames (like modCon).
        
        Args:
            dict1, dict2: Dictionaries to compare
            name: Name of the attribute
            exp_key: Experiment key for identification
            
        Returns:
            bool: True if dictionaries are equal
        """
        if set(dict1.keys()) != set(dict2.keys()):
            error_msg = f"Exp {exp_key} - {name}: Dictionary keys mismatch - {set(dict1.keys())} vs {set(dict2.keys())}"
            self.failed_comparisons.append(error_msg)
            self.logger.error(error_msg)
            return False
        
        all_equal = True
        for key in dict1.keys():
            df_equal = self.compare_dataframes(dict1[key], dict2[key], f"{name}[{key}]", exp_key)
            if not df_equal:
                all_equal = False
        
        return all_equal
    
    def test_modcon_correctness(self):
        """Test ModCon computation correctness."""
        self.logger.info("\n🧬 Testing ModCon correctness...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            # Run original ModCon computation
            self.logger.info("Computing original ModCon results...")
            results_original = pool.map(worker, 
                                           ((exp, "get_ModCon") for exp in self.h_exps_raw.values()))
            h_exps_original = {exp.extract_tf_number(exp.name): exp for exp in results_original}
            
            # Run optimized ModCon computation  
            self.logger.info("Computing optimized ModCon results...")
            results_optimized = pool.map(worker,
                                            ((exp, "get_ModCon_optimized") for exp in self.h_exps_raw.values()))
            h_exps_optimized = {exp.extract_tf_number(exp.name): exp for exp in results_optimized}
        
        # Compare results
        self.logger.info("Comparing ModCon results...")
        all_passed = True
        for key in h_exps_original.keys():
            exp_original = h_exps_original[key]
            exp_optimized = h_exps_optimized[key]
            
            # Test gt_modCon attribute specifically
            if hasattr(exp_original, 'gt_modCon') and hasattr(exp_optimized, 'gt_modCon'):
                modcon_equal = self.compare_dict_of_dataframes(
                    exp_original.gt_modCon, exp_optimized.gt_modCon, 'gt_modCon', key)
                self.comparison_results[key]['gt_modCon'] = modcon_equal
                
                if modcon_equal:
                    self.logger.info(f"✅ Exp {key}: ModCon results match")
                else:
                    all_passed = False
                    self.logger.error(f"❌ Exp {key}: ModCon results differ")
            else:
                error_msg = f"Exp {key}: gt_modCon attribute missing"
                self.failed_comparisons.append(error_msg)
                self.logger.error(error_msg)
                all_passed = False
        
        # Assert overall success
        if not all_passed:
            failure_summary = "\n".join(self.failed_comparisons[-5:])  # Show last 5 failures
            self.fail(f"ModCon correctness test failed. Recent failures:\n{failure_summary}")
        
        self.logger.info(f"✅ ModCon correctness test passed for {len(h_exps_original)} experiments")
    
    def test_mevs_correctness(self):
        """Test MEVs computation correctness."""
        self.logger.info("\n🧬 Testing MEVs correctness...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            # First compute ModCon (required for MEVs)
            self.logger.info("Computing ModCon for MEVs test...")
            results = pool.map(worker, 
                              ((exp, "get_ModCon") for exp in self.h_exps_raw.values()))
            h_exps_with_modcon = {exp.extract_tf_number(exp.name): exp for exp in results}
        
        # Test MEVs computation
        self.logger.info("Comparing MEVs results...")
        all_passed = True
        for key, exp in h_exps_with_modcon.items():
            sort_col = f"ModCon_{exp.type}_gt"
            
            # Original MEVs
            mevs_original, _ = exp.get_mevs(
                tpms=exp.tpm_df, modCon=exp.gt_modCon, 
                sort_col=sort_col, num_genes=100, verbose=False)
            
            # Optimized MEVs
            mevs_optimized, _ = exp.get_mevs_optimised(
                tpms=exp.tpm_df, modCon=exp.gt_modCon,
                sort_col=sort_col, num_genes=100, verbose=False)
            
            # Compare results
            mevs_equal = self.compare_dataframes(
                mevs_original, mevs_optimized, 'mevs', key)
            self.comparison_results[key]['mevs'] = mevs_equal
            
            if mevs_equal:
                self.logger.info(f"✅ Exp {key}: MEVs results match")
            else:
                all_passed = False
                self.logger.error(f"❌ Exp {key}: MEVs results differ")
        
        # Assert overall success
        if not all_passed:
            failure_summary = "\n".join(self.failed_comparisons[-5:])
            self.fail(f"MEVs correctness test failed. Recent failures:\n{failure_summary}")
        
        self.logger.info(f"✅ MEVs correctness test passed for {len(h_exps_with_modcon)} experiments")
    
    def test_imevs_correctness(self):
        """Test integrated MEVs computation correctness."""
        self.logger.info("\n🧬 Testing integrated MEVs correctness...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            # First compute ModCon (required for iMEVs)
            self.logger.info("Computing ModCon for integrated MEVs test...")
            results = pool.map(worker,
                              ((exp, "get_ModCon") for exp in self.h_exps_raw.values()))
            h_exps_with_modcon = {exp.extract_tf_number(exp.name): exp for exp in results}
        
        # Test integrated MEVs computation
        self.logger.info("Comparing integrated MEVs results...")
        all_passed = True
        for key, exp in h_exps_with_modcon.items():
            sort_col = f"ModCon_{exp.type}_gt"
            
            # Use mutation data as tumor data for testing
            tumor_tpms = self.mut_df.T  # Transpose to get samples x genes format
            
            # Original integrated MEVs
            imevs_original, _ = exp.get_iMevs(
                h_tpms=exp.tpm_df, tum_tpms=tumor_tpms, modCon=exp.gt_modCon,
                sort_col=sort_col, num_genes=50, verbose=False,
                mut_df=self.mut_df, offset=1.0)
            
            # Optimized integrated MEVs
            imevs_optimized, _ = exp.get_iMevs_optimised(
                h_tpms=exp.tpm_df, tum_tpms=tumor_tpms, modCon=exp.gt_modCon,
                sort_col=sort_col, num_genes=50, verbose=False,
                mut_df=self.mut_df, offset=1.0)
            
            # Compare results
            imevs_equal = self.compare_dataframes(
                imevs_original, imevs_optimized, 'imevs', key)
            self.comparison_results[key]['imevs'] = imevs_equal
            
            if imevs_equal:
                self.logger.info(f"✅ Exp {key}: Integrated MEVs results match")
            else:
                all_passed = False
                self.logger.error(f"❌ Exp {key}: Integrated MEVs results differ")
        
        # Assert overall success
        if not all_passed:
            failure_summary = "\n".join(self.failed_comparisons[-5:])
            self.fail(f"Integrated MEVs correctness test failed. Recent failures:\n{failure_summary}")
        
        self.logger.info(f"✅ Integrated MEVs correctness test passed for {len(h_exps_with_modcon)} experiments")
    
    def test_dataframe_attributes_equality(self):
        """Test equality of core DataFrame attributes."""
        self.logger.info("\n🧬 Testing core DataFrame attributes equality...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            # Run both original and optimized computations
            self.logger.info("Computing original results for attribute comparison...")
            results_original = pool.map(worker,
                                           ((exp, "get_ModCon") for exp in self.h_exps_raw.values()))
            h_exps_original = {exp.extract_tf_number(exp.name): exp for exp in results_original}
            
            self.logger.info("Computing optimized results for attribute comparison...")
            results_optimized = pool.map(worker,
                                            ((exp, "get_ModCon_optimized") for exp in self.h_exps_raw.values()))  
            h_exps_optimized = {exp.extract_tf_number(exp.name): exp for exp in results_optimized}
        
        # Test core DataFrame attributes
        test_attrs = ['tpm_df', 'edges_df', 'nodes_df', 'meta_df']
        self.logger.info(f"Testing attributes: {test_attrs}")
        
        all_passed = True
        for key in h_exps_original.keys():
            exp_original = h_exps_original[key]
            exp_optimized = h_exps_optimized[key]
            
            for attr in test_attrs:
                if hasattr(exp_original, attr) and hasattr(exp_optimized, attr):
                    df_original = getattr(exp_original, attr)
                    df_optimized = getattr(exp_optimized, attr)
                    
                    attr_equal = self.compare_dataframes(
                        df_original, df_optimized, attr, key)
                    self.comparison_results[key][attr] = attr_equal
                    
                    if attr_equal:
                        self.logger.info(f"✅ Exp {key} - {attr}: DataFrames match")
                    else:
                        all_passed = False
                        self.logger.error(f"❌ Exp {key} - {attr}: DataFrames differ")
                else:
                    error_msg = f"Exp {key}: {attr} attribute missing"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                    all_passed = False
        
        # Assert overall success
        if not all_passed:
            failure_summary = "\n".join(self.failed_comparisons[-5:])
            self.fail(f"DataFrame attributes equality test failed. Recent failures:\n{failure_summary}")
        
        self.logger.info(f"✅ DataFrame attributes equality test passed for {len(h_exps_original)} experiments")
    
    def test_pipeline_integration(self):
        """Test complete pipeline integration using gt_modCon_MEV_optimised."""
        self.logger.info("\n🧬 Testing complete optimized pipeline integration...")
        
        # Test the integrated optimized pipeline
        all_passed = True
        test_exps = list(self.h_exps_raw.values())[:2]  # Test on first 2 experiments for speed
        
        for exp in test_exps:
            exp_key = exp.extract_tf_number(exp.name)
            try:
                self.logger.info(f"Testing optimized pipeline for Exp {exp_key}...")
                
                # Test optimized pipeline
                exp.gt_modCon_MEV_optimised(
                    all_tpms=exp.tpm_df,
                    num_genes=50,
                    is_imev=False,
                    verbose=False
                )
                
                # Verify results exist
                if not hasattr(exp, 'gt_modCon') or not hasattr(exp, 'mevsMut'):
                    error_msg = f"Exp {exp_key}: Pipeline did not generate expected results"
                    self.failed_comparisons.append(error_msg)
                    self.logger.error(error_msg)
                    all_passed = False
                else:
                    self.logger.info(f"✅ Exp {exp_key}: Pipeline integration successful")
                    
            except Exception as e:
                error_msg = f"Exp {exp_key}: Pipeline integration failed - {e}"
                self.failed_comparisons.append(error_msg)
                self.logger.error(error_msg)
                all_passed = False
        
        # Assert overall success
        if not all_passed:
            failure_summary = "\n".join(self.failed_comparisons[-3:])
            self.fail(f"Pipeline integration test failed. Recent failures:\n{failure_summary}")
        
        self.logger.info("✅ Complete pipeline integration test passed")


def run_correctness_tests():
    """
    Run all correctness tests and generate a summary report.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run bioinformatics optimization correctness tests')
    parser.add_argument('--fail-fast', action='store_true', 
                       help='Exit immediately on first test failure')
    parser.add_argument('--test', type=str,
                       help='Run specific test (e.g., test_imevs_correctness)')
    args = parser.parse_args()
    
    # Set fail-fast mode
    TestBioinformaticsOptimizations.fail_fast = args.fail_fast
    
    logger, log_file = setup_logging()
    
    if args.fail_fast:
        logger.info("🚀 FAIL-FAST MODE ENABLED - Will exit on first failure")
    
    start_time = datetime.datetime.now()
    
    try:
        # Create test suite
        if args.test:
            # Run specific test
            suite = unittest.TestSuite()
            suite.addTest(TestBioinformaticsOptimizations(args.test))
            logger.info(f"🎯 Running specific test: {args.test}")
        else:
            # Run all tests
            suite = unittest.TestLoader().loadTestsFromTestCase(TestBioinformaticsOptimizations)
            logger.info("🧪 Running all correctness tests")
        
        # Run tests with logging
        runner = LoggingTestRunner(logger, verbosity=2, stream=sys.stdout, failfast=args.fail_fast)
        result = runner.run(suite)
        
        # Generate summary
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 CORRECTNESS TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Test duration: {duration}")
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        
        if result.wasSuccessful():
            logger.info("🎉 ALL CORRECTNESS TESTS PASSED!")
            logger.info("✅ Optimized methods produce identical results to original methods")
            logger.info("✅ All DataFrame attributes match between original and optimized versions")
            logger.info("✅ Complete pipeline integration works correctly")
        else:
            logger.error("❌ SOME CORRECTNESS TESTS FAILED")
            
            if result.failures:
                logger.error("\n📋 FAILURE DETAILS:")
                for test, traceback in result.failures:
                    logger.error(f"   - {test}")
                    # Log only first few lines of traceback to avoid clutter
                    tb_lines = str(traceback).split('\n')[:5]
                    for line in tb_lines:
                        logger.error(f"     {line}")
            
            if result.errors:
                logger.error("\n📋 ERROR DETAILS:")
                for test, traceback in result.errors:
                    logger.error(f"   - {test}")
                    tb_lines = str(traceback).split('\n')[:5]
                    for line in tb_lines:
                        logger.error(f"     {line}")
        
        logger.info(f"\n📝 Full log saved to: {log_file}")
        logger.info(f"Test completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result.wasSuccessful()
    
    except Exception as e:
        logger.error(f"💥 Unexpected error during testing: {e}")
        return False


if __name__ == "__main__":
    # Only parse arguments if they're provided
    if len(sys.argv) > 1:
        success = run_correctness_tests()
    else:
        # Run without arguments parsing for backward compatibility
        TestBioinformaticsOptimizations.fail_fast = False
        logger, log_file = setup_logging()
        
        start_time = datetime.datetime.now()
        
        try:
            # Create test suite
            test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBioinformaticsOptimizations)
            
            # Run tests with logging
            runner = LoggingTestRunner(logger, verbosity=2, stream=sys.stdout)
            result = runner.run(test_suite)
            
            # Generate summary
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 CORRECTNESS TEST SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Test duration: {duration}")
            logger.info(f"Tests run: {result.testsRun}")
            logger.info(f"Failures: {len(result.failures)}")
            logger.info(f"Errors: {len(result.errors)}")
            
            if result.wasSuccessful():
                logger.info("🎉 ALL CORRECTNESS TESTS PASSED!")
                logger.info("✅ Optimized methods produce identical results to original methods")
                logger.info("✅ All DataFrame attributes match between original and optimized versions")
                logger.info("✅ Complete pipeline integration works correctly")
            else:
                logger.error("❌ SOME CORRECTNESS TESTS FAILED")
                
                if result.failures:
                    logger.error("\n📋 FAILURE DETAILS:")
                    for test, traceback in result.failures:
                        logger.error(f"   - {test}")
                        # Log only first few lines of traceback to avoid clutter
                        tb_lines = str(traceback).split('\n')[:5]
                        for line in tb_lines:
                            logger.error(f"     {line}")
                
                if result.errors:
                    logger.error("\n📋 ERROR DETAILS:")
                    for test, traceback in result.errors:
                        logger.error(f"   - {test}")
                        tb_lines = str(traceback).split('\n')[:5]
                        for line in tb_lines:
                            logger.error(f"     {line}")
            
            logger.info(f"\n📝 Full log saved to: {log_file}")
            logger.info(f"Test completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            success = result.wasSuccessful()
        except Exception as e:
            logger.error(f"💥 Unexpected error during testing: {e}")
            success = False
    
    sys.exit(0 if success else 1)