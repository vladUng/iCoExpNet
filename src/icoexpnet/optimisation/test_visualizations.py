#!/usr/bin/env python

"""
@File    :   test_visualizations.py
@Time    :   2025/08/31
@Author  :   Vlad Ungureanu & Claude
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Visualization suite for bioinformatics pipeline optimization results
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import datetime
import json
import argparse
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse co    def generate_summary_report(self, results, fail_fast=False):
    Generate comprehensive text summary report.
        self.logger.info("📝 Generating summary report...")
        
        report_file = self.figures_dir / "optimization_summary.txt"line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualizations for bioinformatics optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_visualizations.py                    # Generate all visualizations
  python test_visualizations.py --fail-fast        # Exit immediately on first error
  python test_visualizations.py --test             # Quick test mode (reduced output)
        """
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately on first error instead of continuing"
    )
    
    parser.add_argument(
        "--test",
        action="store_true", 
        help="Run in test mode with reduced logging output"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../../results/optimization_tests/figures",
        help="Directory to save visualization files (default: ../../../results/optimization_tests/figures)"
    )
    
    return parser.parse_args()


def _handle_error(logger, message, fail_fast=False):
    """
    Handle errors with optional fail-fast behavior.
    
    Args:
        logger: Logger instance for error reporting
        message: Error message to log
        fail_fast: If True, exit immediately; if False, log warning and continue
    """
    if fail_fast:
        logger.error(f"💥 CRITICAL ERROR (fail-fast mode): {message}")
        logger.error("🚨 Exiting immediately due to --fail-fast flag")
        sys.exit(1)
    else:
        logger.warning(f"⚠️ WARNING: {message}")
        logger.warning("🔄 Continuing execution...")


def setup_visualization_logging(test_mode=False):
    """Set up logging for visualization generation with timestamp."""
    # Create logs directory if it doesn't exist
    log_dir = Path("../../../results/optimization_tests/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"visualizations_{timestamp}.log"
    
    # Configure logging level based on test mode
    log_level = logging.WARNING if test_mode else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    if not test_mode:
        logger.info("=" * 80)
        logger.info("📊 BIOINFORMATICS OPTIMIZATION VISUALIZATIONS")
        logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Visualization generation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger, log_file


class VisualizationGenerator:
    """
    Generates comprehensive visualizations for bioinformatics optimization results.
    
    Creates box plots, timing comparisons, statistical summaries, and research-ready figures.
    """
    
    def __init__(self, logger, output_dir="../../../results/optimization_tests/figures"):
        """Initialize visualization generator."""
        self.logger = logger
        
        # Create timestamp-based subdirectory
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.figures_dir = Path(output_dir) / f"run_{self.timestamp}"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path("../../../results/optimization_tests/results")
        
        # Set style preferences
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info(f"📁 Figures will be saved to: {self.figures_dir}")
    
    def load_performance_results(self, fail_fast=False):
        """Load the most recent performance test results."""
        self.logger.info("📂 Loading performance test results...")
        
        try:
            # Find the most recent results files
            modcon_files = list(self.results_dir.glob("modcon_performance_*.csv"))
            mevs_files = list(self.results_dir.glob("mevs_performance_*.csv"))
            imevs_files = list(self.results_dir.glob("imevs_performance_*.csv"))
            pipeline_files = list(self.results_dir.glob("pipeline_performance_*.csv"))
            
            results = {}
            
            if modcon_files:
                latest_modcon = max(modcon_files, key=os.path.getctime)
                results['modcon'] = pd.read_csv(latest_modcon)
                self.logger.info(f"✅ Loaded ModCon results: {len(results['modcon'])} experiments from {latest_modcon.name}")
            
            if mevs_files:
                latest_mevs = max(mevs_files, key=os.path.getctime)
                results['mevs'] = pd.read_csv(latest_mevs)
                self.logger.info(f"✅ Loaded MEVs results: {len(results['mevs'])} experiments from {latest_mevs.name}")
            
            if imevs_files:
                latest_imevs = max(imevs_files, key=os.path.getctime)
                results['imevs'] = pd.read_csv(latest_imevs)
                self.logger.info(f"✅ Loaded integrated MEVs results: {len(results['imevs'])} experiments from {latest_imevs.name}")
            
            if pipeline_files:
                latest_pipeline = max(pipeline_files, key=os.path.getctime)
                results['pipeline'] = pd.read_csv(latest_pipeline)
                self.logger.info(f"✅ Loaded pipeline results: {len(results['pipeline'])} experiments from {latest_pipeline.name}")
            
            return results
            
        except Exception as e:
            _handle_error(self.logger, f"Failed to load performance results: {e}", fail_fast)
            return {}
    
    def create_speedup_comparison_plot(self, results, fail_fast=False):
        """Create comprehensive speedup comparison across all methods."""
        self.logger.info("📊 Creating speedup comparison plot...")
        
        # Combine data from all methods
        combined_data = []
        
        for method, df in results.items():
            if not df.empty and 'speedup' in df.columns:
                method_data = df[['speedup', 'tf_count']].copy()
                method_data['method'] = method.upper()
                method_data['method_full'] = {
                    'modcon': 'ModCon Optimization',
                    'mevs': 'MEVs Optimization', 
                    'imevs': 'Integrated MEVs Optimization',
                    'pipeline': 'Complete Pipeline'
                }.get(method, method)
                combined_data.append(method_data)
        
        if not combined_data:
            _handle_error(self.logger, "No speedup data available for comparison plot", fail_fast)
            return None
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Box Plot: Speedup Distribution', 'Scatter: Speedup vs TF Count', 
                           'Bar Chart: Average Speedup by Method', 'Violin Plot: Speedup Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Box plot
        methods = combined_df['method_full'].unique()
        colors = px.colors.qualitative.Set3[:len(methods)]
        
        for i, method in enumerate(methods):
            method_data = combined_df[combined_df['method_full'] == method]['speedup']
            fig.add_trace(
                go.Box(y=method_data, name=method, marker_color=colors[i]),
                row=1, col=1
            )
        
        # 2. Scatter plot
        for i, method in enumerate(methods):
            method_df = combined_df[combined_df['method_full'] == method]
            fig.add_trace(
                go.Scatter(
                    x=method_df['tf_count'], 
                    y=method_df['speedup'],
                    mode='markers',
                    name=method,
                    marker=dict(color=colors[i])
                ),
                row=1, col=2
            )
        
        # 3. Bar chart - average speedup
        avg_speedup = combined_df.groupby('method_full')['speedup'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=avg_speedup['method_full'], 
                y=avg_speedup['speedup'],
                marker_color=colors[:len(avg_speedup)]
            ),
            row=2, col=1
        )
        
        # 4. Violin plot (simplified as box plot for plotly)
        for i, method in enumerate(methods):
            method_data = combined_df[combined_df['method_full'] == method]['speedup']
            fig.add_trace(
                go.Violin(y=method_data, name=method, fillcolor=colors[i]),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Bioinformatics Pipeline Optimization: Speedup Analysis",
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_yaxes(title_text="Speedup (x)", row=1, col=1)
        fig.update_xaxes(title_text="TF Count", row=1, col=2)
        fig.update_yaxes(title_text="Speedup (x)", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Average Speedup (x)", row=2, col=1)
        
        # Save plot
        filename = self.figures_dir / "speedup_comparison.html"
        fig.write_html(filename)
        
        # Also save as static image
        static_filename = self.figures_dir / "speedup_comparison.png"
        fig.write_image(static_filename, width=1200, height=800, scale=2)
        
        self.logger.info(f"✅ Speedup comparison plot saved: {filename}")
        return fig
    
    def create_timing_before_after_plot(self, results, fail_fast=False):
        """Create before/after timing comparison plots."""
        self.logger.info("📊 Creating before/after timing comparison plots...")
        
        # Create subplots for each method
        methods_with_data = [(method, df) for method, df in results.items() 
                            if not df.empty and 'avg_original_time' in df.columns and 'avg_optimized_time' in df.columns]
        
        if not methods_with_data:
            _handle_error(self.logger, "No timing data available for before/after comparison", fail_fast)
            return None
        
        n_methods = len(methods_with_data)
        rows = (n_methods + 1) // 2
        cols = 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{method.upper()} Timing Comparison" for method, _ in methods_with_data]
        )
        
        colors = ['#FF6B6B', '#4ECDC4']  # Red for original, teal for optimized
        
        for idx, (method, df) in enumerate(methods_with_data):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            # Prepare data for plotting
            x_labels = [f"TF{tf}" for tf in df['tf_count']]
            
            # Original times
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=df['avg_original_time'],
                    name=f'Original {method}',
                    marker_color=colors[0],
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Optimized times
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=df['avg_optimized_time'],
                    name=f'Optimized {method}',
                    marker_color=colors[1],
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Update axes for this subplot
            fig.update_xaxes(title_text="Experiment", row=row, col=col)
            fig.update_yaxes(title_text="Time (seconds)", row=row, col=col)
        
        fig.update_layout(
            height=300 * rows,
            title_text="Before vs After Optimization: Execution Times",
            barmode='group'
        )
        
        # Save plot
        filename = self.figures_dir / "timing_before_after.html"
        fig.write_html(filename)
        
        static_filename = self.figures_dir / "timing_before_after.png"
        fig.write_image(static_filename, width=1200, height=300 * rows, scale=2)
        
        self.logger.info(f"✅ Before/after timing plot saved: {filename}")
        return fig
    
    def create_statistical_summary_plot(self, results, fail_fast=False):
        """Create statistical summary with box plots and error bars."""
        self.logger.info("📊 Creating statistical summary plots...")
        
        # Create matplotlib figure for detailed statistical plots
        n_methods = len([method for method, df in results.items() if not df.empty])
        
        if n_methods == 0:
            _handle_error(self.logger, "No data available for statistical summary", fail_fast)
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Analysis of Optimization Performance', fontsize=16, y=0.98)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Colors for different methods
        colors = sns.color_palette("husl", n_methods)
        method_colors = {}
        
        plot_idx = 0
        
        # 1. Speedup distribution box plots
        ax = axes_flat[plot_idx]
        speedup_data = []
        labels = []
        
        for method, df in results.items():
            if not df.empty and 'speedup' in df.columns:
                speedup_data.append(df['speedup'].values)
                labels.append(method.upper())
                method_colors[method] = colors[len(method_colors)]
        
        if speedup_data:
            bp = ax.boxplot(speedup_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], [method_colors[method.lower()] for method in labels]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Speedup Distribution by Method')
            ax.set_ylabel('Speedup (x)')
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        # 2. Time saved percentage
        ax = axes_flat[plot_idx]
        time_saved_data = []
        time_saved_labels = []
        
        for method, df in results.items():
            if not df.empty and 'time_saved_percentage' in df.columns:
                time_saved_data.append(df['time_saved_percentage'].values)
                time_saved_labels.append(method.upper())
        
        if time_saved_data:
            bp = ax.boxplot(time_saved_data, tick_labels=time_saved_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], [method_colors[method.lower()] for method in time_saved_labels]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Time Saved Percentage by Method')
            ax.set_ylabel('Time Saved (%)')
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        # 3. Average execution times comparison
        ax = axes_flat[plot_idx]
        methods = []
        original_times = []
        optimized_times = []
        original_stds = []
        optimized_stds = []
        
        for method, df in results.items():
            if not df.empty and 'avg_original_time' in df.columns:
                methods.append(method.upper())
                original_times.append(df['avg_original_time'].mean())
                optimized_times.append(df['avg_optimized_time'].mean())
                original_stds.append(df['avg_original_time'].std())
                optimized_stds.append(df['avg_optimized_time'].std())
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            ax.bar(x - width/2, original_times, width, label='Original', 
                  yerr=original_stds, capsize=5, color='lightcoral', alpha=0.8)
            ax.bar(x + width/2, optimized_times, width, label='Optimized', 
                  yerr=optimized_stds, capsize=5, color='lightseagreen', alpha=0.8)
            
            ax.set_title('Average Execution Times with Standard Deviation')
            ax.set_ylabel('Time (seconds)')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        # 4. Correlation matrix of performance metrics
        ax = axes_flat[plot_idx]
        
        # Combine all numeric data for correlation analysis
        combined_numeric = pd.DataFrame()
        
        for method, df in results.items():
            if not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                method_df = df[numeric_cols].copy()
                method_df.columns = [f"{method}_{col}" for col in method_df.columns]
                
                if combined_numeric.empty:
                    combined_numeric = method_df
                else:
                    combined_numeric = pd.concat([combined_numeric, method_df], axis=1)
        
        if not combined_numeric.empty:
            # Select most relevant columns for correlation
            relevant_cols = [col for col in combined_numeric.columns 
                           if any(keyword in col for keyword in ['speedup', 'time_saved', 'communities', 'tf_count'])]
            
            if relevant_cols:
                corr_data = combined_numeric[relevant_cols].corr()
                
                im = ax.imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr_data.columns)))
                ax.set_yticks(range(len(corr_data.columns)))
                ax.set_xticklabels([col.replace('_', '\n') for col in corr_data.columns], rotation=45, ha='right')
                ax.set_yticklabels([col.replace('_', '\n') for col in corr_data.columns])
                ax.set_title('Performance Metrics Correlation Matrix')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
                # Add correlation values as text
                for i in range(len(corr_data)):
                    for j in range(len(corr_data.columns)):
                        ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}', ha='center', va='center',
                               color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        
        # Save matplotlib figure
        filename = self.figures_dir / "statistical_summary.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Statistical summary plot saved: {filename}")
        return filename
    
    def create_performance_dashboard(self, results, fail_fast=False):
        """Create comprehensive performance dashboard."""
        self.logger.info("📊 Creating performance dashboard...")
        
        # Calculate summary statistics
        summary_stats = {}
        for method, df in results.items():
            if not df.empty and 'speedup' in df.columns:
                summary_stats[method] = {
                    'avg_speedup': df['speedup'].mean(),
                    'median_speedup': df['speedup'].median(),
                    'min_speedup': df['speedup'].min(),
                    'max_speedup': df['speedup'].max(),
                    'std_speedup': df['speedup'].std(),
                    'experiments_count': len(df),
                    'total_time_saved': df.get('time_saved_percentage', pd.Series([0])).mean()
                }
        
        # Create dashboard figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Overall Speedup Summary', 'Experiments by Method',
                           'Speedup vs TF Count', 'Time Savings Distribution',
                           'Performance Variance', 'Method Comparison'],
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # 1. Overall speedup indicator
        if summary_stats:
            overall_avg_speedup = np.mean([stats['avg_speedup'] for stats in summary_stats.values()])
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_avg_speedup,
                    title={"text": "Overall Average Speedup"},
                    gauge={'axis': {'range': [None, 50]},
                           'bar': {'color': "darkblue"},
                           'bgcolor': "white",
                           'borderwidth': 2,
                           'bordercolor': "gray",
                           'steps': [{'range': [0, 10], 'color': 'lightgray'},
                                    {'range': [10, 25], 'color': 'yellow'},
                                    {'range': [25, 50], 'color': 'green'}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 20}}
                ),
                row=1, col=1
            )
        
        # 2. Pie chart of experiments by method
        if summary_stats:
            methods = list(summary_stats.keys())
            experiment_counts = [summary_stats[method]['experiments_count'] for method in methods]
            
            fig.add_trace(
                go.Pie(labels=[method.upper() for method in methods], values=experiment_counts),
                row=1, col=2
            )
        
        # 3. Speedup vs TF Count scatter
        combined_data = []
        for method, df in results.items():
            if not df.empty and 'speedup' in df.columns and 'tf_count' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['tf_count'], 
                        y=df['speedup'],
                        mode='markers',
                        name=method.upper(),
                        text=[f"Method: {method}<br>TF: {tf}<br>Speedup: {speedup:.1f}x" 
                             for tf, speedup in zip(df['tf_count'], df['speedup'])]
                    ),
                    row=2, col=1
                )
        
        # 4. Time savings histogram
        all_time_saved = []
        for method, df in results.items():
            if not df.empty and 'time_saved_percentage' in df.columns:
                all_time_saved.extend(df['time_saved_percentage'].values)
        
        if all_time_saved:
            fig.add_trace(
                go.Histogram(x=all_time_saved, name="Time Saved %"),
                row=2, col=2
            )
        
        # 5. Performance variance box plot
        for method, df in results.items():
            if not df.empty and 'speedup' in df.columns:
                fig.add_trace(
                    go.Box(y=df['speedup'], name=method.upper()),
                    row=3, col=1
                )
        
        # 6. Method comparison bar chart
        if summary_stats:
            methods = list(summary_stats.keys())
            avg_speedups = [summary_stats[method]['avg_speedup'] for method in methods]
            
            fig.add_trace(
                go.Bar(x=[method.upper() for method in methods], y=avg_speedups),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Bioinformatics Optimization Performance Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        filename = self.figures_dir / "performance_dashboard.html"
        fig.write_html(filename)
        
        self.logger.info(f"✅ Performance dashboard saved: {filename}")
        return fig
    
    def generate_summary_report(self, results, fail_fast=False):
        """Generate comprehensive text summary report."""
        self.logger.info("📝 Generating summary report...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.figures_dir / f"optimization_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BIOINFORMATICS OPTIMIZATION PERFORMANCE SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for method, df in results.items():
                if df.empty:
                    continue
                    
                f.write(f"{method.upper()} OPTIMIZATION RESULTS\n")
                f.write("-" * 40 + "\n")
                
                if 'speedup' in df.columns:
                    f.write(f"Number of experiments: {len(df)}\n")
                    f.write(f"Average speedup: {df['speedup'].mean():.1f}x\n")
                    f.write(f"Median speedup: {df['speedup'].median():.1f}x\n")
                    f.write(f"Min speedup: {df['speedup'].min():.1f}x\n")
                    f.write(f"Max speedup: {df['speedup'].max():.1f}x\n")
                    f.write(f"Standard deviation: {df['speedup'].std():.1f}x\n")
                
                if 'time_saved_percentage' in df.columns:
                    f.write(f"Average time saved: {df['time_saved_percentage'].mean():.1f}%\n")
                
                if 'avg_original_time' in df.columns:
                    f.write(f"Average original time: {df['avg_original_time'].mean():.3f}s\n")
                    f.write(f"Average optimized time: {df['avg_optimized_time'].mean():.3f}s\n")
                
                f.write("\n")
            
            # Overall summary
            all_speedups = []
            for method, df in results.items():
                if not df.empty and 'speedup' in df.columns:
                    all_speedups.extend(df['speedup'].values)
            
            if all_speedups:
                f.write("OVERALL SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total experiments tested: {len(all_speedups)}\n")
                f.write(f"Overall average speedup: {np.mean(all_speedups):.1f}x\n")
                f.write(f"Overall median speedup: {np.median(all_speedups):.1f}x\n")
                f.write(f"Best speedup achieved: {np.max(all_speedups):.1f}x\n")
                f.write(f"Worst speedup: {np.min(all_speedups):.1f}x\n")
        
        self.logger.info(f"✅ Summary report saved: {report_file}")
        return report_file


def run_visualizations():
    """
    Generate all visualization plots and reports.
    """
    args = parse_arguments()
    
    logger, log_file = setup_visualization_logging(test_mode=args.test)
    
    start_time = datetime.datetime.now()
    
    try:
        # Initialize visualization generator
        viz_gen = VisualizationGenerator(logger, output_dir=args.output_dir)
        
        # Load performance results
        results = viz_gen.load_performance_results(fail_fast=args.fail_fast)
        
        if not results:
            _handle_error(logger, "No performance results found to visualize", args.fail_fast)
            return False
        
        logger.info(f"📊 Loaded results for {len(results)} methods")
        
        # Generate visualizations
        if not args.test:
            logger.info("\n🎨 Starting visualization generation...")
        
        # 1. Speedup comparison plot
        speedup_fig = viz_gen.create_speedup_comparison_plot(results, fail_fast=args.fail_fast)
        
        # 2. Before/after timing plots
        timing_fig = viz_gen.create_timing_before_after_plot(results, fail_fast=args.fail_fast)
        
        # 3. Statistical summary plots
        stats_plot = viz_gen.create_statistical_summary_plot(results, fail_fast=args.fail_fast)
        
        # 4. Performance dashboard
        dashboard_fig = viz_gen.create_performance_dashboard(results, fail_fast=args.fail_fast)
        
        # 5. Generate summary report
        report_file = viz_gen.generate_summary_report(results, fail_fast=args.fail_fast)
        
        # Final summary
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 VISUALIZATION GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Generation duration: {duration}")
        logger.info(f"Figures saved to: {viz_gen.figures_dir}")
        
        # List generated files
        generated_files = list(viz_gen.figures_dir.glob("*"))
        logger.info(f"\n📁 Generated {len(generated_files)} files:")
        for file in sorted(generated_files, key=os.path.getctime, reverse=True)[:10]:
            logger.info(f"  • {file.name}")
        
        logger.info(f"\n📝 Full log saved to: {log_file}")
        logger.info(f"Generation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
    
    except Exception as e:
        _handle_error(logger, f"Unexpected error during visualization generation: {e}", args.fail_fast)
        return False


if __name__ == "__main__":
    success = run_visualizations()
    sys.exit(0 if success else 1)