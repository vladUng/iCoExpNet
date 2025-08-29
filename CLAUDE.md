# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management & Development

This is a PyPI package (`icoexpnet` v0.1.12) built with setuptools.

**Install from PyPI:**
```bash
pip install icoexpnet
```

**Install in development mode:**
```bash
pip install -e .
```

**Build the package:**
```bash
python -m build
```

**Release process:**
```bash
# 1. Update version in pyproject.toml
# 2. Run the release script to commit, tag, and push
./release.sh

# The GitHub Actions workflow will automatically:
# - Generate CHANGELOG.md
# - Build the package  
# - Upload to PyPI using PYPI_API_TOKEN secret
```

**Run example experiments:**
```bash
# Single network experiment
python src/icoexpnet/examples/playground.py

# Parallel experiments (multiple networks)  
python src/icoexpnet/examples/parallel_playground.py
```

## Architecture Overview

iCoExpNet is a Python toolkit for building and analyzing gene co-expression networks from transcriptomic data with mutation-aware edge weighting and community detection algorithms.

### Core Components

**Main Pipeline (`src/icoexpnet/core/main.py`):**
- `iCoExpNet` class: Central pipeline orchestrating the entire network analysis
- Handles data loading, preprocessing, correlation computation, selective edge pruning, and community detection
- Supports multiple weight modification types: `standard`, `reward`, `penalised`, `sigmoid`
- Implements both Leiden algorithm (via igraph) and Stochastic Block Models (via graph-tool)

**Analysis Modules (`src/icoexpnet/analysis/`):**
- `ExperimentSet.py`: Experiment management and batch processing
- `GraphHelper.py`: Graph manipulation utilities 
- `GraphToolExp.py`: Graph-tool specific implementations
- `NetworkComp.py`: Network comparison and analysis
- `NetworkOutput.py`: Output formatting and export utilities

**Utilities (`src/icoexpnet/analysis/utilities/`):**
- `clustering.py`: Community detection algorithms
- `helpers.py`: General utility functions
- `pre_processing.py`: Data preprocessing utilities
- `sankey_consensus_plot.py`: Visualization utilities

### Key Pipeline Steps

1. **Data Loading**: Load gene expression (TPM), transcription factor lists, and mutation data
2. **Preprocessing**: Filter top N most variable genes (default: 5000) using relative standard deviation
3. **Correlation**: Compute Spearman correlation matrix between genes
4. **Weight Modification**: Apply mutation-aware edge weight modifiers (optional)
5. **Selective Edge Pruning**: Retain top K edges per gene (3 for regular genes, 6 for transcription factors)
6. **Network Construction**: Create igraph and/or graph-tool network objects
7. **Community Detection**: Run Leiden algorithm and/or Stochastic Block Models (SBM/hSBM)
8. **Analysis**: Generate statistics, export results, and create visualizations

### Data Requirements

**Required input files:**
- Gene expression data (TPM format, `.tsv` with `gene` as index column)
- Transcription factor list (`.txt` file for selective edge pruning)

**Optional input files:**
- Mutation data (`.tsv` with `gene` as index, `count` column for weight modification)

**Test data available in `data/` directory:**
- `test_data_10000_genes.tsv`: Sample gene expression data
- `TF_names_v_1.01.txt`: Transcription factor list
- `test_mutation_data.tsv`: Sample mutation data

### Configuration Parameters

**Edge pruning:**
- `edges_pg`: Number of edges per regular gene (default: 3)
- `edges_sel`: Number of edges per transcription factor (default: 6) 
- `genes_kept`: Number of top variable genes to analyze (default: 5000)

**Community detection:**
- `graph_type`: "gt" (graph-tool) or "ig" (igraph)
- `sbm_method`: "sbm" or "hsbm" for stochastic block models
- `mod_type`: "mod_max" or "CPM" for Leiden algorithm

**Weight modification:**
- `modifier_type`: "standard", "reward", "penalised", or "sigmoid"

### Output Structure

Results are organized in `results/` directory:
```
results/
├── Networks/[experiment_name]/EPG[edges_per_gene]/
│   ├── Leiden/Best/                     # Top 10 Leiden partitions
│   ├── gene_stats.tsv                   # Gene-level statistics
│   └── gt_[sbm_method]_[exp_name].pickle # Pickled SBM results
├── Processed/                           # Cached correlation matrices
├── Stats/                               # Experiment metadata and objects
└── old/                                 # Archived results
```

### Dependencies

**Required Python packages:**
- Core: `pandas`, `numpy`, `scipy`, `matplotlib`, `plotly`, `scikit-learn`
- Network: `igraph`, `leidenalg` 
- Data: `pyarrow`, `nbformat`, `lifelines`

**Critical external dependency:**
- `graph-tool` must be installed separately via system package manager or conda
- This is NOT available via pip and must be installed before using SBM features

**Installation:**
```bash
# Linux
sudo apt install python3-graph-tool

# Conda (recommended)
conda install -c conda-forge graph-tool
```

### Package Structure

Built with setuptools, packages defined in `pyproject.toml`:
- `icoexpnet`: Main package
- `icoexpnet.core`: Core pipeline functionality
- `icoexpnet.analysis`: Analysis modules and utilities
- `icoexpnet.analysis.utilities`: Helper functions

GitHub repository: https://github.com/vladUng/iCoExpNet