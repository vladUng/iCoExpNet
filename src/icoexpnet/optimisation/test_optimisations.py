#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_optimisations.py
@Time    :   2025/08/31 09:59:25
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Test how much faster the optimisations made to modcon, mev are compared with non-optimised and also check if the results are consistent.
'''

# Standard imports

from icoexpnet.analysis.utilities.helpers import save_fig, survival_plot
from icoexpnet.analysis.utilities import sankey_consensus_plot as sky
from icoexpnet.analysis.utilities import clustering as cs
from icoexpnet.analysis import GraphHelper as gh
from icoexpnet.analysis.GraphToolExp import GraphToolExperiment as GtExp
from icoexpnet.analysis.ExperimentSet import ExperimentSet
import os
import sys
import gc
import psutil
import datetime
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict

# iCoExpNet imports
import sys
sys.path.append('../..')

############## Set up the files ##############

print("Imports completed successfully")
print(f"Python version: {sys.version}")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

results_path = "../../../results/"
data_base = "../../../data/"
base_path = "../../../"
test_exps_path = "results/test/"
test_cltrs_path = "results/testCtrl/"

figures_path = f'{results_path}/memory_optimisation/'

mut_df = pd.read_csv(f"{data_base}/test_mutation_data.tsv",
                     sep="\t", index_col="gene")

log_file_path = f'{results_path}/memory_optimisation/memory_optimization_log_v1.1.tsv'


# tf list
tf_path = f"{data_base}/TF_names_v_1.01.txt"
if os.path.exists(tf_path):
    tf_list = np.genfromtxt(fname=tf_path, delimiter="\t",
                            skip_header=1, dtype="str")
    

############## Load experiments ##############

# --> Experiments
exp_test = ExperimentSet("test", base_path=base_path, exp_path=test_exps_path,
                         mut_df=mut_df, sel_sets=None, exp_type="iNet")

# Load the hsbm experiments with the real biological TFs
h_exps, h_entropy = GtExp.load_hsbm_exps(exp_test)
h_entropy["Type"] = "Experiment"


# --> Controls

# This gets the list of folders of the control experiments
folders = next(os.walk(base_path + test_cltrs_path), (None, None, []))[1]

# Create a dictionary of the control experiments where each key is the index of the control
test_ctrls = {}
for folder in folders:
    hCtrl_path = f"{test_cltrs_path}/{folder}/"
    idx = int(folder.split("tctrl_")[-1])
    test_ctrls[idx] = ExperimentSet(
        "tCtrl", base_path, hCtrl_path, mut_df, sel_sets=None, rel_path="../", exp_type="iNet")
    test_ctrls[idx].export_to_gephi(save=False)

# Load the hCtrl experiments with control TFs
ctrl_exps, cmb_df = {}, pd.DataFrame()

# Iterate over the control experiments
for key in range(1, len(test_ctrls) + 1, 1):
    print(f"-->Loading control experiment #{key}")
    exps, entropy = GtExp.load_hsbm_exps(test_ctrls[key])
    entropy["Type"] = "hCtrl{}".format(key)
    cmb_df = pd.concat([cmb_df, entropy], axis=0)
    ctrl_exps[key] = {"entropy": entropy, "exps": exps}

##### Mod Con comparison for h_exps

import multiprocess as mp

pool = mp.Pool(mp.cpu_count())

# Define a worker function for parallel processing
def worker(arg):
    obj, methname = arg[:2]
    _ = getattr(obj, methname)()
    return obj

results = pool.map(worker, ((exp, "get_ModCon") for exp in h_exps.values()))
h_exps_ref = {exp.extract_tf_number(exp.name): exp for exp in results}

results = pool.map(worker, ((exp, "get_ModCon_optimized")
                   for exp in h_exps.values()))
h_exps_optimised = {exp.extract_tf_number(exp.name): exp for exp in results}

# Check if are attrs are equal for each experiment

for key in h_exps_ref.keys():
    print(f"Comparing experiment {key}")
    exp_ref = h_exps_ref[key]
    exp_optimised = h_exps_optimised[key]
    
    # Compare main DataFrame attributes
    attrs = ['tpm_df', 'edges_df', 'nodes_df', 'meta_df', 'modCon', 'mevsMut']
    comparison_results = {}
    for attr in attrs:
        if hasattr(exp_optimised, attr) and hasattr(exp_ref, attr):
            df1 = getattr(exp_optimised, attr)
            df2 = getattr(exp_ref, attr)
            if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
                comparison_results[attr] = df1.equals(df2)
            else:
                comparison_results[attr] = (df1 == df2)
        else:
            comparison_results[attr] = None  # Attribute missing


##### Speed test for controls

#### --> Ref
# Compute the ModCon and MEV for controls
# ModCon
for key in ctrl_exps.keys():
    print(f"### ModCon for control #{key}")
    results = pool.map(worker, ((exp, "get_ModCon")
                       for exp in ctrl_exps[key]["exps"].values()))
    ctrl_exps[key]["exps"] = {exp.extract_tf_number(
        exp.name): exp for exp in results}

# MEV
for key in ctrl_exps.keys():
    print(f"### MEV for control #{key}")
    for key, exp in ctrl_exps[key]["exps"].items():
        sort_col = f"ModCon_{exp.type}_gt"
        exp.mevsMut, _ = exp.get_mevs(
            tpms=mut_df, modCon=exp.gt_modCon, sort_col=sort_col, num_genes=100, verbose=False)
        

# --> Optimized
for key in ctrl_exps.keys():
    print(f"### ModCon for control #{key}")
    results = pool.map(worker, ((exp, "get_ModCon_optimized")
                       for exp in ctrl_exps[key]["exps"].values()))
    ctrl_exps[key]["exps"] = {exp.extract_tf_number(
        exp.name): exp for exp in results}

# MEV
for key in ctrl_exps.keys():
    print(f"### MEV for control #{key}")
    for key, exp in ctrl_exps[key]["exps"].items():
        sort_col = f"ModCon_{exp.type}_gt"
        exp.mevsMut, _ = exp.get_mevs(
            tpms=mut_df, modCon=exp.gt_modCon, sort_col=sort_col, num_genes=100, verbose=False)