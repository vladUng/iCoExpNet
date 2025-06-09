#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   parallel.py
@Time    :   2024/01/15 18:31:29
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Generate multiple iNET experiments in parallel for both controls and non-controls.
'''

import multiprocess as mp
import pandas as pd 
import time as time
import os
import sys

# Tell Python where to find the iCoExpNet module
# Add the parent directory of 'examples/' to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from iCoExpNet.main import iCoExpNet


# Worker used for multiprocessing
def worker(arg):
    obj, meth_name = arg[:2]
    _ = getattr(obj, meth_name)()
    return obj


########## First we need to create the objects to run ##########
start_time = time.time()
experiments = []


# Define the input folder and files
input_folder = "data/"
ge_file = 'test_data_10000_genes.tsv'
gene_subset_file = 'TF_names_v_1.01.txt'
mut_file = 'test_mutation_data.tsv'

# Edge weight modifiers in used, choose between: reward, penalised, standard, sigmoid
modifiers = ["reward", 'penalised', 'standard', 'sigmoid']
genes_kept = 5000

# Configuration for stochastic block model
sbm_config = {
    'method': 'hsbm', # sbm or hsbm
    'n_iter': 10, # number of iterations for the SBM; recommended value 10000
    'mc_iter': 2, # number of Monte Carlo iterations for the SBM, recommended value 10
    'deg_cor': True # degree corrected for the SBM
}

edges_pg, edges_sel = 3, 6

# for non-control runs
if 1:
    output_folder = 'results/test/'
    label = 'testData'

    # Step 1: Generate the network objects with the parameters we want to test
    # e.g. here we generate two networks with 3 and 5 edges for the selected subset genes
    for edges_sel in range(3, 5):

        for modifier in ['standard']:
            # name the network to contain the variables that are changing
            # e.g. label (recognise from other experiments), modifier (weight), edges_sel (number of connections)
            name = f"{label}_{modifier}_{edges_sel}TF"

            inet = iCoExpNet(exp_name=name, ge_file = ge_file, input_folder=input_folder, output_folder=output_folder, gene_subset_file=gene_subset_file, mut_file=mut_file, edges_pg=edges_pg, edges_sel=edges_sel, modifier_type=modifier, genes_kept=genes_kept, sbm_config=sbm_config)
            experiments.append(inet)


    # Run the experiments ina parallel
    if __name__ == "__main__":
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(worker, ((exp, "run") for exp in experiments))
        pool.close()  
        pool.join()    

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\n######## iCoExpNet Experiments took {execution_time:.4f} seconds to execute\n\n")

# Run this for controls
if 1:
 
    output_folder = 'results/testCtrl/'
    label = 'testCtrlData'

    # read the control genes - these are non-TF genes that are used to generate the control networks with higher number of edges
    path_ctrls = 'data/controls/'
    tf_ctrls = next(os.walk(path_ctrls), (None, None, []))[2]

    ctrls = []
    for tf_ctrl in tf_ctrls:
        if '.DS_Store' == tf_ctrl:
            continue
        
        ctrls.append(tf_ctrl)

    modifier = 'standard'

    for idx, tf_ctrl in enumerate(ctrls):
        print(tf_ctrl)

        for edges_sel in range(3, 5):
            name = f"{label}_{modifier}_{edges_sel}TF"
            print(f'{output_folder}/tctrl_{idx+1}')

            inet = iCoExpNet(exp_name=name, ge_file = ge_file, input_folder=input_folder, output_folder=f'{output_folder}/tctrl_{idx+1}', gene_subset_file=gene_subset_file, mut_file=mut_file, edges_pg=edges_pg, edges_sel=edges_sel, modifier_type=modifier, genes_kept=genes_kept, sbm_config=sbm_config)

            # Setting the gene expression file for the control network
            inet.set_sel_ge(folder_path=path_ctrls, filename=tf_ctrl)
            experiments.append(inet)

    print(len(experiments))
    if __name__ == "__main__":
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(worker, ((exp, "run") for exp in experiments))
        pool.close()
        pool.join()    

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\n######## iCoExpNet Experiments took {execution_time:.4f} seconds to execute\n\n")