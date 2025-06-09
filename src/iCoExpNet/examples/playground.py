#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   playground.py
@Time    :   2024/01/15 18:32:05
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Generate a single iNet experiment.
'''
import pandas as pd 
import time as time
import os
import sys

# Tell Python where to find the iCoExpNet module
# Add the parent directory of 'examples/' to sys.path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, package_path)

# import the iCoExpNet class
from icoexpnet.core import iCoExpNet

# Worker used for multiprocessing
def worker(arg):
    obj, meth_name = arg[:2]
    _ = getattr(obj, meth_name)()
    return obj


########## First we need to create the objects to run ##########
start_time = time.time()

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


label = 'testData'
modifier = 'standard' # Choose one of the modifiers from the list above: reward, penalised, standard, sigmoid
name = f"{label}_{modifier}_{edges_sel}TF"

output_folder = 'results/test/'

inet = iCoExpNet(exp_name=name, ge_file = ge_file, input_folder=input_folder, output_folder=output_folder, gene_subset_file=gene_subset_file, mut_file=mut_file, edges_pg=edges_pg, edges_sel=edges_sel, modifier_type=modifier, genes_kept=genes_kept, sbm_config=sbm_config)

inet.run()

