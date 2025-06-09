# iCoExpNet
A Python toolkit for building and analysing gene co-expression networks from transcriptomic data with mutation-aware edge weighting and community detection.


# Setup guide

TODO

# How to use iCoExpNet

* After installation you can use the example/parallel_playground.py to generate two different types of network - with the control genes for TF and the ones from Human Transcription Factor
* example/playground.py is to run a single network
  

Note: Make sure that you have configured the desired data paths and files, look in the data/ folder for more information.

TODO: Explain the types of input files and their formats.

# Weight modifiers 

There are four different options to compute the edges weights:
* standard - no change to the spearman correlation
* reward - increase the weights proportional to the mutations 
* sigmoid - proportional but has a sigmoid like function to increase the edges weights
* penalised - reduced the edges weights proportional to the mutations

TODO: add graph to show the different types of edge weights modifier



# To-Do

* The mutation file is not always needed so adapt the code to have the mutation file as an optional