# iCoExpNet
A Python toolkit for building and analysing gene co-expression networks from transcriptomic data with mutation-aware edge weighting and community detection.


# Weight modifiers 

There are four different options to compute the edges weights:
* standard - no change to the spearman correlation
* reward - increase the weights proportional to the mutations 
* sigmoid - proportional but has a sigmoid like function to increase the edges weights
* penalised - reduced the edges weights proportional to the mutations

TODO: add graph to show the different types of edge weights modifier



# To-Do

* The mutation file is not always needed so adapt the code to have the mutation file as an optional