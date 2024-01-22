# DNML for GMM Embedding
This repository has simulation codes for selecting the dimension and number of clusters for GMM graph embedding.
## Analysis of GMM data
Execute ```python artificial_gmm.py X Y Z W```, where X \in {0, 1, 2, 3, 4} indicates the number of nodes 2^X\*400, Y \in {2, 4, 8} is the true dimensionality, and Z \in {2, 4, 8} is the true number of clusters, and W \in {0, 1, 2, 3} indicates beta 0.2+3\*W.
## Analysis of SBM data
Execute ```python artificial_sbm.py X Y Z W```, where X \in {0, 1, 2, 3, 4} indicates the number of nodes 2^X\*400, Y \in {2, 4, 8} is the true number of blocks, and Z \in {0, 1, 2} indicates p_other 0.1\*(Z+1), and W \in {0, 1, 2, 3} indicates (p_same - p_other) 0.1\*(W+1).
