## Clustering/Segmentation on [eTraM dataset](https://eventbasedvision.github.io/eTraM/) 

Want to separate between objects and noise so are using DBSCAN variants.

Spatio-Temporal DBSCAN with two epsilon distance thresholds: eps_spatial , eps_temporal
Hierarchical DBSCAN - varies density threshold to find the strongest clusters
Mean Shift - not useful here, does not have a noise category
