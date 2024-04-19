# RCSB Protein Structure Embedding Search

This repository provides the scripts to train a transformer network for Protein Structure Search based on Embeddings.

## Protein encoding
The model ingest protein embeddings computed with a structural aware protein language model [ref](https://github.com/BorgwardtLab/PST)

## Training and Testing datasets
Training and testing the model can be done using the same datasets described in this [publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007970).
Two ready-to-use CATH/ECOD domains datasets can be downloaded from [zenodo](https://zenodo.org/records/10995163). 
Domain classification for the different structures can be found in the `resource` folder.
