# RCSB Protein Structure Embedding Search

This repository contains the scripts to train a neural network model for Protein Structure Search based on protein-level embeddings.

## Protein Encoding
Our model ingest protein residue-level embeddings computed with a structural aware protein language model available in this [repo](https://github.com/BorgwardtLab/PST).

## Training and Testing Datasets
Training and testing the model was achieved using the same datasets described in this [publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007970).
The training and testing datasets were carefully curated from CATH and ECOD domain classifications, ensuring independence and relevance.
Ready-to-use residue-level embeddings for the training and testing domains in the datasets can be downloaded from [zenodo](https://zenodo.org/records/10995163). 
Domain classification identifiers for the different CATH and ECOD domain structures can be found in the `resource` folder.

## Training and Testing
The main script for training and testing is `lightning_structure_embedding.py`. Two params are required
- `--class_path` Should point to the path were CATH and ECOD classification Ids files are located. 
The files are expected to be named as `cath.tsv` and `ecod.tsv`
- `--embedding_path` Should point to the path where the training and testing residue-level embeddings are stored. 
The script expect CATH domains embeddings to be stored in `<embedding_path>/cath/embedding` and ECOD `<embedding_path>/ecod/embedding`