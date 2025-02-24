## RCSB Protein Structure Embedding Search

This repository contains the scripts to train a neural network model for Protein Structure Search based on 3D structure embeddings.

### Embedding Model
The embedding model consists of two components: 
- A protein language model PLM that computes residue-level embeddings a given 3D structure
- A transformer-based neural network that aggregates these residue-level embeddings into a single vector

![Embedding model architecture](assets/embedding-model-architecture.png)

#### PLM 
Protein residue-level embeddings are computed with the [ESM](https://www.evolutionaryscale.ai/) generative protein language model.

#### Residue Embedding Aggregator

The aggregator consists of six transformer encoder layers, with 3,072 neurons feedforward layer and ReLU activations. 
Following the encoders, a summation pooling operation and 12 fully connected residual layers aggregate the resulting embeddings into a single 1,536-dimensional vector.

### Model Training
The dataset was compiled by Chengxin Zhang, and is available in this Zenodo [repository](https://zenodo.org/records/7324964).
The training set consists of 115 million SCOPe domain pais with their corresponding TM-scores.
