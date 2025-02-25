## RCSB Protein Structure Embedding Search

This repository contains the scripts to train a neural network model for Protein Structure Search based on 3D structure embeddings.
A web application implementing this method is available at [rcsb-embedding-search](http://embedding-search.rcsb.org)

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
The embedding model was trained to predict the maximum TM-score between pairs of 3D structures.
During training the model operated as a twin neural network, utilizing shared weights to produce embeddings for pairs of 3D structures. 
ESM3 weights were frozen and the aggregator network parameters were optimized to minimize the MSE between cosine similarity and TM-score. 
The training set was compiled by Chengxin Zhang, and is available in this Zenodo [repository](https://zenodo.org/records/7324964).
The dataset set consists of 115 million SCOPe domain pais with their corresponding TM-scores.

### Training
The embedding model con be trained with the python script `src/training/embedding_tm_score.py`. 
Previously ESM3 embeddings for the training set 3D structures need to be pre-computed

```shell
python src/training/embedding_tm_score.py \
        --config-path=/config_foler \
        --config-name=training_config_file
```

#### Computing ESM3 Embeddings
The script `src/scripts/esm3_embeddings_from_pdb.py` facilitates computing ESM3 embeddings for 3D structures
```shell
python src/scripts/esm3_embeddings_from_pdb.py \
        --pdb_path /path_to_pdb_structures \
        --out_path /output_path
```

#### Training Configuration File
Example of training configuration file
```yaml
computing_resources:
  devices: 1
  nodes: 1
  strategy: auto

training_set:
  batch_size: 2
  workers: 0
  tm_score_file: /<local_path>/<training_dataset_filename>.csv
  data_path: /<local_path>/<embedding-esm3>
  data_ext: pt
  
validation_set:
  batch_size: 2
  workers: 0
  tm_score_file: /<local_path>/<validation_dataset_filename>.csv
  data_path: /<local_path>/<embedding-esm3>
  data_ext: pt
  
training_parameters:
  learning_rate: 1e-5
  weight_decay: 0.
  warmup_epochs: 2
  epochs: 100
  check_val_every_n_epoch: 1
  epoch_size: 100000
  lr_frequency: 1
  lr_interval: epoch
  
embedding_network:
  _target_: networks.transformer_nn.TransformerEmbeddingCosine
  input_features: 1536
  nhead: 12
  num_layers: 6
  dim_feedforward: 3072
  hidden_layer: 1536
  res_block_layers: 12
```
### Inference

Embeddings for 3D structures can be calculated using the script `src/inference/embedding_inference.py`. 
ESM3 embeddings need to be precomputed before inference. 

```shell
python src/inference/embedding_inference.py \
        --config-path=/config_foler \
        --config-name=inference_config_file
```
#### Inference Configuration File
Example of inference configuration file

```yaml
checkpoint: /<local_path>/<model>.pt

inference_writer:
  _target_: writer.batch_writer.CsvBatchWriter
  postfix: csv
  write_interval: batch
  output_path: /<output_path>

inference_set:
  embedding_source: /<local_path>/<inference_dataset_filename>.txt
  embedding_path: /<local_path>/<embedding-esm3>
  batch_size: 1
  workers: 0

embedding_network:
  _target_: networks.transformer_nn.TransformerEmbeddingCosine
  input_features: 1536
  nhead: 12
  num_layers: 6
  dim_feedforward: 3072
  hidden_layer: 1536
  res_block_layers: 12
```
