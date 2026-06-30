## RCSB Embedding Search Codebase

### 1. PROJECT OVERVIEW

This is a PyTorch Lightning-based repository for training and inferencing neural networks that predict protein structure similarity using 3D embeddings. The project implements a twin neural network architecture that learns to rank protein structure pairs based on TM-scores (Template Modeling scores, a structural similarity metric).

**Key Reference:** Segura et al. (2026) - "Multi-scale structural similarity embedding search across entire proteomes" (Bioinformatics)

---

### 2. PROJECT STRUCTURE

```
rcsb-embedding-search/
├── config/                          # Hydra configuration files
│   ├── training_config.yaml         # Default training configuration
│   ├── inference_config.yaml        # Default inference configuration
│   └── core_config.yaml             # Core configuration (devices, strategy)
├── src/
│   ├── config/                      # Configuration and utilities
│   │   ├── schema_config.py         # Dataclass schemas (TrainingConfig, InferenceConfig)
│   │   ├── logging_config.py        # Logging setup
│   │   └── utils.py                 # Configuration utilities
│   ├── networks/                    # Neural network models
│   │   ├── transformer_nn.py        # 3 main model classes
│   │   └── layers.py                # ResBlock implementation
│   ├── lightning_module/            # PyTorch Lightning modules
│   │   ├── lightning_core.py        # LitStructureCore (base class)
│   │   ├── utils.py                 # get_cosine_schedule_with_warmup
│   │   ├── training/
│   │   │   └── embedding_training.py # LitEmbeddingTraining, LitEmbeddingMaskedTraining
│   │   └── inference/
│   │       └── embedding_inference.py # LitEmbeddingInference, LitEmbeddingOrNullInference
│   ├── dataset/                     # Data loading and preprocessing
│   │   ├── embeddings_dataset.py    # EmbeddingsDataset - loads pre-computed embeddings
│   │   ├── tm_score_from_embeddings_dataset.py      # Training dataset with TM-score labels
│   │   ├── tm_score_masked_from_embeddings_dataset.py # With spatial attention masks
│   │   ├── dual_tm_score_from_embeddings_dataset.py  # Dual TM-score variant
│   │   ├── tm_score_fold_cross_validation_from_embeddings_dataset.py
│   │   ├── esm3_embeddings_from_coord_dataset.py     # ESM3 embedding computation
│   │   ├── esm2_embeddings_from_coord_dataset.py     # ESM2 embedding computation
│   │   ├── assembly_embeddings_from_chain_dataset.py # Multi-chain assembly handling
│   │   ├── biotite_structure_from_list.py            # Direct structure embedding
│   │   └── utils/
│   │       ├── tools.py             # collate_fn, padding, label utilities
│   │       ├── tm_score_weight.py   # TM-score weighting strategies
│   │       ├── embedding_provider.py
│   │       ├── neighbor_mask.py     # CA-neighbor attention mask utils
│   │       ├── custom_weighted_random_sampler.py
│   │       ├── coords_augmenter.py
│   │       ├── coords_getter.py
│   │       ├── geometric_tools.py
│   │       ├── geometry.py
│   │       ├── tools.py
│   │       └── biopython_getter.py
│   ├── training/                    # Training scripts
│   │   ├── embedding_tm_score.py        # Main training script (Hydra)
│   │   ├── masked_embedding_tm_score.py # Masked training variant
│   │   └── fold_cross_validation_tm_score.py
│   ├── inference/                   # Inference scripts
│   │   ├── embedding_inference.py      # Main inference script (Hydra)
│   │   ├── embedding_assembly_inference.py
│   │   └── embedding_from_list_inference.py
│   ├── scripts/                     # Utility scripts
│   │   ├── esm3_embeddings_from_pdb.py      # ESM3 embedding generation
│   │   ├── esm3_embeddings_from_rcsb.py     # ESM3 from RCSB API
│   │   ├── esm3_embeddings_from_cif.py      # ESM3 from CIF files
│   │   ├── esm3_assemblies_from_pdb.py      # Multi-chain assemblies
│   │   ├── esm3_assemblies_from_rcsb.py
│   │   ├── esm3_mean_embeddings.py          # Average embeddings
│   │   ├── ca_neighbors.py                  # CA-neighbor computation
│   │   ├── batch_ca_neighbors.py
│   │   ├── pairwise_sequence_identity.py
│   │   ├── extract_sequences.py
│   │   ├── save_model_for_production.py
│   │   └── prepare_datasets.py
│   └── writer/
│       └── batch_writer.py          # CSV output writer
└── environment.yml                  # Conda environment (Python 3.10, PyTorch 2.7)
```

---

### 3. PYTORCH & LIGHTNING MODELS

#### **A. Neural Network Models** (`src/networks/`)

**1. LinearEmbeddingCosine** (input_features=1280)
- Path: `src/networks/transformer_nn.py:7-26`
- Simple linear aggregator for sequence embeddings
- Architecture: LayerNorm → Dropout → Linear → ReLU
- Forward: Sums sequence embeddings along dim=1, applies transformation, returns cosine similarity

**2. TransformerEmbeddingCosine** (input_features=640, dim_feedforward=1280)
- Path: `src/networks/transformer_nn.py:28-79`
- Multi-head transformer encoder with residual blocks
- Architecture:
    - 6 transformer encoder layers (configurable)
    - 10 attention heads (configurable)
    - Feedforward layer: 1280 neurons
    - Summation pooling → LayerNorm/ResBlocks → Linear projection
    - Output: Cosine similarity between embeddings
- Key methods: `embedding_pooling()`, `forward()`

**3. MaskedTransformerEmbeddingCosine** (input_features=640)
- Path: `src/networks/transformer_nn.py:81-147`
- Same architecture as TransformerEmbeddingCosine but with spatial attention masking
- Accepts per-residue (L×L) attention masks restricting attention to CA-neighbors
- Forward signature: `forward(x, x_pad_mask, x_attn_mask, y, y_pad_mask, y_attn_mask)`

**4. ResBlock** (utility layer)
- Path: `src/networks/layers.py:4-28`
- Residual connection block: LayerNorm → Dropout → Linear → ReLU → LayerNorm → Dropout → Linear → ReLU + residual

#### **B. Lightning Modules** (`src/lightning_module/`)

**1. LitStructureCore** (base class)
- Path: `src/lightning_module/lightning_core.py:10-88`
- Extends `L.LightningModule`
- Manages:
    - Model wrapping, optimizer (AdamW), cosine annealing LR scheduler with warmup
    - Batch prediction accumulation (z_pred, z)
    - Metrics: PR-AUC, ROC-AUC, MSE loss
    - TensorBoard logging

**2. LitEmbeddingTraining**
- Path: `src/lightning_module/training/embedding_training.py:5-27`
- Extends `LitStructureCore`
- Training step: computes MSE loss between cosine_similarity and TM-score label
- Validation step: accumulates predictions for metrics

**3. LitEmbeddingMaskedTraining**
- Path: `src/lightning_module/training/embedding_training.py:29-50`
- Same as above but handles spatial attention masks

**4. LitEmbeddingInference**
- Path: `src/lightning_module/inference/embedding_inference.py:8-20`
- `predict_step()`: Returns `embedding_pooling()` output (residue-aggregated embeddings)

**5. LitEmbeddingOrNullInference**
- Path: `src/lightning_module/inference/embedding_inference.py:22-41`
- Error-tolerant variant, returns None on failure

---

### 4. DATA LOADING PATTERNS

#### **Embedding Format**
- **Source:** ESM3 (Evolutionary Scale Model) per-residue embeddings
- **ESM3 Details:**
    - Protein Language Model: Computes 1536-dimensional residue-level embeddings
    - ESM3 prepends a BOS (Beginning of Sequence) token
    - Multi-chain assemblies: Each chain contributes (N_residues + 2) tokens (BOS + residues + EOS)
    - Files stored as PyTorch tensors (`.pt` files) with shape `(seq_len, 1536)`

#### **ESM2 Alternative**
- Path: `src/dataset/esm2_embeddings_from_coord_dataset.py`
- Uses ESM2-t30-150M model: `esm.pretrained.esm2_t30_150M_UR50D()`
- Per-token alphabet tokenization via `alphabet.get_batch_converter()`

#### **Core Dataset Classes**

**1. EmbeddingsDataset**
- Path: `src/dataset/embeddings_dataset.py:14-54`
- Loads pre-computed embeddings from disk
- Returns: (tensor, domain_id)

**2. TmScoreFromEmbeddingsDataset**
- Path: `src/dataset/tm_score_from_embeddings_dataset.py:19-80`
- Twin dataset: Loads pairs of embeddings + TM-score labels
- Returns: (embedding_i, embedding_j, label)
- Supports weighted sampling based on TM-score distribution

**3. TmScoreMaskedFromEmbeddingsDataset**
- Path: `src/dataset/tm_score_masked_from_embeddings_dataset.py:62-90`
- Extends TmScoreFromEmbeddingsDataset
- Adds CA-neighbor attention masks (`.tsv` files)
- Returns: ((emb_i, pad_mask_i, attn_mask_i), (emb_j, pad_mask_j, attn_mask_j), label)

**4. DualTmScoreFromEmbeddingsDataset**
- Path: `src/dataset/dual_tm_score_from_embeddings_dataset.py`
- Supports dual TM-scores (global + local)

#### **Tokenization/Vocabulary**
- **NO traditional amino acid tokenization** - uses pre-computed ESM3 embeddings
- Sequence information is encoded in ESM3 model during embedding computation
- ESM3 handles special tokens (BOS at position 0, EOS after sequence)

#### **Collate Functions**

**collate_seq_embeddings()**
- Path: `src/dataset/utils/tools.py:104-133`
- Pads variable-length sequences to max length in batch
- Returns: (padded_batch shape (B, L, D), mask shape (B, L))
- Mask: False = real data, True = padding

**collate_fn()**
- Path: `src/dataset/utils/tools.py:143-148`
- Collates twin pairs: `((x, x_mask), (y, y_mask), labels)`

**collate_masked_fn()**
- Path: `src/dataset/tm_score_masked_from_embeddings_dataset.py:34-59`
- Collates with attention masks: `((x, x_pad_mask, x_attn_mask), (y, y_pad_mask, y_attn_mask), labels)`

---

### 5. TRAINING PIPELINE

#### **Main Training Script**
- Path: `src/training/embedding_tm_score.py`
- Framework: Hydra-based configuration
- Key steps:
    1. Load training dataset (TmScoreFromEmbeddingsDataset)
    2. Compute sample weights based on TM-score distribution
    3. Create weighted random sampler for balanced batching
    4. Instantiate model (via Hydra config)
    5. Create Lightning trainer with callbacks:
        - ModelCheckpoint (monitoring PR-AUC)
        - LearningRateMonitor
        - TensorBoardLogger

#### **Training Configuration** (`config/training_config.yaml`)
- **Global seed:** 42
- **Training parameters:**
    - Learning rate: 1e-5
    - Weight decay: 0 (default)
    - Warmup epochs: 2
    - Max epochs: 1000
    - Check validation every: 5 epochs
    - Epoch size: 320,000 samples
    - LR scheduler: Cosine annealing with warmup
- **Embedding network config:**
    - Input features: 1536 (ESM3 embedding dimension)
    - Num heads: 12
    - Num layers: 6
    - Feedforward dim: 3072
    - Hidden layer: 1536
    - Residual blocks: 12

#### **Loss Function**
- **MSE Loss:** Difference between cosine_similarity and TM-score
- Path: `src/lightning_module/training/embedding_training.py:20`

#### **Metrics**
- PR-AUC (Precision-Recall Area Under Curve)
- ROC-AUC (Receiver Operating Characteristic AUC)
- Computed at validation epoch end

#### **TM-Score Handling**

**TmScoreWeight class**
- Path: `src/dataset/utils/tm_score_weight.py:9-35`
- Divides scores into intervals (default n_intervals=5: [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0])
- Computes inverse frequency weights to balance samples across score ranges

**Scoring transformations:**
- `fraction_score_of(f=10)`: Rounds scores to nearest 0.1 (e.g., 0.87 → 0.9)
- `binary_score(thr=0.7)`: Converts to binary (similar if >= 0.7, else dissimilar)

---

### 6. INFERENCE PIPELINE

#### **Main Inference Script**
- Path: `src/inference/embedding_inference.py`
- Loads checkpoint, processes dataset, outputs embeddings to CSV

#### **Inference Configuration** (`config/inference_config.yaml`)
- **Checkpoint:** Path to `.ckpt` file
- **Inference set:** List of domain IDs + embedding path
- **Writer:** CsvBatchWriter (outputs to directory)

#### **Inference Process**
1. Load pre-computed embeddings
2. Batch with collate_seq_embeddings (padding)
3. Forward through model.embedding_pooling() to get aggregated vectors
4. Output as CSV

---

### 7. EMBEDDING COMPUTATION

#### **ESM3 Embedding Generation**

**esm3_embeddings_from_pdb.py**
- Path: `src/scripts/esm3_embeddings_from_pdb.py`
- Uses `ESM3.from_pretrained(ESM3_OPEN_SMALL)`
- Flow:
    - Load PDB file → ProteinChain → ESMProtein
    - Encode with ESM3 model
    - Sample with `SamplingConfig(return_per_residue_embeddings=True)`
    - Save `output.per_residue_embedding` as `.pt` file

**esm3_assemblies_from_pdb.py**
- Path: `src/scripts/esm3_assemblies_from_pdb.py`
- Multi-chain assemblies: Concatenates per-chain embeddings

**esm3_mean_embeddings.py**
- Path: `src/scripts/esm3_mean_embeddings.py`
- Averages per-residue embeddings to single vector per protein

#### **CA-Neighbor Masks**

**ca_neighbors.py**
- Path: `src/scripts/ca_neighbors.py`
- Computes spatial attention masks based on CA-atom distances
- Key functions:
    - `extract_ca_coords()`: Gets CA coordinates
    - `compute_neighbor_lists()`: Finds neighbors within threshold (default distance-based)
    - `build_esm3_index_map_*()`: Maps residues to ESM3 token positions (accounts for BOS token)
    - `write_tsv()`: Saves neighbor lists as TSV
- Outputs square boolean masks (L×L) where True = masked out

---

### 8. EMBEDDING & SIMILARITY CODE

#### **Similarity Computation**
- **Method:** Cosine similarity via `nn.functional.cosine_similarity()`
- **Location:** All three model forward methods
- **Dimensionality:** 1536-dimensional vectors (from 12-layer ResBlock aggregation)

#### **Embedding Aggregation Strategy**
1. **Input:** (B, L, 1536) - batch of variable-length sequences
2. **Transformer:** 6-layer encoder → (B, L, 1536)
3. **Summation pooling:** Sum along sequence dim → (B, 1536)
4. **ResBlocks:** 12 residual blocks with skip connections → (B, 1536)
5. **Output:** Final 1536-dim embedding vector

#### **Key Embedding Methods**
- `embedding_pooling(x, x_mask)`: Transformer + pooling + ResBlocks
- `embedding(x)`: Direct ResBlock/Linear projection (used post-pooling)

---

### 9. KEY CONFIGURATION DETAILS

#### **Config Schema** (`src/config/schema_config.py`)

**TrainingConfig:**
- computing_resources (devices, strategy, nodes)
- training_set / validation_set (TMScoreDataset)
- training_parameters (learning_rate, epochs, warmup, lr_schedule)
- embedding_network (Hydra instantiable)
- logger (TensorBoard)
- metadata (optional)

**InferenceConfig:**
- checkpoint, network_parameters, embedding_network
- inference_set (embedding_source, embedding_path, batch_size)
- inference_writer (CSV output config)

**Dataset configs:**
- tm_score_file: TSV with (domain_i, domain_j, score)
- embedding_path: Directory of `.pt` files
- neighbor_path: Optional directory of `.tsv` attention masks
- data_ext: File extension (pt, npy, etc.)
- tm_score_intervals: Number of score bins for weighting

---

### 10. AMINO ACID & VOCABULARY INSIGHTS

**No explicit amino acid vocabulary** because:
- Input uses pre-computed ESM3 embeddings (1536-dimensional)
- ESM3 is a foundation model trained on protein sequences
- Sequence information is already encoded in the embeddings

**ESM3 Model Details:**
- Open Small variant used
- Sequence context: Handles 20 standard amino acids + gaps
- BOS token management:
    - Single chain: BOS at position 0, residues at positions 1+
    - Multi-chain: Each chain has BOS + residues + EOS, then next chain continues
    - Mask remapping for correct attention: `build_esm3_index_map_*()`

---

### 11. FILE LOCATIONS SUMMARY

| Component | Location                                                           |
|-----------|--------------------------------------------------------------------|
| Main models | `src/networks/transformer_nn.py`                                   |
| Lightning modules | `src/lightning_module/`                            |
| Training datasets | `src/dataset/tm_score_*.py`                     |
| Training script | `src/training/embedding_tm_score.py`                               |
| Inference script | `src/inference/embedding_inference.py`                             |
| Collate functions | `src/dataset/utils/tools.py`                                       |
| TM-score weighting | `src/dataset/utils/tm_score_weight.py` |
| ESM3 embedding gen | `src/scripts/esm3_*.py`    |
| CA-neighbor masks | `src/scripts/ca_neighbors.py` |
| Config schemas | `src/config/schema_config.py` |
| Default configs | `config/*.yaml`            |

---

### 12. KEY INSIGHTS

1. **Architecture Pattern:** Twin network with shared weights for comparing protein structures
2. **No tokenization:** Uses pre-computed dense embeddings from ESM3 rather than token vocabularies
3. **Spatial-aware attention:** Optional CA-neighbor masking restricts transformer attention to realistic spatial neighborhoods
4. **Similarity metric:** Raw cosine similarity between 1536-dim vectors, trained to match TM-scores
5. **Flexible input sizes:** Handles variable-length sequences via padding and masks
6. **Multi-scale training:** Weighted sampling by TM-score intervals to balance learning
7. **Production ready:** Includes model checkpointing, inference pipelines, and batch processing scripts