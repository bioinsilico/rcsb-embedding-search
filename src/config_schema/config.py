from __future__ import annotations

from dataclasses import dataclass, MISSING
from enum import Enum
from pathlib import Path
from typing import Optional, Any


@dataclass
class TrainingConfig:
    computing_resources: ComputingResources = MISSING
    global_seed: int = MISSING
    checkpoint: Optional[str]
    default_root_dir: Optional[str]
    training_set: TMScoreDataset = MISSING
    validation_set: TMScoreDataset = MISSING
    training_parameters: TrainingParameters = MISSING
    network_parameters: NetworkParams = MISSING
    embedding_network: Any = MISSING
    metadata: Optional[str]


@dataclass
class InferenceConfig:
    computing_resources: ComputingResources = MISSING
    checkpoint: str = MISSING
    network_parameters: NetworkParams = MISSING
    embedding_network: Any = MISSING
    inference_set: EmbeddingDataset = MISSING
    inference_writer: Any = MISSING


@dataclass
class NetworkParams:
    input_features: int = MISSING
    nhead: int = MISSING
    num_layers: int = MISSING
    dim_feedforward: int = MISSING
    hidden_layer: int = MISSING


@dataclass
class EmbeddingDataset:
    embedding_source: Path = MISSING
    embedding_path: Path = MISSING
    batch_size: int = MISSING


@dataclass
class TMScoreDataset:
    tm_score_file: Path = MISSING
    data_path: Path = MISSING
    data_ext: Optional[DataExtension]
    batch_size: int = MISSING
    embedding_tmp_path: Optional[Path]
    data_augmenter: Optional[Any] = None


@dataclass
class TrainingParameters:
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    warmup_epochs: int = MISSING
    epochs: int = MISSING
    check_val_every_n_epoch: int = MISSING
    epoch_size: int = MISSING
    lr_frequency: int = MISSING
    lr_interval: LrInterval = MISSING


@dataclass
class ComputingResources:
    devices: int = MISSING
    workers: int = MISSING
    strategy: Strategy = MISSING
    nodes: int = 1


class LrInterval(str, Enum):
    epoch = "epoch"
    step = "step"


class Strategy(str, Enum):
    auto = "auto"
    ddp = "ddp"


class DataExtension(str, Enum):
    pdb = "pdb"
    ent = "ent"
    mmcif = "mmcif"
    bcif = "bcif"
    none = ""