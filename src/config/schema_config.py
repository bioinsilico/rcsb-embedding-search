from __future__ import annotations

from dataclasses import dataclass, MISSING
from enum import Enum
from pathlib import Path
from typing import Optional, Any

@dataclass
class LoggerConfig:
    save_dir: str = "lightning_logs"
    name: str = "embedding_logs"


@dataclass
class TrainingConfig:
    local_folder: Optional[LocalFolder]
    computing_resources: ComputingResources = MISSING
    global_seed: int = MISSING
    checkpoint: Optional[str]
    default_root_dir: Optional[str]
    training_set: TMScoreDataset = MISSING
    validation_set: TMScoreDataset = MISSING
    training_parameters: TrainingParameters = MISSING
    network_parameters: NetworkParams = MISSING
    embedding_network: Any = MISSING
    metadata: Optional[Any] = None
    logger: LoggerConfig = LoggerConfig()


@dataclass
class InferenceConfig:
    local_folder: Optional[LocalFolder]
    computing_resources: ComputingResources = MISSING
    checkpoint: str = MISSING
    network_parameters: NetworkParams = MISSING
    embedding_network: Any = MISSING
    inference_set: EmbeddingDataset = MISSING
    inference_writer: InferenceWriter = MISSING
    metadata: Optional[Any] = None


@dataclass
class NetworkParams:
    input_features: int = MISSING
    nhead: int = MISSING
    num_layers: int = MISSING
    dim_feedforward: int = MISSING
    hidden_layer: int = MISSING


@dataclass
class EmbeddingDataset:
    local_folder: Optional[LocalFolder]
    embedding_source: Path = MISSING
    embedding_path: Path = MISSING
    batch_size: int = MISSING
    workers: Optional[int] = 0


@dataclass
class TMScoreDataset:
    local_folder: Optional[LocalFolder]
    tm_score_file: Path = MISSING
    data_path: Path = MISSING
    data_ext: Optional[DataExtension]
    batch_size: int = MISSING
    embedding_tmp_path: Optional[Path]
    training_class_file: Optional[Path]
    testing_class_file: Optional[Path]
    number_of_folds: Optional[int]
    fold_number: Optional[int]
    workers: Optional[int] = 0
    data_augmenter: Any = None
    include_self_comparison: bool = False
    fraction_score: Optional[int] = 10


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
    strategy: Strategy = MISSING
    nodes: int = 1


@dataclass
class InferenceWriter:
    _target_: Any
    output_path: Path
    postfix: str
    write_interval: WriteInterval


@dataclass
class LocalFolder:
    data: Path = MISSING
    trained_model: Path = MISSING


class LrInterval(str, Enum):
    epoch = "epoch"
    step = "step"


class Strategy(str, Enum):
    auto = "auto"
    ddp = "ddp"


class DataExtension(str, Enum):
    pdb = "pdb"
    ent = "ent"
    cif = "cif"
    bcif = "bcif"
    pt = "pt"
    gz = "gz"
    none = ""


class WriteInterval(str, Enum):
    batch = "batch"
    epoch = "epoch"
    batch_and_epoch = "batch_and_epoch"
