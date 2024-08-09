from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GraphDataset


@dataclass
class Config:
    training_set: Dataset | GraphDataset
    training_loader: DataLoader | GraphDataLoader
    validation_set: Dataset | GraphDataset
    validation_loader: Dataset | GraphDataset
    global_seed: int
    training_sources: TrainingSources
    training_parameters: TrainingParameters
    computing_resources: ComputingResources
    nn_model: nn.Module


@dataclass
class TrainingSources:
    train_class_file: str
    train_embedding_path: str
    train_embedding_ext: str | None
    test_class_file: str
    test_embedding_path: str
    test_embedding_ext: str | None
    checkpoint: str | None
    default_root_dir: str | None


@dataclass
class TrainingParameters:
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    batch_size: int
    testing_batch_size: int
    epochs: int
    check_val_every_n_epoch: int
    epoch_size: int
    lr_frequency: int
    lr_interval: LrInterval


@dataclass
class ComputingResources:
    devices: 1
    workers: 1
    strategy: Strategy


class LrInterval(Enum):
    epoch = "epoch"
    step = "step"


class Strategy(Enum):
    auto = "auto"
    ddp = "ddp"
