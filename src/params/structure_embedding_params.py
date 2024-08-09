import os

from hydra.utils import instantiate
from omegaconf import DictConfig


class StructureEmbeddingParams:

    def __init__(self, cfg: DictConfig, cfg_file_name: str):
        self.cfg = cfg
        self.config_file_name = cfg_file_name

        self.train_class_file = self.cfg.training_sources.train_class_file
        self.train_embedding_path = self.cfg.training_sources.train_embedding_path
        self.train_embedding_ext = self.cfg.training_sources.train_embedding_ext
        self.test_class_file = self.cfg.training_sources.test_class_file
        self.test_embedding_path = self.cfg.training_sources.test_embedding_path

        if (
                self.cfg.training_sources.checkpoint is not None
                and not os.path.isfile(self.cfg.training_sources.checkpoint)
        ):
            raise Exception(
                f"Config training_sources.checkpoint is defined but "
                f"file {self.cfg.training_sources.checkpoint} does not exists"
            )
        self.checkpoint = self.cfg.training_sources.checkpoint

        if (
                self.cfg.training_sources.default_root_dir is not None
                and not os.path.isdir(self.cfg.training_sources.default_root_dir)
        ):
            raise Exception(
                f"Config training_sources.default_root_dir is defined but "
                f"path {self.cfg.training_sources.checkpoint} does not exists"
            )
        self.default_root_dir = self.cfg.training_sources.default_root_dir

        self.devices = self.cfg.computing_resources.devices
        self.workers = self.cfg.computing_resources.workers
        self.strategy = self.cfg.computing_resources.strategy

        self.learning_rate = self.cfg.training_parameters.learning_rate
        self.weight_decay = self.cfg.training_parameters.weight_decay
        self.warmup_epochs = self.cfg.training_parameters.warmup_epochs
        self.batch_size = self.cfg.training_parameters.batch_size
        self.testing_batch_size = self.cfg.training_parameters.testing_batch_size
        self.epochs = self.cfg.training_parameters.epochs
        self.check_val_every_n_epoch = self.cfg.training_parameters.check_val_every_n_epoch
        self.epoch_size = self.cfg.training_parameters.epoch_size
        self.lr_frequency = self.cfg.training_parameters.lr_frequency
        self.lr_interval = self.cfg.training_parameters.lr_interval

        self.input_layer = self.cfg.network.input_layer
        self.dim_feedforward = self.cfg.network.dim_feedforward
        self.nhead = self.cfg.network.nhead
        self.num_layers = self.cfg.network.num_layers
        self.hidden_layer = self.cfg.network.hidden_layer
        self.res_block_layers = self.cfg.network.res_block_layers
        self.pst_model_path = self.cfg.network.pst_model_path

        self.global_seed = self.cfg.global_seed
        self.profiler_file = self.cfg.profiler_file
        self.metadata = self.cfg.metadata

        self.data_augmenter = instantiate(self.cfg.data_augmenter) if self.cfg.data_augmenter is not None else None

    def text_params(self, params=None):
        if params is None:
            params = {}
        params.update(self.__dict__)
        return '\n'.join(["%s: %s  " % (k, v) for k, v in params.items() if k != "cfg"])
