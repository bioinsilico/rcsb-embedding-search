import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader as GraphDataLoader

from dataset.tm_score_from_coord_dataset import TmScoreFromCoordDataset
from dataset.tm_score_polars_dataset import TmScorePolarsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from dataset.utils.tools import collate_fn
from lightning_module.lightning_pst_graph import LitStructurePstGraph
from networks.transformer_pst import TransformerPstEmbeddingCosine
from schema.config import TrainingConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainingConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="training_default")
def main(cfg: TrainingConfig):
    seed_everything(cfg.global_seed, workers=True)

    training_set = TmScoreFromCoordDataset(
        tm_score_file=cfg.training_set.tm_score_file,
        coords_path=cfg.training_set.data_path,
        ext=cfg.training_set.data_ext,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5),
        num_workers=cfg.computing_resources.workers,
        coords_augmenter=instantiate(
            cfg.training_set.data_augmenter
        ) if cfg.training_set.data_augmenter is not None else None
    )

    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=cfg.training_parameters.epoch_size if cfg.training_parameters.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = GraphDataLoader(
        dataset=training_set,
        sampler=sampler,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.computing_resources.workers,
        persistent_workers=True if cfg.computing_resources.workers > 0 else False
    )

    validation_set = TmScorePolarsDataset(
        tm_score_file=cfg.validation_set.tm_score_file,
        embedding_path=cfg.validation_set.data_path
    )
    validation_dataloader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.computing_resources.workers,
        persistent_workers=True if cfg.computing_resources.workers > 0 else False,
        collate_fn=collate_fn
    )

    nn_model = TransformerPstEmbeddingCosine(
        pst_model_path=cfg.embedding_network.pst_model_path,
        input_features=cfg.embedding_network.input_layer,
        nhead=cfg.embedding_network.nhead,
        num_layers=cfg.embedding_network.num_layers,
        dim_feedforward=cfg.embedding_network.dim_feedforward,
        hidden_layer=cfg.embedding_network.hidden_layer,
        res_block_layers=cfg.embedding_network.res_block_layers
    )

    model = LitStructurePstGraph(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        params=cfg,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructurePstGraph.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitStructurePstGraph.PR_AUC_METRIC_NAME + ':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        max_epochs=cfg.training_parameters.epochs,
        check_val_every_n_epoch=cfg.training_parameters.check_val_every_n_epoch,
        num_nodes=cfg.computing_resources.nodes,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.default_root_dir
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint
    )


if __name__ == '__main__':
    main()
