import os.path
import signal

import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch_geometric.data.lightning import LightningDataset

from torch_geometric.loader import DataLoader

from dataset.geo_graph_dataset import GeoGraphDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from lightning_module.lightning_geo_graph import LitStructureGeoGraph
from networks.transformer_graph_nn import TransformerGraphEmbeddingCosine
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':

    params = StructureEmbeddingParams()

    train_classes = params.train_class_file
    train_embedding = params.train_embedding_path
    test_classes = params.test_class_file
    test_embedding = params.test_embedding_path

    training_set = GeoGraphDataset(
        train_classes,
        train_embedding,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5)
    )
    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=params.epoch_size if params.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = DataLoader(
        training_set,
        sampler=sampler,
        batch_size=params.batch_size,
        num_workers=params.workers,
        persistent_workers=True if params.workers > 0 else False
    )

    validation_set = GeoGraphDataset(
        test_classes,
        test_embedding
    )
    validation_sampler = CustomWeightedRandomSampler(
        weights=validation_set.weights(),
        num_samples=1000,
        replacement=True
    )
    validation_dataloader = DataLoader(
        validation_set,
        sampler=validation_sampler,
        batch_size=params.testing_batch_size,
        num_workers=params.workers,
        persistent_workers=True if params.workers > 0 else False
    )

    net = TransformerGraphEmbeddingCosine()
    model = LitStructureGeoGraph.load_from_checkpoint(
        params.checkpoint,
        params=params
    ) if os.path.isfile(params.checkpoint) else LitStructureGeoGraph(
        nn_model=net,
        learning_rate=params.learning_rate,
        params=params,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructureGeoGraph.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{'+LitStructureGeoGraph.PR_AUC_METRIC_NAME+':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        max_epochs=params.epochs,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        devices=params.devices,
        strategy=params.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
    )
    datamodule = LightningDataset(
        train_dataset=training_set,
        val_dataset=validation_set
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=params.checkpoint if os.path.isfile(params.checkpoint) else None
    )
