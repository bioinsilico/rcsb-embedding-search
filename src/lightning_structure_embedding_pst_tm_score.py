import os.path
import signal

import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch_geometric.data.lightning import LightningDataset

from torch_geometric.loader import DataLoader

from dataset.pst.pst import load_pst_model
from dataset.tm_score_from_coord_dataset import TmScoreFromCoordDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from lightning_module.lightning_batch_graph import LitStructureBatchGraph
from networks.transformer_pst import TransformerPstEmbeddingCosine
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':

    params = StructureEmbeddingParams()

    train_classes = params.train_class_file
    train_embedding = params.train_embedding_path
    test_classes = params.test_class_file
    test_embedding = params.test_embedding_path

    training_set = TmScoreFromCoordDataset(
        train_classes,
        train_embedding,
        ext="",
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

    validation_set = TmScoreFromCoordDataset(
        test_classes,
        test_embedding,
        ext="pdb"
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=params.testing_batch_size,
        num_workers=params.workers,
        persistent_workers=True if params.workers > 0 else False
    )
    pst_model = load_pst_model({
        'model_path': params.pst_model_path
    })
    nn_model = TransformerPstEmbeddingCosine(
        pst_model=pst_model,
        input_features=params.input_layer,
        dim_feedforward=params.dim_feedforward,
        hidden_layer=params.hidden_layer,
        nhead=params.nhead,
        num_layers=params.num_layers
    )
    model = LitStructureBatchGraph(
        nn_model=nn_model,
        learning_rate=params.learning_rate,
        params=params,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructureBatchGraph.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitStructureBatchGraph.PR_AUC_METRIC_NAME + ':.2f}'
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
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=params.default_root_dir if os.path.isdir(params.default_root_dir) else None
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
