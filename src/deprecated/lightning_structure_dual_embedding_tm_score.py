import os.path
import signal

import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.profilers import PyTorchProfiler

from torch.utils.data import DataLoader

from deprecated.dataset.tm_dual_score_polars_dataset import TmDualScorePolarsDataset
from dataset.tm_score_from_embeddings_dataset import TmScoreFromEmbeddingsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from dataset.utils.tools import collate_fn, collate_dual_fn
from lightning_module.lightning_dual_embedding import LitStructureDualEmbedding
from networks.transformer_dual_embedding_nn import TransformerDualEmbeddingCosine
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':

    params = StructureEmbeddingParams()

    train_classes = params.train_class_file
    train_embedding = params.train_embedding_path
    test_classes = params.test_class_file
    test_embedding = params.test_embedding_path

    training_set = TmDualScorePolarsDataset(
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
        persistent_workers=True if params.workers > 0 else False,
        collate_fn=collate_dual_fn
    )

    validation_set = TmScoreFromEmbeddingsDataset(
        test_classes,
        test_embedding
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=params.testing_batch_size,
        num_workers=params.workers,
        persistent_workers=True if params.workers > 0 else False,
        collate_fn=collate_fn
    )

    nn_model = TransformerDualEmbeddingCosine(
        input_features=params.input_layer,
        dim_feedforward=params.dim_feedforward,
        hidden_layer=params.hidden_layer,
        nhead=params.nhead,
        num_layers=params.num_layers
    )
    model = LitStructureDualEmbedding(
        nn_model=nn_model,
        learning_rate=params.learning_rate,
        params=params,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructureDualEmbedding.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{'+LitStructureDualEmbedding.PR_AUC_METRIC_NAME+':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    if params.profiler_file:
        profiler = PyTorchProfiler(
            filename=params.profiler_file
        )
    else:
        profiler = None

    trainer = L.Trainer(
        max_epochs=params.epochs,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        devices=params.devices,
        strategy=params.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=params.default_root_dir if os.path.isdir(params.default_root_dir) else None,
        profiler=profiler
    )

    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=params.checkpoint if os.path.isfile(params.checkpoint) else None
    )
