import os.path

import lightning as L

from torch.utils.data import DataLoader

from src.dataset.structure_embedding_dataset import StructureEmbeddingDataset
from src.dataset.tm_score_dataset import TmScoreDataset, fraction_score, tm_score_weights
from src.dataset.utils import collate_fn, CustomWeightedRandomSampler
from src.lightning_module.lightning_embedding import LitStructureEmbedding
from src.networks.transformer_nn import TransformerEmbeddingCosine
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':

    params = StructureEmbeddingParams()

    train_classes = params.train_class_file
    train_embedding = params.train_embedding_path
    test_classes = params.test_class_file
    test_embedding = params.test_embedding_path

    training_set = TmScoreDataset(
        train_classes,
        train_embedding,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5)
    )
    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_dataloader = DataLoader(
        training_set,
        sampler=sampler,
        batch_size=params.batch_size,
        num_workers=params.workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    testing_set = StructureEmbeddingDataset(
        test_classes,
        test_embedding
    )
    test_dataloader = DataLoader(
        testing_set,
        batch_size=params.batch_size,
        num_workers=params.workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    net = TransformerEmbeddingCosine(
        input_features=params.input_layer,
        dim_feedforward=params.dim_feedforward,
        hidden_layer=params.hidden_layer,
        nhead=params.nhead,
        num_layers=params.num_layers
    )
    model = LitStructureEmbedding.load_from_checkpoint(
        params.checkpoint,
        net=net
    ) if os.path.isfile(params.checkpoint) else LitStructureEmbedding(
        net=net,
        learning_rate=params.learning_rate,
        params=params,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructureEmbedding.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{'+LitStructureEmbedding.PR_AUC_METRIC_NAME+':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        val_check_interval=params.test_every_n_steps,
        max_steps=params.test_every_n_steps * params.epochs,
        devices=params.devices,
        strategy=params.strategy,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model, train_dataloader, test_dataloader)
