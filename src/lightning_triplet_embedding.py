import lightning as L

from torch.utils.data import DataLoader

from src.dataset.structure_embedding_dataset import StructureEmbeddingDataset
from src.dataset.triplet_embedding_dataset import TripletEmbeddingDataset
from dataset.utils.tools import collate_fn, triplet_collate_fn
from src.lightning_module.lightning_triplet import LitTripletEmbedding
from src.networks.transformer_nn import TransformerEmbedding
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':

    params = StructureEmbeddingParams()

    train_classes = params.train_class_file
    train_embedding = params.train_embedding_path
    test_classes = params.test_class_file
    test_embedding = params.test_embedding_path

    training_set = TripletEmbeddingDataset(
        train_classes,
        train_embedding
    )
    train_dataloader = DataLoader(
        training_set,
        shuffle=True,
        batch_size=params.batch_size,
        num_workers=2,
        persistent_workers=True,
        collate_fn=triplet_collate_fn
    )

    testing_set = StructureEmbeddingDataset(
        test_classes,
        test_embedding
    )
    test_dataloader = DataLoader(
        testing_set,
        batch_size=params.batch_size,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    model = LitTripletEmbedding(
        net=TransformerEmbedding(
            input_features=params.input_layer,
            dim_feedforward=params.dim_feedforward,
            hidden_layer=params.hidden_layer,
            nhead=params.nhead,
            num_layers=params.num_layers
        ),
        learning_rate=params.learning_rate,
        params=params,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitTripletEmbedding.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{'+LitTripletEmbedding.PR_AUC_METRIC_NAME+':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        val_check_interval=params.test_every_n_steps,
        max_steps=params.test_every_n_steps * params.epochs,
        devices=params.devices,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model, train_dataloader, test_dataloader)
