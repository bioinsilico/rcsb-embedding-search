import argparse
import lightning as L

from torch.utils.data import WeightedRandomSampler, DataLoader

from src.dataset.structure_embedding_dataset import StructureEmbeddingDataset
from src.dataset.utils import collate_fn
from src.lightning_module.lightning_embedding import LitStructureEmbedding
from src.networks.transformer_nn import TransformerEmbeddingCosine
from src.params.structure_embedding_params import StructureEmbeddingParams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--input_layer', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--hidden_layer', type=int)

    parser.add_argument('--test_every_n_steps', type=int)
    parser.add_argument('--devices', type=int)

    parser.add_argument('--class_path', required=True)
    parser.add_argument('--embedding_path', required=True)

    parser.add_argument('--metadata', type=str)

    args = parser.parse_args()

    params = StructureEmbeddingParams(args)

    cath_classes = f"{params.class_path}/cath.tsv"
    cath_embedding = f"{params.embedding_path}/cath/embedding"
    ecod_classes = f"{params.class_path}/ecod.tsv"
    ecod_embedding = f"{params.embedding_path}/ecod/embedding"

    training_set = StructureEmbeddingDataset(cath_classes, cath_embedding)
    weights = training_set.weights()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_dataloader = DataLoader(
        training_set,
        sampler=sampler,
        batch_size=params.batch_size,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    testing_set = StructureEmbeddingDataset(ecod_classes, ecod_embedding)
    test_dataloader = DataLoader(
        testing_set,
        batch_size=params.batch_size,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    model = LitStructureEmbedding(
        net=TransformerEmbeddingCosine(
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
        monitor=LitStructureEmbedding.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{'+LitStructureEmbedding.PR_AUC_METRIC_NAME+':.2f}'
    )

    trainer = L.Trainer(
        val_check_interval=params.test_every_n_steps,
        max_epochs=params.epochs,
        devices=params.devices,
        callbacks=checkpoint_callback
    )
    trainer.fit(model, train_dataloader, test_dataloader)
