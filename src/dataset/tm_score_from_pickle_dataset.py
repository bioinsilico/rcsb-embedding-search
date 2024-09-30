from __future__ import annotations

import argparse
import logging

import numpy as np
from torch_geometric.loader import DataLoader
import pandas as pd

from dataset.pst.pst import load_pst_model
from dataset.utils.coords_augmenter import SelfAugmenterRandomFraction
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from networks.transformer_pst import TransformerPstEmbeddingCosine
from dataset.tm_score_from_coord_dataset import TmScoreFromCoordDataset as Dataset

d_type = np.float32
logger = logging.getLogger(__name__)


class TmScoreFromCoordDataset(Dataset):

    def load_coords(self, coords_path, ext):
        logger.info(f"Loading coords from path {coords_path}")
        self.coords = pd.read_pickle(coords_path)
        logger.info(f"Total structures: {len(self.coords)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--pdb_path', type=str, required=True)
    # parser.add_argument('--pst_model_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScoreFromCoordDataset(
        args.tm_score_file,
        args.pdb_path,
        ext="pdb",
        weighting_method=tm_score_weights(5, 0.25),
        score_method=fraction_score,
        coords_augmenter=SelfAugmenterRandomFraction(
            fraction=0.1,
            max_remove=10
        )
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler
    )
    for (g_i), (g_j), z in dataloader:
        print(z.shape)

    pst_model = load_pst_model({
        'model_path': args.pst_model_path,
        'device': 'cpu'
    })
    model = TransformerPstEmbeddingCosine(pst_model)
    for (g_i), (g_j), z in dataloader:
        z_pred = model(g_i, g_j)
        print(z_pred.shape, z.shape)
