import argparse
import logging
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from dataset.utils.tools import collate_fn, load_class_pairs, load_classes

d_type = np.float32

logger = logging.getLogger(__name__)


class Step(Enum):
    TRAIN = "train"
    TEST = "test"


class TmScoreFoldCrossValidationFromEmbeddingsDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            embedding_path,
            training_class_file,
            testing_class_file,
            number_of_folds,
            fold_number,
            step=Step.TRAIN,
            score_method=None,
            weighting_method=None
    ):
        self.embedding = pd.DataFrame()
        self.class_pairs = pd.DataFrame()
        self.classes_to_filter = pd.DataFrame()
        self.step = step
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(
            tm_score_file,
            embedding_path,
            training_class_file,
            testing_class_file,
            number_of_folds,
            fold_number
        )

    def __exec(
            self,
            tm_score_file,
            embedding_path,
            training_class_file,
            testing_class_file,
            number_of_folds,
            fold_number
    ):
        self.filtered_classes(
            training_class_file,
            testing_class_file,
            number_of_folds,
            fold_number
        )
        self.load_class_pairs(tm_score_file)
        self.load_embedding(embedding_path)

    def filtered_classes(self, training_class_file, testing_class_file, number_of_folds, fold_number):
        training_classes = load_classes(training_class_file)
        logger.info(f"Loaded {len(training_classes)} classes from {training_class_file}")
        testing_classes = load_classes(testing_class_file)
        logger.info(f"Loaded {len(testing_classes)} classes from {testing_class_file}")
        delta = len(testing_classes) // number_of_folds
        beg = fold_number * delta
        end = (fold_number + 1) * delta
        logger.info(f"Fold {fold_number} of {number_of_folds}")
        if end > len(testing_classes):
            end = len(testing_classes)
        testing_classes_fold = testing_classes.iloc[beg:end]
        logger.info(f"Filtering {len(testing_classes_fold['class'].unique())} classes")
        self.classes_to_filter = training_classes[
            training_classes['class'].isin(testing_classes_fold['class'])
        ]
        logger.info(f"Filtering {len(self.classes_to_filter)} domain structures from training")

    def load_class_pairs(self, tm_score_file):
        logger.info(f"Loading pairs from file {tm_score_file}")
        self.class_pairs = load_class_pairs(tm_score_file)

        if self.step == Step.TRAIN:
            self.class_pairs = self.class_pairs[
                ~self.class_pairs['domain_i'].isin(self.classes_to_filter['domain']) &
                ~self.class_pairs['domain_j'].isin(self.classes_to_filter['domain'])
            ]
        else:
            self.class_pairs = self.class_pairs[
                self.class_pairs['domain_i'].isin(self.classes_to_filter['domain']) |
                self.class_pairs['domain_j'].isin(self.classes_to_filter['domain'])
            ]
        logger.info(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, embedding_path):
        logger.info(f"Loading embeddings from path {embedding_path}")
        self.embedding = pd.DataFrame(
            data=[
                (dom_id, os.path.join(embedding_path, f"{dom_id}.pt")) for dom_id in pd.concat([
                    self.class_pairs["domain_i"], self.class_pairs["domain_j"]
                ]).unique()
            ],
            columns=['domain', 'embedding']
        )
        logger.info(f"Total embeddings: {len(self.embedding)}")

    def weights(self):
        return self.weighting_method(self.class_pairs['score'].to_numpy())

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        domain_i = self.class_pairs.iloc[idx]['domain_i']
        row_i = self.embedding.loc[self.embedding['domain'] == domain_i]
        embedding_i = row_i['embedding'].values[0]

        domain_j = self.class_pairs.iloc[idx]['domain_j']
        row_j = self.embedding.loc[self.embedding['domain'] == domain_j]
        embedding_j = row_j['embedding'].values[0]

        return (
            torch.load(embedding_i),
            torch.load(embedding_j),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.iloc[idx]['score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    import config.logging_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--training_class_file', type=str, required=True)
    parser.add_argument('--testing_class_file', type=str, required=True)
    parser.add_argument('--number_of_folds', type=int, required=True)
    parser.add_argument('--fold_number', type=int, required=True)
    args = parser.parse_args()
    dataset = TmScoreFoldCrossValidationFromEmbeddingsDataset(
        args.tm_score_file,
        args.embedding_path,
        args.training_class_file,
        args.testing_class_file,
        args.number_of_folds,
        args.fold_number,
        step=Step.TEST,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25)
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=16,
        collate_fn=collate_fn
    )
    for (x, x_mask), (y, y_mask), z in dataloader:
        print(z)
