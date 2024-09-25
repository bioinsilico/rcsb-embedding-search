import logging

import torch

from dataset.utils.tools import load_class_pairs
import pandas as pd

logger = logging.getLogger(__name__)


def populate_zero_embeddings(
        tm_score_file,
        dim,
        embedding_provider
):
    class_pairs = load_class_pairs(tm_score_file)
    logger.info(f"Initializing zero embeddings for {type(embedding_provider).__name__}, dim: {dim}, source: {tm_score_file}")
    for dom_id in list(pd.concat([
        class_pairs["domain_i"], class_pairs["domain_j"]
    ]).unique()):
        embedding_provider.set(dom_id, torch.zeros(dim).tolist())
