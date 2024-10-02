import logging

import torch

from dataset.utils.tools import get_unique_pairs

logger = logging.getLogger(__name__)


def populate_zero_embeddings(
        tm_score_file,
        dim,
        embedding_provider
):
    embedding_provider.set_many([(dom_id, torch.zeros(dim).tolist()) for dom_id in get_unique_pairs(tm_score_file)])
