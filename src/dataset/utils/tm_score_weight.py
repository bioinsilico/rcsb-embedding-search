import logging
import math
import torch
import numpy as np

logger = logging.getLogger(__name__)


def tm_score_weights(n_intervals, identity_scale_factor):
    def __tm_score_weights(np_scores):
        class_weights = TmScoreWeight(np_scores, n_intervals, identity_scale_factor)
        return torch.tensor(
            np.vectorize(class_weights.get_weight)(np_scores)
        )
    return __tm_score_weights


def binary_score(thr):
    def __binary_score(score):
        return 1. if score > thr else 0.

    return __binary_score


def binary_weights(thr):
    def __binary_weights(np_scores):
        p = 1 / np.sum(np_scores >= thr)
        n = 1 / np.sum(np_scores < thr)
        return torch.tensor(
            np.vectorize(lambda s: p if s >= thr else n)(np_scores)
        )

    return __binary_weights


def fraction_score(score):
    return round(10 * score) / 10


def fraction_score_of(f=10):
    def __fraction_score(score):
        return round(f * score) / f
    return __fraction_score


class TmScoreWeight:

    def __init__(
            self,
            np_scores,
            n_intervals=5,
            identity_scale_factor=1.
    ):
        self.weights = np.array([])
        self.n_intervals = n_intervals
        self.identity_scale_factor = identity_scale_factor
        self.__compute(np_scores)

    def __compute(self, np_scores):
        h = 1 / self.n_intervals
        for idx in range(self.n_intervals):
            f = np.sum((np_scores >= idx * h) & (np_scores < (idx + 1) * h))
            logger.info(f"Found {f} pairs in range {idx * h} <= s < {(idx + 1) * h}")
            self.weights = np.append(self.weights, [1 / f if f > 0. else 0.])
        f = self.identity_scale_factor * np.sum(np_scores == 1.)
        logger.info(f"Found {f} pairs for s == 1")
        self.weights = np.append(self.weights, [1 / f if f > 0. else 0.])

    def get_weight(self, score):
        idx = math.floor(score * self.n_intervals)
        return self.weights[idx]
