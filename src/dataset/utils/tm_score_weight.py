import math
import torch


def tm_score_weights(n_intervals):
    def __tm_score_weights(scores):
        class_weights = TmScoreWeight(scores, n_intervals)
        return torch.tensor([class_weights.get_weight(s) for s in scores])
    return __tm_score_weights


def binary_score(thr):
    def __binary_score(score):
        return 1. if score > thr else 0.
    return __binary_score


def binary_weights(thr):
    def __binary_weights(scores):
        p = 1 / sum([1 for s in scores if s >= thr])
        n = 1 / sum([1 for s in scores if s < thr])
        return torch.tensor([p if s >= thr else n for s in scores])
    return __binary_weights


def fraction_score(score):
    return round(10 * score) / 10


class TmScoreWeight:

    def __init__(self, scores, n_intervals=5):
        self.weights = []
        self.n_intervals = n_intervals
        self.__compute(scores)

    def __compute(self, scores):
        h = 1 / self.n_intervals
        for idx in range(self.n_intervals):
            f = sum([1 for s in scores if idx * h <= s < (idx + 1) * h])
            self.weights.append(
                1 / f
            )

    def get_weight(self, score):
        idx = math.floor(score * self.n_intervals)
        if idx == self.n_intervals:
            idx = self.n_intervals - 1
        return self.weights[idx]