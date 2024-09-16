import math
import torch
import numpy as np


def tm_score_weights(n_intervals):
    def __tm_score_weights(df_scores):
        class_weights = TmScoreWeight(df_scores, n_intervals)
        return torch.tensor(
            np.vectorize(class_weights.get_weight)(df_scores['score'].to_numpy())
        )
    return __tm_score_weights


def binary_score(thr):
    def __binary_score(score):
        return 1. if score > thr else 0.
    return __binary_score


def binary_weights(thr):
    def __binary_weights(df_scores):
        p = 1 / df_scores.filter(df_scores["score"] >= thr).shape[0]
        n = 1 / df_scores.filter(df_scores["score"] < thr).shape[0]
        return torch.tensor(
            np.vectorize(lambda s: p if s >= thr else n)(df_scores['score'].to_numpy())
        )
    return __binary_weights


def fraction_score(score):
    return round(10 * score) / 10


class TmScoreWeight:

    def __init__(self, df_scores, n_intervals=5):
        self.weights = np.array([])
        self.n_intervals = n_intervals
        self.__compute(df_scores)

    def __compute(self, df_scores):
        h = 1 / self.n_intervals
        for idx in range(self.n_intervals):
            f = df_scores.filter((df_scores["score"] >= idx * h) & (df_scores["score"] < (idx + 1) * h)).shape[0]
            print(f"Found {f} pairs in range {idx*h} <= s < {(idx + 1) * h}")
            self.weights = np.append(self.weights, [1/f if f > 0. else 0.])

        f = df_scores.filter(df_scores["score"] == 1.).shape[0]
        print(f"Found {f} pairs for s == 1")
        self.weights = np.append(self.weights, [1/f if f > 0. else 0.])

    def get_weight(self, score):
        idx = math.floor(score * self.n_intervals)
        return self.weights[idx]
