"""Confusion-aware loss for 3Di prediction.

The benchmark behind this model says the *kind* of mistake matters more than the
error rate: at the same 65% per-residue accuracy, errors drawn in proportion to the
3Di substitution matrix keep alignment F1 at 0.609, while uniform-random errors drop
it to 0.394.  Plain cross-entropy treats all 20 wrong states alike, so it optimizes
the wrong thing.

Three configurable terms, all off by default so the default is plain cross-entropy:

* ``similarity_temperature`` replaces the one-hot target with
  ``softmax(M[true] / T)``, the substitution matrix's own view of which states are
  interchangeable.  Small T is nearly one-hot, large T spreads mass over geometric
  neighbours.  This is label smoothing shaped by geometry rather than uniform.
* ``expected_cost_weight`` adds ``sum_k p(k) * cost(true, k)`` with
  ``cost(true, k) = M[true, true] - M[true, k] >= 0``.  This is exactly what the
  downstream aligner computes: it scores a query position by
  ``sum_k p(k) * M[k, s]``, so the loss penalizes probability mass in proportion to
  how much it will distort that score.
* ``class_weight_power`` re-weights residues by ``(1 / frequency) ** power`` to
  counter the 19x imbalance between the most and least common state (``v`` 20.6%,
  ``m`` 1.1%).  1.0 is full inverse-frequency weighting, 0.5 a common compromise.

``forward`` takes the loss mask and averages over unmasked residues only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset.utils.three_di_labels import THREE_DI


def substitution_matrix() -> torch.Tensor:
    """Foldseek's 3Di substitution matrix as a (20, 20) tensor in ``THREE_DI`` order."""
    from biotite.sequence.align import SubstitutionMatrix

    matrix = SubstitutionMatrix.std_3di_matrix()
    alphabet = matrix.get_alphabet1()
    order = [alphabet.encode(state) for state in THREE_DI]
    return torch.tensor(matrix.score_matrix()[np.ix_(order, order)], dtype=torch.float32)


class ThreeDiLoss(nn.Module):
    """Cross-entropy over 3Di states, optionally shaped by the substitution matrix."""

    def __init__(
            self,
            similarity_temperature: float | None = None,
            expected_cost_weight: float = 0.0,
            class_weight_power: float = 0.0,
            label_smoothing: float = 0.0,
    ):
        super().__init__()
        if similarity_temperature is not None and similarity_temperature <= 0:
            raise ValueError(f"similarity_temperature must be positive, got {similarity_temperature}")
        self.similarity_temperature = similarity_temperature
        self.expected_cost_weight = expected_cost_weight
        self.class_weight_power = class_weight_power
        self.label_smoothing = label_smoothing

        # The matrix and everything derived from it are rebuilt in __init__, so they are
        # not persisted: otherwise a checkpoint trained with one loss configuration fails
        # a strict state_dict load under another (soft_target present in one, absent in
        # the other). class_weight *is* persisted - it comes from the data, not the code.
        matrix = substitution_matrix()
        self.register_buffer("matrix", matrix, persistent=False)
        # cost(true, k): 0 for the true state, larger the more distant k is from it.
        # Scaled by the mean diagonal so the term is O(1) and comparable across matrices.
        cost = matrix.diagonal().unsqueeze(1) - matrix
        self.register_buffer("cost", cost / matrix.diagonal().mean(), persistent=False)
        self.register_buffer(
            "soft_target",
            F.softmax(matrix / similarity_temperature, dim=-1) if similarity_temperature else None,
            persistent=False,
        )
        self.register_buffer("class_weight", torch.ones(len(THREE_DI)))

    def set_class_counts(self, counts: torch.Tensor) -> None:
        """Set inverse-frequency class weights from label counts (normalized to mean 1)."""
        if self.class_weight_power == 0:
            return
        frequency = counts.to(torch.float64) / counts.sum()
        weight = (1.0 / frequency.clamp_min(1e-12)) ** self.class_weight_power
        self.class_weight = (weight / weight.mean()).to(torch.float32).to(self.class_weight.device)

    def forward(self, logits: torch.Tensor, label: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """``logits`` (B, L, 20), ``label`` (B, L), ``loss_mask`` (B, L) True where used."""
        logits = logits[loss_mask]
        label = label[loss_mask]
        if label.numel() == 0:
            return logits.sum() * 0.0     # keeps the graph alive on an all-masked batch

        logits = logits.float()
        log_probability = F.log_softmax(logits, dim=-1)
        if self.soft_target is not None:
            target = self.soft_target[label]
            if self.label_smoothing:
                target = (1 - self.label_smoothing) * target + self.label_smoothing / target.size(-1)
            per_residue = -(target * log_probability).sum(dim=-1)
        else:
            per_residue = F.cross_entropy(
                logits, label, reduction="none", label_smoothing=self.label_smoothing
            )

        if self.expected_cost_weight:
            probability = log_probability.exp()
            per_residue = per_residue + self.expected_cost_weight * (
                probability * self.cost[label]
            ).sum(dim=-1)

        if self.class_weight_power:
            weight = self.class_weight[label]
            return (per_residue * weight).sum() / weight.sum()
        return per_residue.mean()

    @torch.no_grad()
    def expected_substitution_score(self, logits: torch.Tensor, label: torch.Tensor,
                                    loss_mask: torch.Tensor) -> torch.Tensor:
        """Mean ``sum_k p(k) * M[k, true]`` - the score the aligner will actually compute.

        Its ceiling is the mean diagonal of the matrix (a confident, correct prediction),
        so it says how much alignment signal a soft profile retains, which per-residue
        accuracy cannot.
        """
        logits, label = logits[loss_mask], label[loss_mask]
        if label.numel() == 0:
            return torch.zeros((), device=logits.device)
        probability = F.softmax(logits.float(), dim=-1)
        return (probability * self.matrix[label]).sum(dim=-1).mean()
