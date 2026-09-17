"""Training samples for the sequence -> 3Di head: cached ESM3 embeddings + 3Di labels.

One sample is one domain: the per-residue ESM3 embedding from the packed store built
by ``esm3_sequence_embeddings.py``, the 3Di state of each residue, and a boolean mask
saying which of those states may enter the loss.

Both sides are precomputed, so ``__getitem__`` does no file I/O: embeddings come from
a memory-mapped blob and labels from strings held in RAM (8 MB for the whole corpus).

The embedding store is keyed by the SHA-1 of the sequence, so identical domains share
one entry.  At construction every domain's stored sequence is checked against the
sequence its labels were derived from, and mismatches are dropped rather than trained
on: that is the one error that would silently shift labels against embeddings.

``use_reasons`` decides the masking policy from the precomputed exclusion reasons
(``.`` usable, ``t`` terminus, ``b`` backbone, ``x`` chain break), so it can change
without rebuilding anything.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils.packed_embeddings import PackedEmbeddingStore
from dataset.utils.three_di_labels import THREE_DI

logger = logging.getLogger(__name__)

N_STATES = len(THREE_DI)
#: 3Di letter -> class index, matching ``THREE_DI`` and therefore the substitution matrix.
STATE_INDEX = {state: index for index, state in enumerate(THREE_DI)}

LABEL_FILES = {"sequence": "sequences.fasta", "three_di": "three_di.fasta", "exclusion": "exclusion.fasta"}

#: Byte value -> state index, so a label string converts with one numpy lookup instead of
#: a Python loop over millions of residues.
_STATE_TABLE = np.zeros(256, dtype=np.int64)
for _index, _state in enumerate(THREE_DI):
    _STATE_TABLE[ord(_state)] = _index


def _as_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8)


def _usable(exclusion: str, reasons: set[str]) -> np.ndarray:
    table = np.zeros(256, dtype=bool)
    for reason in reasons:
        table[ord(reason)] = True
    return table[_as_bytes(exclusion)]


def read_fasta(path: str) -> dict[str, str]:
    """``{id: sequence}``; the id is the first whitespace-delimited token of the header."""
    records, name, chunks = {}, None, []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    return records


@dataclass
class ThreeDiSample:
    domain: str
    embedding: torch.Tensor   # (L, 1536) float32
    ss8: torch.Tensor | None  # (L, 11) float32
    label: torch.Tensor       # (L,) int64, index into THREE_DI
    loss_mask: torch.Tensor   # (L,) bool, True where the label may enter the loss


class ThreeDiFromEmbeddingsDataset(Dataset):
    """Domains of one split as (embedding, ss8, 3Di label, loss mask)."""

    def __init__(
            self,
            store_path: str,
            labels_path: str,
            split: str | None = None,
            use_reasons: str = ".",
            with_ss8: bool = False,
            min_length: int = 16,
            max_length: int | None = None,
            min_usable: int = 1,
            dtype: torch.dtype = torch.float32,
            embeddings_dir: str | None = None,
            ss8_dir: str | None = None,
    ):
        super().__init__()
        self.with_ss8 = with_ss8
        self.use_reasons = set(use_reasons)
        self.dtype = dtype

        # The stores may live elsewhere than store_path (e.g. staged to node-local disk),
        # while domains.tsv is still read from store_path.
        self.embeddings = PackedEmbeddingStore(embeddings_dir or os.path.join(store_path, "embeddings"))
        self.ss8_store = (PackedEmbeddingStore(ss8_dir or os.path.join(store_path, "ss8_logits"))
                          if with_ss8 else None)

        labels = {field: read_fasta(os.path.join(labels_path, name)) for field, name in LABEL_FILES.items()}
        keys = self._sequence_keys(os.path.join(store_path, "domains.tsv"), split)

        self.domains: list[str] = []
        self.keys: dict[str, str] = {}
        self.three_di: dict[str, str] = {}
        self.exclusion: dict[str, str] = {}
        dropped = {"no labels": 0, "no embedding": 0, "sequence mismatch": 0, "length": 0, "no usable residue": 0}
        for domain, (key, stored_sequence) in keys.items():
            if domain not in labels["three_di"]:
                dropped["no labels"] += 1
                continue
            if key not in self.embeddings or (self.ss8_store is not None and key not in self.ss8_store):
                dropped["no embedding"] += 1
                continue
            # The embedding was computed from the store's sequence; the labels come from the
            # structure. If they disagree, residue i of the embedding is not residue i of the label.
            if labels["sequence"][domain] != stored_sequence:
                dropped["sequence mismatch"] += 1
                continue
            three_di = labels["three_di"][domain]
            exclusion = labels["exclusion"][domain]
            if len(three_di) < min_length or (max_length is not None and len(three_di) > max_length):
                dropped["length"] += 1
                continue
            if int(_usable(exclusion, self.use_reasons).sum()) < min_usable:
                dropped["no usable residue"] += 1
                continue
            self.domains.append(domain)
            self.keys[domain] = key
            self.three_di[domain] = three_di
            self.exclusion[domain] = exclusion

        reported = ", ".join(f"{count:,} {reason}" for reason, count in dropped.items() if count)
        logger.info(
            f"{split or 'all'}: {len(self.domains):,} domains, "
            f"{sum(len(s) for s in self.three_di.values()):,} residues"
            + (f"; dropped {reported}" if reported else "")
        )
        if not self.domains:
            raise ValueError(f"No usable domains for split {split!r} in {store_path}")

    @staticmethod
    def _sequence_keys(path: str, split: str | None) -> dict[str, tuple[str, str]]:
        """``domains.tsv`` -> ``{domain: (sequence_key, sequence)}`` for one split."""
        keys = {}
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                if split is None or row["split"] == split:
                    keys[row["domain_id"]] = (row["sequence_key"], row["sequence"])
        return keys

    def __len__(self) -> int:
        return len(self.domains)

    def lengths(self) -> list[int]:
        """Residue count per sample, for length-bucketed batching."""
        return [len(self.three_di[domain]) for domain in self.domains]

    def label_counts(self) -> torch.Tensor:
        """Residues per 3Di state over the usable positions - the loss's class weights.

        Vectorized because this runs on every rank before training starts; a Python
        loop over the 6.9 M training residues takes ~14 s of startup per rank.
        """
        counts = np.zeros(N_STATES, dtype=np.int64)
        for domain in self.domains:
            states = _STATE_TABLE[_as_bytes(self.three_di[domain])]
            counts += np.bincount(
                states[_usable(self.exclusion[domain], self.use_reasons)], minlength=N_STATES
            )
        return torch.from_numpy(counts)

    def __getitem__(self, index: int) -> ThreeDiSample:
        domain = self.domains[index]
        key = self.keys[domain]
        three_di = self.three_di[domain]
        exclusion = self.exclusion[domain]

        embedding = self.embeddings.get(key).to(self.dtype)
        if embedding.size(0) != len(three_di):
            raise RuntimeError(
                f"{domain}: embedding has {embedding.size(0)} rows for {len(three_di)} labelled residues"
            )
        ss8 = self.ss8_store.get(key).to(self.dtype) if self.ss8_store is not None else None
        return ThreeDiSample(
            domain=domain,
            embedding=embedding,
            ss8=ss8,
            label=torch.from_numpy(_STATE_TABLE[_as_bytes(three_di)]),
            loss_mask=torch.from_numpy(_usable(exclusion, self.use_reasons)),
        )


def collate_three_di(samples: list[ThreeDiSample]) -> dict:
    """Pad a batch to its longest domain.

    ``padding_mask`` follows this repo's convention (True where padded, as
    ``nn.Transformer`` wants it); ``loss_mask`` is True only on residues that carry a
    usable label, so padded positions are always False there.
    """
    batch, width = len(samples), max(sample.embedding.size(0) for sample in samples)
    dim = samples[0].embedding.size(1)
    dtype = samples[0].embedding.dtype

    embedding = torch.zeros((batch, width, dim), dtype=dtype)
    label = torch.zeros((batch, width), dtype=torch.int64)
    loss_mask = torch.zeros((batch, width), dtype=torch.bool)
    padding_mask = torch.ones((batch, width), dtype=torch.bool)
    ss8 = (torch.zeros((batch, width, samples[0].ss8.size(1)), dtype=dtype)
           if samples[0].ss8 is not None else None)

    for i, sample in enumerate(samples):
        length = sample.embedding.size(0)
        embedding[i, :length] = sample.embedding
        label[i, :length] = sample.label
        loss_mask[i, :length] = sample.loss_mask
        padding_mask[i, :length] = False
        if ss8 is not None:
            ss8[i, :length] = sample.ss8

    return {
        "domain": [sample.domain for sample in samples],
        "embedding": embedding,
        "ss8": ss8,
        "label": label,
        "loss_mask": loss_mask,
        "padding_mask": padding_mask,
    }
