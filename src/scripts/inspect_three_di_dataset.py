"""Look at what the 3Di Dataset actually serves before training on it.

Prints, per split: how many domains and residues survive the joins, the label
distribution over usable residues, the share of residues each masking policy keeps,
batch shapes from the real collate function, and one domain rendered as aligned
sequence / 3Di / mask lines with its embedding statistics.

    python inspect_three_di_dataset.py --store esm3-sequence --labels esm3-sequence
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.three_di_from_embeddings_dataset import (
    STATE_INDEX, ThreeDiFromEmbeddingsDataset, collate_three_di, read_fasta,
)
from dataset.utils.three_di_labels import THREE_DI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="directory with embeddings/, ss8_logits/, domains.tsv")
    parser.add_argument("--labels", required=True, help="directory with the three label FASTA files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--use-reasons", default=".", help="exclusion characters kept in the loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--show", type=int, default=1, help="domains to print residue by residue")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sequences = read_fasta(os.path.join(args.labels, "sequences.fasta"))
    for split in args.splits:
        print(f"\n===== {split}")
        dataset = ThreeDiFromEmbeddingsDataset(
            store_path=args.store, labels_path=args.labels, split=split,
            use_reasons=args.use_reasons, with_ss8=True,
        )
        lengths = np.array(dataset.lengths())
        print(f"length  : median {int(np.median(lengths))}, p95 {int(np.percentile(lengths, 95))}, "
              f"max {lengths.max()}, total {lengths.sum():,} residues")

        # What each masking policy would keep, from the stored exclusion reasons.
        reasons = "".join(dataset.exclusion.values())
        total = len(reasons)
        for policy, label in ((".", "usable only (default)"), (".x", "+ chain breaks"), (".xt", "+ termini")):
            kept = sum(reasons.count(char) for char in policy)
            print(f"policy  : {policy:<4} keeps {kept / total:6.2%}  ({label})")

        counts = np.zeros(len(THREE_DI), dtype=np.int64)
        for domain in dataset.domains:
            three_di, exclusion = dataset.three_di[domain], dataset.exclusion[domain]
            for state, reason in zip(three_di, exclusion):
                if reason in dataset.use_reasons:
                    counts[STATE_INDEX[state]] += 1
        share = counts / counts.sum()
        order = np.argsort(-share)
        print("labels  : " + "  ".join(f"{THREE_DI[i]}:{share[i]:.1%}" for i in order))
        print(f"          most common / least common = {share[order[0]] / share[order[-1]]:.0f}x")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_three_di)
        batch = next(iter(loader))
        print(f"batch   : embedding {tuple(batch['embedding'].shape)} {batch['embedding'].dtype}, "
              f"ss8 {tuple(batch['ss8'].shape)}, label {tuple(batch['label'].shape)}, "
              f"loss_mask keeps {batch['loss_mask'].float().mean():.1%} of padded positions, "
              f"padding is {batch['padding_mask'].float().mean():.1%}")
        assert not (batch["loss_mask"] & batch["padding_mask"]).any(), "padding must never be in the loss"

        for index in range(min(args.show, len(dataset))):
            sample = dataset[index]
            three_di = dataset.three_di[sample.domain]
            exclusion = dataset.exclusion[sample.domain]
            embedding = sample.embedding
            print(f"\n  {sample.domain}  L={len(three_di)}  usable={sample.loss_mask.float().mean():.0%}  "
                  f"embedding |max| {embedding.abs().max():.0f}, mean {embedding.mean():.3f}, "
                  f"std {embedding.std():.3f}; ss8 argmax {sample.ss8.argmax(-1)[:10].tolist()}")
            width = 100
            for start in range(0, min(len(three_di), 2 * width), width):
                stop = start + width
                print(f"    {start + 1:>5} seq  {sequences[sample.domain][start:stop]}")
                print(f"          3di  {three_di[start:stop]}")
                print(f"          mask {exclusion[start:stop]}")


if __name__ == "__main__":
    main()
