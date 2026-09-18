"""Predict 3Di strings from a trained head, for the alignment benchmark.

Per-residue accuracy is not what this model is for: the benchmark measures how well a
*predicted* query 3Di aligns against exact target 3Di, which is the number the whole
design is aimed at.  That benchmark runs on structures, so this script writes the
predictions to a FASTA that ``structure_alignment_benchmark.py --query-3di`` reads.

It also reports per-residue accuracy, top-3 accuracy and calibration on the split, and
writes the per-residue confidence (max probability) as a second FASTA, since masking
low-confidence query positions is a lever the brief calls out.

``profiles.npz`` keeps the full distribution: one float32 (L, 20) array of log-probabilities
per domain, columns in THREE_DI order (also stored under ``__alphabet__``), for
``structure_alignment_benchmark.py --query-profile``. Log-probabilities rather than
probabilities, so the benchmark can re-temper them (softmax(log p / T) = softmax(logits / T)).

    python predict_three_di.py --checkpoint .../epoch=39.ckpt \\
        --store .../esm3-sequence --labels .../esm3-sequence --split test \\
        --out-dir .../predictions/cnn

The head architecture is read from the checkpoint when it was trained by a version
that stores it, and otherwise inferred from the parameter shapes.  ``nhead`` cannot be
inferred - it does not change any shape - so a transformer head trained with a
non-default ``nhead`` needs ``--head-arg nhead=N``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.three_di_from_embeddings_dataset import ThreeDiFromEmbeddingsDataset, collate_three_di
from dataset.utils.three_di_labels import THREE_DI
from networks.three_di_head import CnnThreeDiHead, LinearThreeDiHead, TransformerThreeDiHead

CONFIDENCE_ALPHABET = "0123456789"


def head_from_checkpoint(checkpoint: dict, overrides: dict) -> torch.nn.Module:
    """Rebuild the head: from the stored config if present, else from parameter shapes."""
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    if "head_config" in checkpoint:
        from hydra.utils import instantiate
        head = instantiate({**checkpoint["head_config"], **overrides})
        head.load_state_dict(state)
        return head.eval()

    in_features = state["norm.weight"].shape[0]
    hidden = state["classifier.weight"].shape[1]
    kwargs = dict(hidden=hidden, dropout=0.0)
    if "transformer.layers.0.self_attn.in_proj_weight" in state:
        cls = TransformerThreeDiHead
        kwargs["num_layers"] = 1 + max(
            int(key.split(".")[2]) for key in state if key.startswith("transformer.layers.")
        )
        kwargs["dim_feedforward"] = state["transformer.layers.0.linear1.weight"].shape[0]
        kwargs["nhead"] = 8
    elif "layers.0.0.weight" in state:
        cls = CnnThreeDiHead
        kwargs["kernel_size"] = state["layers.0.0.weight"].shape[-1]
        kwargs["num_layers"] = 1 + max(int(key.split(".")[1]) for key in state if key.startswith("layers."))
    else:
        cls = LinearThreeDiHead
    with_ss8 = in_features > 1536
    kwargs.update(input_features=in_features - (11 if with_ss8 else 0), with_ss8=with_ss8)
    kwargs.update({k: v for k, v in overrides.items() if k in cls.__init__.__code__.co_varnames})
    head = cls(**kwargs)
    head.load_state_dict(state)
    return head.eval()


def write_fasta(path: str, records: dict[str, str], header: dict[str, str]) -> None:
    with open(path, "w") as handle:
        for name, sequence in records.items():
            handle.write(f">{name} {header.get(name, '')}\n{sequence}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--head-arg", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override a head argument that cannot be inferred, e.g. nhead=8")
    args = parser.parse_args()

    overrides = {}
    for item in args.head_arg:
        key, value = item.split("=", 1)
        overrides[key] = int(value) if value.isdigit() else value

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = head_from_checkpoint(checkpoint, overrides).to(args.device)
    print(f"{type(head).__name__}, {sum(p.numel() for p in head.parameters()):,} parameters, "
          f"with_ss8={head.with_ss8}, from {args.checkpoint}")

    dataset = ThreeDiFromEmbeddingsDataset(
        store_path=args.store, labels_path=args.labels, split=args.split, with_ss8=head.with_ss8,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_three_di)

    predictions, confidences, profiles = {}, {}, {}
    n = correct = top3 = 0
    confidence_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = head(
                batch["embedding"].to(args.device),
                padding_mask=batch["padding_mask"].to(args.device),
                ss8=batch["ss8"].to(args.device) if head.with_ss8 else None,
            ).float().cpu()
            probability = F.softmax(logits, dim=-1)
            log_probability = F.log_softmax(logits, dim=-1)
            best = probability.max(dim=-1)
            for i, domain in enumerate(batch["domain"]):
                length = int((~batch["padding_mask"][i]).sum())
                states = best.indices[i, :length].numpy()
                scores = best.values[i, :length].numpy()
                predictions[domain] = "".join(THREE_DI[s] for s in states)
                # float32: float16 can flip the argmax on exact ties (~1e-4 of residues).
                profiles[domain] = log_probability[i, :length].numpy().astype(np.float32)
                # 0-9 buckets: readable next to the 3Di string and enough to threshold on.
                confidences[domain] = "".join(
                    CONFIDENCE_ALPHABET[min(int(s * 10), 9)] for s in scores
                )
                # Accuracy is reported on the loss-masked residues only, as in training.
                mask = batch["loss_mask"][i, :length].numpy()
                label = batch["label"][i, :length].numpy()
                n += int(mask.sum())
                correct += int((states[mask] == label[mask]).sum())
                top3 += int((logits[i, :length].topk(3, dim=-1).indices.numpy()[mask]
                             == label[mask, None]).any(axis=-1).sum())
                confidence_sum += float(scores[mask].sum())

    os.makedirs(args.out_dir, exist_ok=True)
    header = {d: f"split={args.split} length={len(s)}" for d, s in predictions.items()}
    write_fasta(os.path.join(args.out_dir, "predicted_three_di.fasta"), predictions, header)
    write_fasta(os.path.join(args.out_dir, "confidence.fasta"), confidences, header)
    np.savez_compressed(os.path.join(args.out_dir, "profiles.npz"),
                        __alphabet__=np.array(list(THREE_DI)), **profiles)
    summary = dict(
        checkpoint=os.path.abspath(args.checkpoint), split=args.split, head=type(head).__name__,
        domains=len(predictions), residues_scored=n,
        accuracy=correct / max(n, 1), top3_accuracy=top3 / max(n, 1),
        mean_confidence=confidence_sum / max(n, 1),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{len(predictions):,} domains, {n:,} scored residues: accuracy {summary['accuracy']:.4f}, "
          f"top-3 {summary['top3_accuracy']:.4f}, mean confidence {summary['mean_confidence']:.3f}")
    print(f"wrote {args.out_dir}/predicted_three_di.fasta")


if __name__ == "__main__":
    main()
