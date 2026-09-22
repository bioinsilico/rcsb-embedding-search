"""Package a trained 3Di head for release, the way ``save_model_for_production.py`` does
for the embedding model - but with the architecture travelling inside the file.

The embedding model can get away with a bare ``state_dict`` because its consumer builds
``ResidueEmbeddingAggregator()`` with no arguments: the defaults *are* the production
architecture, so there is nothing to get wrong. A 3Di head is different in one specific
way: ``nhead`` cannot be recovered from the saved parameters, because
``nn.MultiheadAttention`` packs ``in_proj_weight`` as ``(3*hidden, hidden)`` whatever it
is. Load ``tf_d256_reg`` with ``nhead=4`` instead of 8 and every tensor matches, nothing
raises, and the head predicts nonsense - while the temperature, the target kernel and the
fitted lambda/K all keep describing the model that was *supposed* to be running.

So the bundle is a dict, not a bare state dict::

    {"format": "foldmatch-three-di-head/1",
     "architecture": {"class": "TransformerThreeDiHead", "hidden": 256, "nhead": 8, ...},
     "provenance":   {"checkpoint": ..., "temperature": ..., "accuracy": ...},
     "state_dict":   {...}}

``torch.load(..., weights_only=True)`` still reads it - that flag allows tensors plus
plain containers of primitives - so the consumer keeps the same safety property.

    python save_three_di_for_production.py \\
        --checkpoint /pscratch/.../epoch=18-nll=1.1824.ckpt \\
        --head-config /path/to/rcsb-embedding-search-config/config/training/head/transformer.yaml \\
        --out-file three-di-head-v1.pt --name 3di-v1 --temperature 1.080
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import torch
import yaml

from networks.three_di_head import (
    CnnThreeDiHead, LinearThreeDiHead, TransformerThreeDiHead,
)

FORMAT = "foldmatch-three-di-head/1"
HEADS = {
    "LinearThreeDiHead": LinearThreeDiHead,
    "CnnThreeDiHead": CnnThreeDiHead,
    "TransformerThreeDiHead": TransformerThreeDiHead,
}
#: Constructor arguments that define the loaded model. ``dropout`` is deliberately absent:
#: it changes nothing in eval mode and is recorded under provenance instead.
ARCHITECTURE_KEYS = ("input_features", "hidden", "nhead", "num_layers",
                     "dim_feedforward", "kernel_size", "with_ss8")


def architecture_from_config(path: str) -> dict:
    """Read the Hydra head config that trained the model - the authoritative record."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    target = config.get("_target_")
    if not target:
        raise SystemExit(f"{path}: no _target_; is this a head config?")
    name = target.rsplit(".", 1)[-1]
    if name not in HEADS:
        raise SystemExit(f"{path}: unknown head class {name!r}; have {sorted(HEADS)}")
    architecture = {"class": name}
    architecture.update({k: v for k, v in config.items() if k in ARCHITECTURE_KEYS})
    return architecture, config.get("dropout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt from training")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--head-config", required=True,
                        help="the config/training/head/*.yaml the run was trained with")
    parser.add_argument("--name", required=True, help="model name, e.g. 3di-v1")
    parser.add_argument("--temperature", type=float, required=True,
                        help="profile temperature fitted on validation (calibration)")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="architecture overrides if the run differed from the config")
    parser.add_argument("--trained-dropout", type=float, default=None,
                        help="dropout the run actually used, if it overrode the config "
                             "(tf_d256_reg used 0.3 against the config's 0.1). Recorded "
                             "for the record only - inference always runs at 0.")
    parser.add_argument("--note", default="", help="free-text provenance note")
    args = parser.parse_args()

    architecture, trained_dropout = architecture_from_config(args.head_config)
    for item in args.override:
        key, _, value = item.partition("=")
        if key not in ARCHITECTURE_KEYS:
            raise SystemExit(f"--override {key}: not an architecture key {ARCHITECTURE_KEYS}")
        architecture[key] = yaml.safe_load(value)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.")}
    if not state:
        raise SystemExit(f"{args.checkpoint}: no 'model.' parameters in the state dict")

    # Build with dropout 0: inference is eval-mode, and a mismatch here must not be able
    # to change a served prediction.
    kwargs = {k: v for k, v in architecture.items() if k != "class"}
    head = HEADS[architecture["class"]](dropout=0.0, **kwargs)
    # strict=True is the real check that the recorded architecture is the trained one -
    # every parameter must match by name AND shape.
    head.load_state_dict(state, strict=True)
    head.eval()

    provenance = {
        "name": args.name,
        "checkpoint": args.checkpoint.rsplit("/", 1)[-1],
        "head_config": args.head_config.rsplit("/", 1)[-1],
        "trained_dropout": (args.trained_dropout if args.trained_dropout is not None
                            else trained_dropout),
        "profile_temperature": args.temperature,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        # str(): torch.__version__ is a TorchVersion instance, which weights_only=True
        # refuses to unpickle. Everything stored here must be a plain primitive.
        "torch": str(torch.__version__),
        "note": args.note,
        "parameters": int(sum(p.numel() for p in head.parameters())),
    }
    bundle = {"format": FORMAT, "architecture": architecture,
              "provenance": provenance, "state_dict": head.state_dict()}
    torch.save(bundle, args.out_file)

    # Verify the artefact, not the intention: reload exactly as the consumer will, rebuild
    # from the recorded architecture alone, and require identical outputs.
    reloaded = torch.load(args.out_file, weights_only=True, map_location="cpu")
    if reloaded["format"] != FORMAT:
        raise SystemExit("format tag did not survive the round trip")
    rebuilt = HEADS[reloaded["architecture"]["class"]](
        dropout=0.0, **{k: v for k, v in reloaded["architecture"].items() if k != "class"})
    rebuilt.load_state_dict(reloaded["state_dict"], strict=True)
    rebuilt.eval()
    torch.manual_seed(0)
    probe = torch.randn(2, 37, architecture.get("input_features", 1536))
    with torch.no_grad():
        before, after = head(probe), rebuilt(probe)
    if not torch.equal(before, after):
        raise SystemExit(f"round trip changed the output by {(before - after).abs().max():.3g}")

    digest = hashlib.sha256(open(args.out_file, "rb").read()).hexdigest()
    print(json.dumps({**provenance, "architecture": architecture,
                      "file": args.out_file, "sha256": digest}, indent=2))
    print(f"\nRound trip verified: rebuilt from the recorded architecture alone, "
          f"outputs bit-identical on a random batch.", file=sys.stderr)
    print(f"Pin this sha256 (and the HF revision) wherever the model is loaded.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
