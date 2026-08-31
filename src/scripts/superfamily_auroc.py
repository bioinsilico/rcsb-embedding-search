"""Score an embedding checkpoint by how well cosine similarity separates CATH superfamilies.

Training metrics cannot compare models trained on different targets: a spearman
of 0.835 against *coverage* and 0.80 against *matched residues* measure different
tasks.  CATH superfamily membership is a homology label neither target was
trained on, so AUROC over "same superfamily vs not" is a common yardstick — and
it is much closer to the question a search system actually answers.

Run it on held-out domains (the same file passed to ``holdout_domains_file``) so
the score reflects transfer to unseen folds.

Two read-outs are reported, because a segment model can represent a domain in
more than one way:

* ``mean``  - average of the domain's window embeddings, renormalised.  Cheap,
  one vector per domain, and what an embedding index would usually store.
* ``max``   - the largest cosine over all window pairs of the two domains.
  Closer to local search: two domains match if any part of them does.
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict

import numpy as np
import torch
from torcheval.metrics.functional import binary_auprc, binary_auroc

from dataset.utils.cath_superfamily import (load_cath_levels, load_superfamilies,
                                            parse_cath_domain_id)
from networks.sequence_autoencoder import SequenceAutoencoder, tokenize_sequence

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

_RESIDUES = 'ACDEFGHIKLMNPQRSTVWYXBZUO'


def infer_architecture(state: dict, nhead: int) -> SequenceAutoencoder:
    """Rebuild the network from the checkpoint's own tensor shapes.

    Only ``nhead`` cannot be recovered: attention packs its projections as
    ``(3*d_model, d_model)`` whatever the head count, so it must be supplied.
    """
    vocab, d_model = state['aa_embedding.weight'].shape
    dim_ff = state['encoder.layers.0.linear1.weight'].shape[0]
    n_enc = 1 + max(int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.'))
    n_dec = 1 + max(int(k.split('.')[2]) for k in state if k.startswith('decoder.layers.'))
    max_len = state['decoder_queries.weight'].shape[0]
    res_blocks = len({k.split('.')[1] for k in state
                      if k.startswith('to_latent.block')})
    latent_key = 'to_latent.linear.weight' if res_blocks else 'to_latent.1.weight'
    latent_dim = state[latent_key].shape[0]
    logger.info(
        f"  architecture from checkpoint: d_model={d_model} latent={latent_dim} "
        f"enc={n_enc} dec={n_dec} ff={dim_ff} max_len={max_len} "
        f"res_blocks={res_blocks} vocab={vocab} (nhead={nhead} supplied)"
    )
    return SequenceAutoencoder(
        d_model=d_model, latent_dim=latent_dim, nhead=nhead,
        num_encoder_layers=n_enc, num_decoder_layers=n_dec,
        dim_feedforward=dim_ff, max_seq_len=max_len, vocab_size=vocab,
        res_block_layers=res_blocks,
    )


def load_model(path: str, nhead: int, device: torch.device,
               randomize: bool = False) -> SequenceAutoencoder:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    # Lightning stores the network under `self.model`.
    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
    model = infer_architecture(state, nhead)
    if randomize:
        # Same architecture, untrained.  Essential context: random embeddings
        # still track amino-acid composition, which correlates with family, so
        # an untrained network scores far above 0.5 and the trained model's
        # AUROC only means something relative to this.
        logger.warning("  --randomize: using UNTRAINED weights as a baseline")
        return model.to(device).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(f"  state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    return model.to(device).eval()


def read_domains(fasta: str, keep: set[str] | None, segment_length: int):
    """{domain_id: sequence} for records long enough, optionally restricted."""
    out, header, parts = {}, None, []

    def flush():
        if header is None:
            return
        domain = parse_cath_domain_id(header)
        if keep is not None and domain not in keep and header not in keep:
            return
        seq = ''.join(parts).upper()
        seq = ''.join(c if c in _RESIDUES else 'X' for c in seq)
        if len(seq) >= segment_length:
            out[domain] = seq

    with open(fasta) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                flush()
                header, parts = line[1:].split()[0], []
            elif header is not None:
                parts.append(line)
        flush()
    return out


def windows_of(seq: str, length: int, count: int) -> list[str]:
    """Up to ``count`` windows spread evenly across the sequence."""
    span = len(seq) - length
    if span <= 0:
        return [seq[:length]]
    n = min(count, span + 1)
    starts = np.unique(np.linspace(0, span, n).round().astype(int))
    return [seq[s:s + length] for s in starts]


@torch.no_grad()
def encode(model, segments: list[str], device, batch_size: int) -> torch.Tensor:
    """L2-normalised embeddings; encoder only, the decoder is never needed here."""
    out = []
    for i in range(0, len(segments), batch_size):
        chunk = segments[i:i + batch_size]
        tokens = torch.tensor([tokenize_sequence(s) for s in chunk],
                              dtype=torch.long, device=device)
        out.append(model.encode(tokens).float().cpu())
        if (i // batch_size) % 50 == 0:
            logger.info(f"    encoded {min(i + batch_size, len(segments)):,}/{len(segments):,}")
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--domain-list', required=True)
    ap.add_argument('--domains', default=None,
                    help='restrict to these domain ids (the holdout file)')
    ap.add_argument('--segment-length', type=int, default=50)
    ap.add_argument('--windows-per-domain', type=int, default=8)
    ap.add_argument('--max-domains', type=int, default=4000)
    ap.add_argument('--per-family', type=int, default=10,
                    help='cap per superfamily, so the budget spans many folds')
    ap.add_argument('--pairs', type=int, default=100000,
                    help='positive pairs, and an equal number of negatives')
    ap.add_argument('--nhead', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--negative-level', default='any',
                    choices=['any', 'class', 'architecture', 'topology'],
                    help='negatives share CATH levels down to this depth; \'topology\' is the hardest')
    ap.add_argument('--randomize', action='store_true',
                    help='untrained weights, to establish the baseline AUROC')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        ('cuda' if torch.cuda.is_available() else
         'mps' if torch.backends.mps.is_available() else 'cpu')
        if args.device == 'auto' else args.device)
    rng = random.Random(args.seed)

    keep = None
    if args.domains:
        with open(args.domains) as handle:
            keep = {line.split()[0] for line in handle if line.strip()}
        logger.info(f"restricting to {len(keep):,} domains from {args.domains}")

    superfamily = load_superfamilies(args.domain_list)
    sequences = read_domains(args.fasta, keep, args.segment_length)
    logger.info(f"{len(sequences):,} domains long enough to evaluate")

    families = defaultdict(list)
    for domain in sequences:
        if domain in superfamily:
            families[superfamily[domain]].append(domain)
    families = {f: m for f, m in families.items() if len(m) >= 2}

    # Spread the domain budget across many families.  Taking the largest first
    # would fill it from a single superfamily -- CATH's are wildly uneven -- and
    # then no cross-family pair exists to sample as a negative.
    order = list(families)
    rng.shuffle(order)
    chosen: list[str] = []
    for family in order:
        if len(chosen) >= args.max_domains:
            break
        members = families[family]
        take = members if len(members) <= args.per_family else rng.sample(members, args.per_family)
        chosen.extend(take)
    kept = set(chosen)
    families = {f: [m for m in mem if m in kept] for f, mem in families.items()}
    families = {f: m for f, m in families.items() if len(m) >= 2}
    logger.info(f"evaluating {len(chosen):,} domains across {len(families):,} superfamilies")

    segments, spans = [], {}
    for domain in chosen:
        w = windows_of(sequences[domain], args.segment_length, args.windows_per_domain)
        spans[domain] = (len(segments), len(segments) + len(w))
        segments.extend(w)
    logger.info(f"encoding {len(segments):,} windows on {device} ...")
    emb = encode(load_model(args.checkpoint, args.nhead, device, args.randomize),
                 segments, device, args.batch_size)

    mean_vec = {}
    for domain, (lo, hi) in spans.items():
        v = emb[lo:hi].mean(0)
        mean_vec[domain] = v / v.norm().clamp(min=1e-9)

    fam_list = list(families)
    if len(fam_list) < 2:
        raise SystemExit(
            f"only {len(fam_list)} superfamily survived the filters -- no cross-family "
            f"pair exists. Raise --max-domains or lower --per-family."
        )
    # Draw two distinct families first, then a member of each: negatives by
    # construction, rather than by rejecting same-family draws (which never
    # terminates if the budget lands inside one family).
    pos = []
    for _ in range(args.pairs):
        a, b = rng.sample(families[rng.choice(fam_list)], 2)
        pos.append((a, b))
    neg = []
    if args.negative_level == 'any':
        for _ in range(args.pairs):
            fa, fb = rng.sample(fam_list, 2)
            neg.append((rng.choice(families[fa]), rng.choice(families[fb])))
    else:
        # Hard negatives: same fold down to the chosen level, different
        # homologous superfamily.  Composition no longer separates them.
        depth = {'class': 1, 'architecture': 2, 'topology': 3}[args.negative_level]
        levels = load_cath_levels(args.domain_list)
        siblings = defaultdict(set)
        for family, members in families.items():
            key = levels[members[0]][:depth]
            siblings[key].add(family)
        usable = [k for k, v in siblings.items() if len(v) >= 2]
        if not usable:
            raise SystemExit(
                f"no '{args.negative_level}' group holds two different superfamilies "
                f"among the sampled domains -- raise --max-domains or use "
                f"--negative-level any"
            )
        logger.info(f"hard negatives from {len(usable):,} '{args.negative_level}' groups")
        for _ in range(args.pairs):
            fa, fb = rng.sample(sorted(siblings[rng.choice(usable)]), 2)
            neg.append((rng.choice(families[fa]), rng.choice(families[fb])))

    def score(pairs):
        m = torch.stack([mean_vec[a] for a, _ in pairs])
        n = torch.stack([mean_vec[b] for _, b in pairs])
        mean_cos = (m * n).sum(1)
        mx = []
        for a, b in pairs:
            la, ha = spans[a]; lb, hb = spans[b]
            mx.append(float((emb[la:ha] @ emb[lb:hb].T).max()))
        return mean_cos, torch.tensor(mx)

    logger.info("scoring pairs ...")
    pm, px = score(pos)
    nm, nx = score(neg)
    label = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    print("\n" + "=" * 72)
    print(f"  {len(pos):,} same-superfamily pairs vs {len(neg):,} different-superfamily pairs")
    print(f"  {'readout':10s}{'AUROC':>10s}{'AUPRC':>10s}{'mean cos +':>13s}{'mean cos -':>13s}")
    for name, p, n in (('mean', pm, nm), ('max', px, nx)):
        s = torch.cat([p, n])
        print(f"  {name:10s}{float(binary_auroc(s, label)):10.4f}"
              f"{float(binary_auprc(s, label)):10.4f}{float(p.mean()):13.4f}{float(n.mean()):13.4f}")
    print("\n  AUPRC is at a 1:1 positive:negative ratio; real search is far more skewed.")


if __name__ == '__main__':
    main()
