"""Tests for the stage-1 retraining code (model options, padding fix, packed pair data, sampler, schedule).

    PYTHONPATH=src python -m pytest -q src/tests/test_stage1_retrain.py
"""
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset.packed_tm_pairs_dataset import (BinBalancedPairBatchSampler, PackedTmPairDataset,
                                             ShardedSequentialBatchSampler, cath_twins, tm_bins)
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.packed_embeddings import build_packed_store
from dataset.utils.tm_score_weight import tm_score_weights
from lightning_module.training.embedding_training import LitEmbeddingTrainingV2
from networks.layers import ResBlock
from networks.transformer_nn import TransformerEmbeddingCosine

SMALL = dict(input_features=32, dim_feedforward=64, hidden_layer=32, nhead=4, num_layers=2, res_block_layers=2)
PROD_PT = os.path.expanduser("~/.cache/huggingface/hub/models--rcsb--rcsb-embedding-model/snapshots/"
                             "410606e40b1bb7968ce318c41009355c3ac32503/rcsb-embedding-model.pt")


class OldTransformerEmbeddingCosine(nn.Module):
    """The architecture before the stage-1 changes (sum of unmasked rows), for comparison."""
    dropout = 0.1

    def __init__(self, input_features, dim_feedforward, hidden_layer, nhead, num_layers, res_block_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=self.dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        blocks = OrderedDict((f"block{i}", ResBlock(input_features, hidden_layer, self.dropout))
                             for i in range(res_block_layers))
        blocks.update([("dropout", nn.Dropout(p=self.dropout)), ("linear", nn.Linear(input_features, hidden_layer)),
                       ("activation", nn.ReLU())])
        self.embedding = nn.Sequential(blocks)

    def embedding_pooling(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))


def no_dropout(model):
    """Train mode (the slow, unfused path) without randomness: nn.Dropout and attention dropout off."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
    return model


def padded(lengths, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(n, dim, generator=g) * 3 for n in lengths]
    x = torch.zeros(len(xs), max(lengths), dim)
    mask = torch.ones(len(xs), max(lengths), dtype=torch.bool)
    for b, t in enumerate(xs):
        x[b, :len(t)] = t
        mask[b, :len(t)] = False
    return xs, x, mask


def test_default_architecture_matches_the_old_one():
    torch.manual_seed(0)
    new = TransformerEmbeddingCosine(**SMALL).eval()
    old = OldTransformerEmbeddingCosine(**SMALL).eval()
    assert list(new.state_dict()) == list(old.state_dict())
    old.load_state_dict(new.state_dict())
    xs, _, _ = padded([7], 32)
    with torch.no_grad():
        a = new.embedding_pooling(xs[0][None], torch.zeros(1, 7, dtype=torch.bool))
        b = old.embedding_pooling(xs[0][None], torch.zeros(1, 7, dtype=torch.bool))
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("opts", [
    {},
    dict(output_dim=8, output_activation="none", pooling="mean_ln", norm_first=True, input_norm=True),
])
def test_padding_does_not_change_the_embedding_in_train_mode(opts):
    torch.manual_seed(0)
    model = no_dropout(TransformerEmbeddingCosine(**SMALL, **opts)).train()     # slow path, as in training
    xs, x, mask = padded([5, 11, 3], 32)
    with torch.no_grad():
        batched = model.embedding_pooling(x, mask)
        single = torch.cat([model.embedding_pooling(t[None], torch.zeros(1, len(t), dtype=torch.bool)) for t in xs])
    assert torch.allclose(batched, single, atol=1e-4)


def test_old_architecture_was_padding_dependent_in_train_mode():
    torch.manual_seed(0)
    old = no_dropout(OldTransformerEmbeddingCosine(**SMALL)).train()
    xs, x, mask = padded([5, 11, 3], 32)
    with torch.no_grad():
        batched = old.embedding_pooling(x, mask)
        single = torch.cat([old.embedding_pooling(t[None], torch.zeros(1, len(t), dtype=torch.bool)) for t in xs])
    assert not torch.allclose(batched, single, atol=1e-3)


def test_new_options_shapes_and_input_stats(tmp_path):
    stats = tmp_path / "stats.npz"
    np.savez(stats, mean=np.full(32, 2.0, np.float32), std=np.full(32, 4.0, np.float32))
    model = TransformerEmbeddingCosine(**SMALL, output_dim=8, output_activation="none", pooling="mean_ln",
                                       norm_first=True, input_norm=True, input_stats=str(stats)).eval()
    assert torch.allclose(model.input_mean, torch.full((32,), 2.0)) and "input_std" in model.state_dict()
    assert model.embedding.linear.out_features == 8 and model.embedding.block1.block[2].out_features == 32
    _, x, mask = padded([4, 9], 32)
    with torch.no_grad():
        z = model.embedding_pooling(x, mask)
        cos = model(x, mask, x, mask)
    assert z.shape == (2, 8) and (z < 0).any()               # linear output can be negative
    assert torch.allclose(cos, torch.ones(2), atol=1e-5)
    clone = TransformerEmbeddingCosine(**SMALL, output_dim=8, output_activation="none", pooling="mean_ln",
                                       norm_first=True, input_norm=True)
    clone.load_state_dict(model.state_dict())
    assert torch.allclose(clone.input_std, torch.full((32,), 4.0))


@pytest.mark.skipif(not os.path.exists(PROD_PT), reason="production weights not cached")
def test_production_weights_load_into_the_default_architecture():
    sd = torch.load(PROD_PT, map_location="cpu", weights_only=False)
    sd = sd.state_dict() if hasattr(sd, "state_dict") and not isinstance(sd, dict) else sd
    model = TransformerEmbeddingCosine(input_features=1536, dim_feedforward=3072, hidden_layer=1536, nhead=12,
                                       num_layers=6, res_block_layers=12)
    model.load_state_dict(sd, strict=True)


def make_store(tmp_path, domains, lengths, dim=32):
    src = tmp_path / "pt"
    src.mkdir()
    g = torch.Generator().manual_seed(1)
    paths = {}
    for d, n in zip(domains, lengths):
        torch.save(torch.randn(n, dim, generator=g), src / f"{d}.pt")
        paths[d] = str(src / f"{d}.pt")
    store = tmp_path / "store"
    build_packed_store(paths, str(store), target_dtype=np.float16)
    return str(store)


@pytest.fixture()
def pair_data(tmp_path):
    rng = np.random.default_rng(0)
    domains = [f"d{p}a{k}" for p, k in zip(["1abc", "2xyz", "3qqq"], "123")] + ["1abcA01", "1abcB01", "4zzzA00"]
    domains += [f"5{k:03d}A00" for k in range(14)]
    lengths = rng.integers(5, 40, len(domains))
    store = make_store(tmp_path, domains, lengths)
    n = 4000
    i, j = rng.integers(0, len(domains), n), rng.integers(0, len(domains), n)
    score = np.clip(rng.beta(1.2, 3, n), 0, 1).astype(np.float32)
    score[:40] = 1.0
    pairs = tmp_path / "pairs.npz"
    np.savez(pairs, domains=np.array(domains), i=i.astype(np.int32), j=j.astype(np.int32), score=score)
    return store, str(pairs), domains, i, j, score


def test_cath_twins():
    assert cath_twins({"d1abca1"}, ["1abcA01", "1abcB01", "1abda01"]) == {"1abcA01"}


def test_dataset_items_and_exclusion(pair_data):
    store, pairs, domains, i, j, score = pair_data
    ds = PackedTmPairDataset(store, pairs)
    assert len(ds) == len(i)
    x, y, t = ds[0]
    assert x.dtype == torch.float32 and x.shape[1] == 32 and float(t) == round(10 * float(score[0])) / 10
    ex = PackedTmPairDataset(store, pairs, exclude_domains={"d1abca1"}, exclude_cath_twins=True)
    dropped = {"d1abca1", "1abcA01"}
    kept_names = {ex.domains[k] for k in np.concatenate([ex.i, ex.j])}
    assert not (kept_names & dropped)
    bad = np.array([d in dropped for d in domains])
    assert len(ex) == int((~(bad[i] | bad[j])).sum())


def test_sampler_matches_the_production_distribution(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    sampler = BinBalancedPairBatchSampler(ds, batch_size=8, samples_per_epoch=400_000, num_replicas=1, rank=0)
    drawn = np.concatenate([np.array(b) for b in sampler])
    weights = tm_score_weights(5, 0.25)(np.expand_dims(ds.score, 1)).numpy()
    expected = np.bincount(tm_bins(ds.score), weights=weights, minlength=6) / weights.sum()
    observed = np.bincount(tm_bins(ds.score[drawn]), minlength=6) / len(drawn)
    assert np.abs(observed - expected).max() < 0.005
    # within a bin, every pair is (roughly) equally likely
    in_bin = np.nonzero(tm_bins(ds.score) == 1)[0]
    counts = np.bincount(drawn, minlength=len(ds))[in_bin]
    assert counts.min() > 0 and counts.std() / counts.mean() < 0.25
    # same distribution as the original sampler on the same weights
    ref = np.array(list(iter(CustomWeightedRandomSampler(weights=torch.tensor(weights), num_samples=200_000,
                                                          replacement=True))))
    ref_bins = np.bincount(tm_bins(ds.score[ref]), minlength=6) / len(ref)
    assert np.abs(observed - ref_bins).max() < 0.01


def test_sampler_ranks_epochs_and_batches(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    a = BinBalancedPairBatchSampler(ds, 8, 4000, seed=3, num_replicas=4, rank=0)
    b = BinBalancedPairBatchSampler(ds, 8, 4000, seed=3, num_replicas=4, rank=1)
    ba, bb = list(a), list(b)
    assert len(ba) == len(bb) == len(a) == 4000 // 4 // 8 and all(len(x) == 8 for x in ba)
    assert ba != bb
    a.set_epoch(1)
    assert list(a) != ba
    a.set_epoch(0)
    assert list(a) == ba
    lengths = ds.pair_lengths()
    spread = np.mean([lengths[x].max() - lengths[x].min() for x in ba])
    random_spread = np.mean([np.ptp(lengths[np.random.default_rng(k).integers(0, len(ds), 8)]) for k in range(200)])
    assert spread < 0.5 * random_spread


def test_validation_sampler_is_even_across_ranks(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    parts = [list(ShardedSequentialBatchSampler(ds, 16, num_replicas=3, rank=r)) for r in range(3)]
    assert len({len(p) for p in parts}) == 1
    flat = [k for p in parts for b in p for k in b]
    assert len(flat) == len(set(flat))


def test_schedule_and_parameter_groups():
    model = TransformerEmbeddingCosine(**SMALL, output_dim=8, output_activation="none")
    lit = LitEmbeddingTrainingV2(model, lr_backbone=1e-4, lr_head=1e-3, total_steps=100, warmup_frac=0.1,
                                 weight_decay=0.05)
    conf = lit.configure_optimizers()
    opt, sched = conf["optimizer"], conf["lr_scheduler"]["scheduler"]
    head = [g for g in opt.param_groups if any(p is model.embedding.linear.weight for p in g["params"])][0]
    assert head["lr"] == pytest.approx(1e-3 * 0.1) and head["weight_decay"] == 0.05
    no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert any(any(p is model.embedding.linear.bias for p in g["params"]) for g in no_decay)
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"] / opt.param_groups[0]["initial_lr"])
    assert lrs[8] == pytest.approx(1.0) and lrs[-1] < 1e-3 and all(np.diff(lrs[10:]) <= 1e-12)


def test_training_script_runs_end_to_end(pair_data, tmp_path):
    store, pairs, *_ = pair_data
    root = tmp_path / "run"
    cfg = {
        "global_seed": 0, "default_root_dir": str(root),
        "logger": {"save_dir": str(root), "name": "log"},
        "computing_resources": {"nodes": 1, "devices": 1, "strategy": "auto", "accelerator": "cpu"},
        "data": {"store_path": store, "train_pairs": pairs, "val_pairs": pairs, "batch_size": 8, "workers": 0,
                 "draws": 8 * 12, "samples_per_epoch": 8 * 6, "val_batch_size": 64},
        "embedding_network": {"_target_": "networks.transformer_nn.TransformerEmbeddingCosine", **SMALL,
                              "output_dim": 8, "output_activation": "none", "pooling": "mean_ln",
                              "norm_first": True},
        "init": {"checkpoint": None, "head": None},
        "optim": {"lr_backbone": 1e-4, "lr_head": 1e-3, "warmup_frac": 0.1, "weight_decay": 0.01, "grad_clip": 1.0},
        "checkpoint_every_epochs": 1, "last_every_n_steps": 4, "resume": None,
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    import yaml
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    script = os.path.join(os.path.dirname(__file__), "..", "training", "embedding_tm_score_v2.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    r = subprocess.run([sys.executable, script, f"--config-path={tmp_path}", "--config-name=cfg",
                        f"hydra.run.dir={tmp_path}/hydra"], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    ckpts = sorted(os.listdir(root / "checkpoints"))
    assert "last.ckpt" in ckpts and "epoch000.ckpt" in ckpts and "epoch001.ckpt" in ckpts
    sd = torch.load(root / "checkpoints" / "epoch001.ckpt", map_location="cpu", weights_only=False)
    assert sd["global_step"] == math.ceil(cfg["data"]["draws"] / 8)
    # resume from last.ckpt into a longer run: continues at step 12 (epoch 2, batch 0) up to 18
    last = torch.load(root / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["global_step"] == 12 and "optimizer_states" in last
    r = subprocess.run([sys.executable, script, f"--config-path={tmp_path}", "--config-name=cfg",
                        f"hydra.run.dir={tmp_path}/hydra2", "resume=auto", f"data.draws={8 * 18}"],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = r.stdout + r.stderr                      # Hydra's console handler logs to stdout
    assert "resuming from" in out and "global step 12 = epoch 2 + 0 batches" in out, out[-3000:]
    sd = torch.load(root / "checkpoints" / "epoch002.ckpt", map_location="cpu", weights_only=False)
    assert sd["global_step"] == 18


def test_labels_round_like_production(pair_data):
    from dataset.packed_tm_pairs_dataset import BinaryScore
    from dataset.utils.tm_score_weight import FractionScore
    store, pairs, *_ = pair_data
    z = dict(np.load(pairs))
    z["score"] = np.array([0.05, 0.35, 0.85, 0.95, 0.70001, 0.7] + [0.2] * (len(z["i"]) - 6), dtype=np.float32)
    np.savez(pairs, **z)
    ds = PackedTmPairDataset(store, pairs)
    prod = FractionScore(10)
    for k in range(6):          # production fed np.float32 from pandas .iloc
        assert float(ds[k][2]) == pytest.approx(prod(np.float32(z["score"][k])))
    val = PackedTmPairDataset(store, pairs, target=BinaryScore(0.7))
    assert [float(val[k][2]) for k in (4, 5)] == [1.0, 0.0]


def test_resume_skips_consumed_batches_and_epoch_zero_is_not_replayed(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    ref = BinBalancedPairBatchSampler(ds, 8, 800, seed=5, num_replicas=2, rank=1)
    ref.set_epoch(3)
    full = list(ref)
    resumed = BinBalancedPairBatchSampler(ds, 8, 800, seed=5, num_replicas=2, rank=1)
    resumed.set_epoch(3)
    resumed.resume = (3, 20)
    assert list(resumed) == full[20:] and list(resumed) == full[20:]     # torch may iterate twice
    resumed.set_epoch(3)                                                 # Lightning re-sets the same epoch
    assert list(resumed) == full[20:]
    resumed.set_epoch(4)
    ref.set_epoch(4)
    assert resumed.resume is None and list(resumed) == list(ref)


def test_ranks_share_the_length_order_of_batches(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    lengths = ds.pair_lengths()
    per_rank = [list(BinBalancedPairBatchSampler(ds, 8, 8 * 4 * 64, pool_batches=64, seed=1, num_replicas=4, rank=r))
                for r in range(4)]
    step_max = np.array([[lengths[b].max() for b in batches] for batches in per_rank])     # [rank, step]
    within_step = np.mean(step_max.max(0) - step_max.min(0))
    shuffled = np.stack([np.random.default_rng(r).permutation(row) for r, row in enumerate(step_max)])
    assert within_step < 0.6 * np.mean(shuffled.max(0) - shuffled.min(0))


def test_validation_batches_are_full(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    for r in range(3):
        assert all(len(b) == 16 for b in ShardedSequentialBatchSampler(ds, 16, num_replicas=3, rank=r))


def test_sequence_view_mixing(pair_data, tmp_path):
    store, pairs, domains, i, j, score = pair_data
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    # sequence-only views for all but the last domain, with other values (and lengths) than the structure views
    seq_store = make_store(seq_dir, domains[:-1], [n + 1 for n in range(5, 5 + len(domains) - 1)])
    base = PackedTmPairDataset(store, pairs)
    off = PackedTmPairDataset(store, pairs, seq_store_dir=seq_store, seq_mix=0.0)
    assert off.seq_store is None and all(torch.equal(off[k][0], base[k][0]) for k in range(20))
    full = PackedTmPairDataset(store, pairs, seq_store_dir=seq_store, seq_mix=1.0)
    for k in range(200):
        x, y, t = full[k]
        di, dj = domains[i[k]], domains[j[k]]
        assert torch.equal(x, full.seq_store.get(di).float() if di != domains[-1] else base.store.get(di).float())
        assert torch.equal(y, full.seq_store.get(dj).float() if dj != domains[-1] else base.store.get(dj).float())
        assert float(t) == float(base[k][2])
    torch.manual_seed(0)
    half = PackedTmPairDataset(store, pairs, seq_store_dir=seq_store, seq_mix=0.5)
    n_seq = sum(not torch.equal(half[k][0], base[k][0]) for k in range(400) if domains[i[k]] != domains[-1])
    n_all = sum(domains[i[k]] != domains[-1] for k in range(400))
    assert 0.4 < n_seq / n_all < 0.6


def test_dataset_meta_reports_domains_and_views(pair_data, tmp_path):
    store, pairs, domains, i, j, score = pair_data
    seq_dir = tmp_path / "seqm"
    seq_dir.mkdir()
    seq_store = make_store(seq_dir, domains, [n + 2 for n in range(len(domains))])
    for p, expect in ((0.0, 0), (1.0, 1)):
        ds = PackedTmPairDataset(store, pairs, seq_store_dir=seq_store, seq_mix=p)
        ds.return_meta = True
        for k in range(30):
            x, y, t, meta = ds[k]
            assert meta.tolist() == [int(ds.i[k]), int(ds.j[k]), expect, expect]
            assert x.shape[0] == (ds.seq_store or ds.store).length(domains[ds.i[k]])


def _kd_self(gather=False, tau=0.05):
    from types import SimpleNamespace
    return SimpleNamespace(gather=gather, kd_tau=tau)


def test_kd_loss_is_the_head_screen_recipe():
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(24, 8, generator=g), torch.randn(24, 16, generator=g)
    got = LitEmbeddingTrainingKD.kd_loss(_kd_self(), s, t)
    # train_heads.py (frozen-head screens)
    tt = F.normalize(t, dim=1)
    pt = F.softmax((tt @ tt.T).fill_diagonal_(-1e4) / 0.05, dim=1)
    zs = F.normalize(s, dim=1)
    ls = F.log_softmax((zs @ zs.T).fill_diagonal_(-1e4) / 0.05, dim=1)
    assert torch.allclose(got, F.kl_div(ls, pt, reduction="batchmean"), atol=1e-6)
    assert float(LitEmbeddingTrainingKD.kd_loss(_kd_self(), t, t)) < 1e-6
    # teacher rows: structure or sequence-only view per side
    tab, tab_seq = torch.randn(5, 4, generator=g), torch.randn(5, 4, generator=g)
    m = LitEmbeddingTrainingKD.__new__(LitEmbeddingTrainingKD)
    torch.nn.Module.__init__(m)
    m.register_buffer("teacher", tab)
    m.register_buffer("teacher_seq", tab_seq)
    meta = torch.tensor([[0, 1, 0, 1], [2, 3, 1, 0]])
    tv = m.teacher_vectors(meta)
    assert torch.equal(tv, torch.stack([tab[0], tab_seq[2], tab_seq[1], tab[3]]))


KD_WORKER = """
import sys, torch
import torch.distributed as dist
from types import SimpleNamespace
from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
rank, world, path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=world)
g = torch.Generator().manual_seed(0)
x, t, w0 = torch.randn(16, 6, generator=g), torch.randn(16, 5, generator=g), torch.randn(6, 4, generator=g)
sl = slice(rank * 8, (rank + 1) * 8)
w = w0.clone().requires_grad_(True)
loss = LitEmbeddingTrainingKD.kd_loss(SimpleNamespace(gather=True, kd_tau=0.05), x[sl] @ w, t[sl])
loss.backward()
grad = w.grad.clone()
dist.all_reduce(grad)
grad /= world                                   # what DDP does with parameter gradients
if rank == 0:
    torch.save({"loss": loss.detach(), "grad": grad}, out)
dist.destroy_process_group()
"""


def test_kd_gather_gives_the_full_batch_gradient(tmp_path):
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    script, out = tmp_path / "kd_worker.py", str(tmp_path / "kd.pt")
    script.write_text(KD_WORKER)
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    procs = [subprocess.Popen([sys.executable, str(script), str(r), "2", str(tmp_path / "pg"), out], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in range(2)]
    for p in procs:
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err[-3000:]
    g = torch.Generator().manual_seed(0)
    x, t, w0 = torch.randn(16, 6, generator=g), torch.randn(16, 5, generator=g), torch.randn(6, 4, generator=g)
    w = w0.clone().requires_grad_(True)
    full = LitEmbeddingTrainingKD.kd_loss(_kd_self(), x @ w, t)
    full.backward()
    res = torch.load(out)
    assert torch.allclose(res["loss"], full.detach(), atol=1e-5)
    assert torch.allclose(res["grad"], w.grad, atol=1e-5)


def test_training_script_runs_with_distillation_and_sequence_mixing(pair_data, tmp_path):
    store, pairs, domains, *_ = pair_data
    seq_dir = tmp_path / "seqk"
    seq_dir.mkdir()
    seq_store = make_store(seq_dir, domains, [n + 3 for n in range(len(domains))])
    g = np.random.default_rng(0)
    z = g.normal(size=(len(domains) - 1, 12)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    teacher = tmp_path / "teacher.npz"
    np.savez(teacher, domains=np.array(domains[:-1]), z=z, z_seq=z[::-1].copy())
    root = tmp_path / "runkd"
    cfg = {
        "global_seed": 0, "default_root_dir": str(root),
        "logger": {"save_dir": str(root), "name": "log"},
        "computing_resources": {"nodes": 1, "devices": 1, "strategy": "auto", "accelerator": "cpu"},
        "data": {"store_path": store, "train_pairs": pairs, "val_pairs": pairs, "batch_size": 8, "workers": 2,
                 "draws": 8 * 6, "samples_per_epoch": 8 * 6, "val_batch_size": 64,
                 "seq_store_path": seq_store, "seq_mix": 0.3},
        "embedding_network": {"_target_": "networks.transformer_nn.TransformerEmbeddingCosine", **SMALL,
                              "output_dim": 4, "output_activation": "none"},
        "init": {"checkpoint": None, "head": None},
        "optim": {"lr_backbone": 1e-4, "lr_head": 1e-3, "warmup_frac": 0.1, "grad_clip": 1.0},
        "kd": {"teacher": str(teacher), "weight": 1.0, "tau": 0.05, "cosent": 0.001},
        "checkpoint_every_epochs": 1, "last_every_n_steps": 4, "resume": None, "log_every_n_steps": 1,
    }
    import yaml
    (tmp_path / "cfgkd.yaml").write_text(yaml.safe_dump(cfg))
    script = os.path.join(os.path.dirname(__file__), "..", "training", "embedding_tm_score_v2.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    r = subprocess.run([sys.executable, script, f"--config-path={tmp_path}", "--config-name=cfgkd",
                        f"hydra.run.dir={tmp_path}/hydrakd"], env=env, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "1 of" in out and "have no vector" in out and "sequence-only views True" in out, out[-3000:]
    assert "epoch000.ckpt" in os.listdir(root / "checkpoints")


def test_cosent_term_matches_the_head_recipe():
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    g = torch.Generator().manual_seed(3)
    cos, t = torch.rand(20, generator=g) * 2 - 1, torch.round(torch.rand(20, generator=g) * 10) / 10
    got = LitEmbeddingTrainingKD.cosent_loss(_kd_self(), cos, t)
    diff = (cos[None, :] - cos[:, None]) * 20.0
    order = t[:, None] > t[None, :]
    want = torch.logsumexp(torch.cat([torch.zeros(1), diff[order]]), 0)     # train_heads.py cosent
    assert torch.allclose(got, want)
    perfect = LitEmbeddingTrainingKD.cosent_loss(_kd_self(), t * 10, t)       # correctly ranked: loss -> log(1 + ~0)
    assert float(perfect) < 1e-3


def test_length_feature_gives_mean_pooling_the_chain_length():
    torch.manual_seed(0)
    opts = dict(SMALL, output_dim=8, output_activation="none", pooling="mean_ln", norm_first=True)
    base = no_dropout(TransformerEmbeddingCosine(**opts)).eval()
    lf = no_dropout(TransformerEmbeddingCosine(**opts, length_feature=True)).eval()
    missing = lf.load_state_dict(base.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"length_proj.weight", "length_proj.bias"}
    xs, x, mask = padded([7, 12, 5], SMALL["input_features"])
    with torch.no_grad():
        assert torch.allclose(lf.embedding_pooling(x, mask), base.embedding_pooling(x, mask), atol=1e-6)   # zero init
        lf.length_proj.weight.fill_(0.5)
        # a chain and the same rows twice: identical mean (no positional encoding), different length
        one = xs[0].unsqueeze(0)
        two = torch.cat([one, one], dim=1)
        assert torch.allclose(base.embedding_pooling(one, None), base.embedding_pooling(two, None), atol=1e-4)
        assert not torch.allclose(lf.embedding_pooling(one, None), lf.embedding_pooling(two, None), atol=1e-3)
        # padding still does not matter: the length counts unmasked rows only (rows 0 and 2 are padded)
        for r in (0, 2):
            alone = lf.embedding_pooling(xs[r].unsqueeze(0), None)
            assert torch.allclose(lf.embedding_pooling(x, mask)[r:r + 1], alone, atol=1e-4)


def test_containment_strata_and_sampler(pair_data):
    from dataset.packed_tm_pairs_dataset import containment_strata
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    res = ds.residue_lengths()
    assert np.array_equal(res, ds.domain_length - 2)
    pos, neg = containment_strata(ds, pep_len=20, ratio=1.5, tm_pos=0.4, tm_neg=0.2)
    lo = np.minimum(res[ds.i], res[ds.j])
    hi = np.maximum(res[ds.i], res[ds.j])
    base = (lo < 20) & (hi >= 1.5 * lo)
    assert np.array_equal(pos, np.nonzero(base & (ds.score >= 0.4))[0]) and len(pos) > 0
    assert np.array_equal(neg, np.nonzero(base & (ds.score < 0.2))[0]) and len(neg) > 0
    plain = BinBalancedPairBatchSampler(ds, 8, 40_000, seed=2, num_replicas=1, rank=0)
    assert plain.strata == [] and all(isinstance(k, int) for b in plain for k in b)
    s = BinBalancedPairBatchSampler(ds, 8, 40_000, seed=2, num_replicas=1, rank=0, strata=[(pos, 0.1), (neg, 0.2)])
    batches = list(s)
    assert len(batches) == len(plain) and all(len(b) == 8 for b in batches)
    flat = np.array([kv for b in batches for kv in b])                      # (pair index, stratum)
    frac = np.bincount(flat[:, 1], minlength=3) / len(flat)
    assert abs(frac[1] - 0.1) < 0.01 and abs(frac[2] - 0.2) < 0.01
    assert np.isin(flat[flat[:, 1] == 1, 0], pos).all() and np.isin(flat[flat[:, 1] == 2, 0], neg).all()
    # stratum-0 draws keep the production bin distribution
    prod = flat[flat[:, 1] == 0, 0]
    ref = np.concatenate([np.array(b) for b in plain])
    assert np.abs(np.bincount(tm_bins(ds.score[prod]), minlength=6) / len(prod)
                  - np.bincount(tm_bins(ds.score[ref]), minlength=6) / len(ref)).max() < 0.01
    # deterministic per (seed, rank, epoch), and resume skips consumed batches
    assert list(s) == batches
    r = BinBalancedPairBatchSampler(ds, 8, 40_000, seed=2, num_replicas=1, rank=0, strata=[(pos, 0.1), (neg, 0.2)])
    r.resume = (0, 7)
    assert list(r) == batches[7:]
    with pytest.raises(AssertionError):
        BinBalancedPairBatchSampler(ds, 8, 800, strata=[(pos, 0.6), (neg, 0.5)])
    # an empty pool keeps its weight 0 and the others renormalise (train_heads.py with_strata): 0.1 / 0.9 here
    e = BinBalancedPairBatchSampler(ds, 8, 40_000, seed=2, num_replicas=1, rank=0,
                                    strata=[(np.array([], dtype=np.int64), 0.1), (neg, 0.1)])
    st = np.array([kv[1] for b in e for kv in b])
    assert (st != 1).all() and abs((st == 2).mean() - 0.1 / 0.9) < 0.01
    none = BinBalancedPairBatchSampler(ds, 8, 800, seed=2, num_replicas=1, rank=0, strata=[(np.array([]), 0.1)])
    assert none.strata == [] and list(none) == list(BinBalancedPairBatchSampler(ds, 8, 800, seed=2, num_replicas=1, rank=0))


def test_dataset_meta_carries_the_stratum(pair_data):
    store, pairs, *_ = pair_data
    ds = PackedTmPairDataset(store, pairs)
    x, y, t = ds[(5, 2)]
    x0, y0, t0 = ds[5]
    assert torch.equal(x, x0) and torch.equal(y, y0) and float(t) == float(t0)
    ds.return_meta = True
    assert ds[(5, 2)][3].tolist() == [int(ds.i[5]), int(ds.j[5]), 0, 0, 2]
    assert ds[5][3].tolist() == [int(ds.i[5]), int(ds.j[5]), 0, 0]


def _kd_module(min_len=0, lengths=None):
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    m = LitEmbeddingTrainingKD.__new__(LitEmbeddingTrainingKD)
    torch.nn.Module.__init__(m)
    m.gather, m.kd_tau, m.kd_min_len = False, 0.05, min_len
    if min_len:
        m.register_buffer("domain_length", torch.as_tensor(lengths))
    return m


def test_kd_rows_leave_out_short_domains_and_strata_rows():
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    lengths = torch.tensor([10, 60, 49, 50, 200])
    meta = torch.tensor([[0, 1, 0, 0, 0], [2, 3, 0, 0, 0], [4, 1, 0, 0, 1], [3, 4, 0, 0, 0]])
    keep = LitEmbeddingTrainingKD.kd_rows(_kd_module(50, lengths), meta)
    # rows: i-sides then j-sides; pair 2 is a stratum row (both sides out); domains 0 (10) and 2 (49) too short
    assert keep.tolist() == [False, False, False, True, True, True, False, True]
    assert LitEmbeddingTrainingKD.kd_rows(_kd_module(), meta[:, :4]) is None
    assert LitEmbeddingTrainingKD.kd_rows(_kd_module(), meta).tolist() == [True, True, False, True] * 2
    g = torch.Generator().manual_seed(1)
    s, t = torch.randn(8, 6, generator=g), torch.randn(8, 5, generator=g)
    got = LitEmbeddingTrainingKD.kd_loss(_kd_self(), s, t, keep)
    assert torch.allclose(got, LitEmbeddingTrainingKD.kd_loss(_kd_self(), s[keep], t[keep]), atol=1e-6)
    one = torch.zeros(8, dtype=torch.bool)
    one[3] = True
    assert float(LitEmbeddingTrainingKD.kd_loss(_kd_self(), s, t, one)) == 0.0


KD_WORKER_MASK = KD_WORKER.replace(
    "loss = LitEmbeddingTrainingKD.kd_loss(SimpleNamespace(gather=True, kd_tau=0.05), x[sl] @ w, t[sl])",
    "keep = torch.arange(16) % 3 != 0\n"
    "loss = LitEmbeddingTrainingKD.kd_loss(SimpleNamespace(gather=True, kd_tau=0.05), x[sl] @ w, t[sl], keep[sl])")


def test_kd_gather_with_left_out_rows_gives_the_full_batch_gradient(tmp_path):
    from lightning_module.training.embedding_training import LitEmbeddingTrainingKD
    assert KD_WORKER_MASK != KD_WORKER
    script, out = tmp_path / "kd_worker_mask.py", str(tmp_path / "kdm.pt")
    script.write_text(KD_WORKER_MASK)
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    procs = [subprocess.Popen([sys.executable, str(script), str(r), "2", str(tmp_path / "pgm"), out], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in range(2)]
    for p in procs:
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err[-3000:]
    g = torch.Generator().manual_seed(0)
    x, t, w0 = torch.randn(16, 6, generator=g), torch.randn(16, 5, generator=g), torch.randn(6, 4, generator=g)
    w = w0.clone().requires_grad_(True)
    keep = torch.arange(16) % 3 != 0
    full = LitEmbeddingTrainingKD.kd_loss(_kd_self(), x @ w, t, keep)
    full.backward()
    res = torch.load(out)
    assert torch.allclose(res["loss"], full.detach(), atol=1e-5)
    assert torch.allclose(res["grad"], w.grad, atol=1e-5)


def test_training_script_runs_with_strata_and_masked_distillation(pair_data, tmp_path):
    store, pairs, domains, *_ = pair_data
    g = np.random.default_rng(0)
    z = g.normal(size=(len(domains), 12)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    teacher = tmp_path / "teacher_m.npz"
    np.savez(teacher, domains=np.array(domains), z=z)
    root = tmp_path / "runm"
    cfg = {
        "global_seed": 0, "default_root_dir": str(root),
        "logger": {"save_dir": str(root), "name": "log"},
        "computing_resources": {"nodes": 1, "devices": 1, "strategy": "auto", "accelerator": "cpu"},
        "data": {"store_path": store, "train_pairs": pairs, "val_pairs": pairs, "batch_size": 8, "workers": 2,
                 "draws": 8 * 6, "samples_per_epoch": 8 * 6, "val_batch_size": 64, "pool_batches": 1, "contain": None},
        "embedding_network": {"_target_": "networks.transformer_nn.TransformerEmbeddingCosine", **SMALL,
                              "output_dim": 4, "output_activation": "none", "pooling": "mean_ln", "norm_first": True,
                              "length_feature": True},
        "init": {"checkpoint": None, "head": None},
        "optim": {"lr_backbone": 1e-4, "lr_head": 1e-3, "warmup_frac": 0.1, "grad_clip": 1.0},
        "kd": {"teacher": str(teacher), "weight": 0.3, "tau": 0.05, "cosent": 0.03, "min_len": 0},
        "checkpoint_every_epochs": 1, "last_every_n_steps": 4, "resume": None, "log_every_n_steps": 1,
    }
    import yaml
    (tmp_path / "cfgm.yaml").write_text(yaml.safe_dump(cfg))
    script = os.path.join(os.path.dirname(__file__), "..", "training", "embedding_tm_score_v2.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    # the overrides make_stage1_jobs.py writes
    r = subprocess.run([sys.executable, script, f"--config-path={tmp_path}", "--config-name=cfgm",
                        f"hydra.run.dir={tmp_path}/hydram", "kd.min_len=20",
                        "data.contain={mass:0.2,ratio:1.5,tm_pos:0.4,tm_neg:0.2,pep_len:20}"],
                       env=env, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "containment strata:" in out and "strata (pairs, mass)" in out and "min_len 20" in out, out[-3000:]
    assert "epoch000.ckpt" in os.listdir(root / "checkpoints")


@pytest.mark.parametrize("case", [(m, k) for m in ("add", "film", "blocks", "input") for k in (0, 8)])
def test_length_modes_start_as_the_base_model_and_see_the_length(case):
    mode, rbf = case
    torch.manual_seed(0)
    opts = dict(SMALL, output_dim=8, output_activation="none", pooling="mean_ln", norm_first=True)
    base = no_dropout(TransformerEmbeddingCosine(**opts)).eval()
    lf = no_dropout(TransformerEmbeddingCosine(**opts, length_feature=mode, length_rbf=rbf)).eval()
    missing = set(lf.load_state_dict(base.state_dict(), strict=False).missing_keys)
    assert missing and all(k.startswith("length_") for k in missing)
    xs, x, mask = padded([7, 12, 5], SMALL["input_features"])
    one = xs[0].unsqueeze(0)
    two = torch.cat([one, one], dim=1)
    with torch.no_grad():
        assert torch.allclose(lf.embedding_pooling(x, mask), base.embedding_pooling(x, mask), atol=1e-6)
        for name, prm in lf.named_parameters():
            if name.startswith("length_"):
                prm.copy_(torch.randn_like(prm) * 0.5)
        # the same rows twice: identical mean for the base model, different length for this one
        assert torch.allclose(base.embedding_pooling(one, None), base.embedding_pooling(two, None), atol=1e-4)
        assert not torch.allclose(lf.embedding_pooling(one, None), lf.embedding_pooling(two, None), atol=1e-3)
        # padding still does not matter: the length counts unmasked rows only (rows 0 and 2 are padded)
        batch = lf.embedding_pooling(x, mask)
        for r in (0, 2):
            assert torch.allclose(batch[r:r + 1], lf.embedding_pooling(xs[r].unsqueeze(0), None), atol=1e-4)
            # one unbatched [L, D] chain: same vector, no batch dim (as the base model)
            single = lf.embedding_pooling(xs[r], None)
            assert single.shape == (8,) and torch.allclose(single, batch[r], atol=1e-4)
            assert lf.embedding_pooling(xs[r], torch.zeros(len(xs[r]), dtype=torch.bool)).shape == (8,)
        # the evaluator's path (embed_stage1.py): pool -> blocks (h) -> head
        code = lf.length_code(x, mask)
        assert torch.allclose(lf.head(lf.blocks(lf.pool(x, mask, code), code)), batch, atol=1e-6)


def test_length_rbf_code_and_unknown_mode():
    m = TransformerEmbeddingCosine(**SMALL, output_dim=8, output_activation="none", pooling="mean_ln",
                                   norm_first=True, length_feature="add", length_rbf=8)
    x = torch.zeros(3, 40, SMALL["input_features"])
    mask = torch.ones(3, 40, dtype=torch.bool)
    mask[0, :5], mask[1, :40], mask[2, :int(round(np.exp(np.log(5) + np.log(300)) ** 0.5))] = False, False, False
    code = m.length_code(x, mask)
    assert code.shape == (3, 8) and float(code.max()) <= 1.0 and float(code.min()) >= 0.0
    assert int(code[0].argmax()) == 0 and float(code[0, 0]) == pytest.approx(1.0)       # 5 rows: first centre
    assert TransformerEmbeddingCosine(**SMALL, length_feature=True).length_code(x, mask).shape == (3, 1)
    with pytest.raises(ValueError):
        TransformerEmbeddingCosine(**SMALL, length_feature="sideways")


@pytest.mark.parametrize("mode", ["film", "blocks", "input"])
def test_training_script_runs_with_length_modes(pair_data, tmp_path, mode):
    store, pairs, *_ = pair_data
    root = tmp_path / f"run_{mode}"
    cfg = {
        "global_seed": 0, "default_root_dir": str(root),
        "logger": {"save_dir": str(root), "name": "log"},
        "computing_resources": {"nodes": 1, "devices": 1, "strategy": "auto", "accelerator": "cpu"},
        "data": {"store_path": store, "train_pairs": pairs, "val_pairs": pairs, "batch_size": 8, "workers": 0,
                 "draws": 8 * 4, "samples_per_epoch": 8 * 4, "val_batch_size": 64},
        "embedding_network": {"_target_": "networks.transformer_nn.TransformerEmbeddingCosine", **SMALL,
                              "output_dim": 4, "output_activation": "none", "pooling": "mean_ln", "norm_first": True,
                              "length_feature": False, "length_rbf": 0},
        "init": {"checkpoint": None, "head": None},
        "optim": {"lr_backbone": 1e-4, "lr_head": 1e-3, "warmup_frac": 0.1, "grad_clip": 1.0},
        "checkpoint_every_epochs": 1, "last_every_n_steps": 4, "resume": None,
    }
    import yaml
    (tmp_path / "cfgl.yaml").write_text(yaml.safe_dump(cfg))
    script = os.path.join(os.path.dirname(__file__), "..", "training", "embedding_tm_score_v2.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    r = subprocess.run([sys.executable, script, f"--config-path={tmp_path}", "--config-name=cfgl",
                        f"hydra.run.dir={tmp_path}/hydral", f"embedding_network.length_feature={mode}",
                        "embedding_network.length_rbf=8"], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    sd = torch.load(root / "checkpoints" / "epoch000.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    key = {"film": "model.length_scale.weight", "blocks": "model.length_blocks.0.weight", "input": "model.length_input.weight"}[mode]
    assert key in sd and sd[key].shape == (SMALL["input_features"], 8) and float(sd[key].abs().sum()) > 0
