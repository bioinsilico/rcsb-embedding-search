import math

import torch
from torch import nn, cat
from lightning_module.lightning_core import LitStructureCore


class LitEmbeddingTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred.detach()), dim=0)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred.detach()), dim=0)


class LitEmbeddingTrainingV2(LitEmbeddingTraining):
    """Same loss and validation as :class:`LitEmbeddingTraining`, with a step-based schedule.

    * Two parameter groups: the final Linear (``embedding.linear``) at ``lr_head``, everything else
      at ``lr_backbone``.
    * Linear warm-up over ``warmup_frac`` of ``total_steps``, then cosine decay to 0 at
      ``total_steps`` (the production schedule spanned 1000 epochs and never decayed).
    * Weight decay applies to weight matrices only (not biases, LayerNorms or input statistics).
    """

    def __init__(self, nn_model, lr_backbone, lr_head, total_steps, warmup_frac=0.05, weight_decay=0.0,
                 params=None):
        super().__init__(nn_model, learning_rate=lr_backbone, params=params)
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(warmup_frac * total_steps))
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss_step", loss, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_end(self):
        # AUCs over the whole validation set, not an average of per-rank AUCs
        self.z = self.all_gather(self.z).flatten()
        self.z_pred = self.all_gather(self.z_pred).flatten()
        super().on_validation_epoch_end()

    def configure_optimizers(self):
        groups = {"head_decay": [], "head_no_decay": [], "body_decay": [], "body_no_decay": []}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            part = "head" if name.startswith("embedding.linear.") else "body"
            decay = "decay" if p.dim() >= 2 else "no_decay"
            groups[f"{part}_{decay}"].append(p)
        param_groups = []
        for key, params in groups.items():
            if params:
                param_groups.append({
                    "params": params,
                    "lr": self.lr_head if key.startswith("head") else self.lr_backbone,
                    "weight_decay": 0.0 if key.endswith("no_decay") else self.weight_decay,
                })
        optimizer = torch.optim.AdamW(param_groups)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


class LitEmbeddingTrainingKD(LitEmbeddingTrainingV2):
    """:class:`LitEmbeddingTrainingV2` plus relational distillation from a fixed teacher.

    The teacher is a table of unit vectors per training domain (``teacher``: ``[N, K]``, the domain order
    of the pairs file) and, optionally, per sequence-only view (``teacher_seq``); batches carry
    ``meta = [i, j, i_is_seq, j_is_seq]`` per pair (``PackedTmPairDataset.return_meta``). The term is
    KL(softmax(T T' / tau) || softmax(S S' / tau)) over the 2B embeddings of the batch (self-similarity
    masked), as the frozen-head screens (which drew random batches: run with ``data.pool_batches=1`` so the
    length-sorted sampler does not hand every rank the same length band). With ``gather`` the batch spans all ranks: student vectors
    are all-gathered with gradients (torch.distributed.nn), so every rank computes the same loss and the
    DDP mean of the parameter gradients is the gradient of that loss.

    Rows left out of the distillation batch (as train_heads.py): domains shorter than ``kd_min_len`` residues
    (``domain_length``: residues per domain, pairs-file order), and both sides of pairs drawn from a sampler
    stratum (``meta[:, 4] > 0``, see ``BinBalancedPairBatchSampler.strata``), which feed MSE and CoSENT only.
    """

    def __init__(self, nn_model, lr_backbone, lr_head, total_steps, teacher, teacher_seq=None, kd_weight=1.0,
                 kd_tau=0.05, gather=True, cosent_weight=0.0, kd_min_len=0, domain_length=None, warmup_frac=0.05,
                 weight_decay=0.0, params=None):
        super().__init__(nn_model, lr_backbone, lr_head, total_steps, warmup_frac, weight_decay, params)
        self.cosent_weight = cosent_weight
        self.register_buffer("teacher", torch.as_tensor(teacher, dtype=torch.float32), persistent=False)
        self.register_buffer("teacher_seq", torch.as_tensor(teacher if teacher_seq is None else teacher_seq,
                                                            dtype=torch.float32), persistent=False)
        self.kd_weight, self.kd_tau, self.gather = kd_weight, kd_tau, gather
        self.kd_min_len = int(kd_min_len or 0)
        if self.kd_min_len:
            assert domain_length is not None, "kd_min_len needs the domain lengths"
            self.register_buffer("domain_length", torch.as_tensor(domain_length, dtype=torch.long), persistent=False)

    def kd_rows(self, meta):
        """Rows of the [i-side; j-side] distillation batch that take part, or None for all."""
        keep = None
        if meta.shape[1] > 4:
            prod = meta[:, 4] == 0
            keep = torch.cat([prod, prod])
        if self.kd_min_len:
            long_enough = self.domain_length[torch.cat([meta[:, 0], meta[:, 1]])] >= self.kd_min_len
            keep = long_enough if keep is None else keep & long_enough
        return keep

    def teacher_vectors(self, meta):
        def pick(idx, is_seq):
            return torch.where(is_seq[:, None].bool(), self.teacher_seq[idx], self.teacher[idx])
        return torch.cat([pick(meta[:, 0], meta[:, 2]), pick(meta[:, 1], meta[:, 3])])

    def kd_loss(self, student, teacher, keep=None):
        s = nn.functional.normalize(student, dim=1)
        t = nn.functional.normalize(teacher, dim=1)
        if self.gather and torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed.nn.functional as dist_fn
            s = torch.cat(dist_fn.all_gather(s))
            t = torch.cat(dist_fn.all_gather(t.contiguous()))
            if keep is not None:
                k = keep.to(torch.uint8)
                parts = [torch.empty_like(k) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(parts, k)
                keep = torch.cat(parts).bool()
        if keep is not None:
            if int(keep.sum()) < 2:                      # nothing to distil: a zero that keeps the graph
                return s.sum() * 0.0
            s, t = s[keep], t[keep]
        eye = torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
        log_ps = nn.functional.log_softmax((s @ s.T).masked_fill(eye, -1e4) / self.kd_tau, dim=1)
        with torch.no_grad():
            pt = nn.functional.softmax((t @ t.T).masked_fill(eye, -1e4) / self.kd_tau, dim=1)
        return nn.functional.kl_div(log_ps, pt, reduction="batchmean")

    def cosent_loss(self, cos, target):
        """CoSENT ranking term over the pairs of the (gathered) batch: log(1 + sum exp(20 (cos_b - cos_a))) over
        pairs a, b with target_a > target_b, as train_heads.py's ``cosent`` option."""
        if self.gather and torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed.nn.functional as dist_fn
            cos = torch.cat(dist_fn.all_gather(cos))
            target = torch.cat(dist_fn.all_gather(target.contiguous()))
        diff = (cos[None, :] - cos[:, None]) * 20.0                   # [a, b] = cos_b - cos_a
        order = target[:, None] > target[None, :]                     # a should score above b
        return torch.logsumexp(torch.cat([torch.zeros(1, device=cos.device, dtype=cos.dtype), diff[order]]), 0)

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z, meta = batch
        zx = self.model.embedding_pooling(x, x_mask)
        zy = self.model.embedding_pooling(y, y_mask)
        z_pred = nn.functional.cosine_similarity(zx, zy)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred.detach()), dim=0)
        mse = nn.functional.mse_loss(z_pred, z)
        keep = self.kd_rows(meta)
        if keep is not None:
            self.log("train_kd_rows_frac", keep.float().mean(), on_step=True, on_epoch=False)
        kd = self.kd_loss(torch.cat([zx, zy]), self.teacher_vectors(meta), keep)
        self.log("train_mse_step", mse, on_step=True, on_epoch=False)
        self.log("train_kd_step", kd, on_step=True, on_epoch=False)
        loss = mse + self.kd_weight * kd
        if self.cosent_weight > 0:
            cs = self.cosent_loss(z_pred, z)
            self.log("train_cosent_step", cs, on_step=True, on_epoch=False)
            loss = loss + self.cosent_weight * cs
        self.log("train_loss_step", loss, prog_bar=False, on_step=True, on_epoch=False)
        return loss


class LitEmbeddingMaskedTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        (x, x_mask, x_attn_mask), (y, y_mask, y_attn_mask), z = batch
        z_pred = self.model(x, x_mask, x_attn_mask, y, y_mask, y_attn_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred.detach()), dim=0)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        (x, x_mask, x_attn_mask), (y, y_mask, y_attn_mask), z = batch
        z_pred = self.model(x, x_mask, x_attn_mask, y, y_mask, y_attn_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred.detach()), dim=0)