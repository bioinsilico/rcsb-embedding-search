from __future__ import annotations

import torch


def neighbors_tsv_to_src_mask(
    tsv_path: str,
    seq_len: int | None = None,
    include_self: bool = True,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Return a square attention mask from a Ca-neighbor TSV file.

    The TSV must have been written with ``--esm3-offsets`` (and no ``--labels``),
    so every index already corresponds to an ESM3 embedding position.

    The returned mask has shape ``(seq_len, seq_len)`` and can be passed directly
    as the ``attn_mask`` / ``src_mask`` argument of
    ``torch.nn.MultiheadAttention`` or ``torch.nn.TransformerEncoder``.

    Masking conventions
    -------------------
    * ``dtype=torch.bool``  (default): ``True``  → position is masked out.
    * ``dtype=torch.float``: ``-inf`` → position is masked out; ``0.0`` → attend.

    Attention rules
    ---------------
    * **Residue positions** (rows present in the TSV) attend only to their
      listed neighbors and, when *include_self* is ``True``, to themselves.
    * **Non-residue positions** (BOS and EOS tokens, which are absent from the
      TSV) are left fully unmasked: they attend to all positions.

    Parameters
    ----------
    tsv_path:
        Path to the neighbor TSV written with ``--esm3-offsets``.
        Row format: ``<res_esm3_idx>\\t<nbr_esm3_idx>\\t...``
    seq_len:
        Total sequence length including all BOS and EOS tokens.  When
        omitted it is inferred as ``max_index + 2``, which correctly covers
        the trailing EOS for both single-chain and multi-chain assemblies
        (ESM3 places EOS immediately after the last residue of each chain).
    include_self:
        Whether each residue attends to itself (default ``True``).
    dtype:
        Output tensor dtype.  Use ``torch.bool`` for a boolean mask or
        ``torch.float`` for a float additive mask.

    Returns
    -------
    torch.Tensor
        Shape ``(seq_len, seq_len)``.

    Examples
    --------
    Single chain, 5 residues, ESM3 layout: BOS[0] res[1-5] EOS[6]

    >>> mask = neighbors_tsv_to_src_mask("1abc_A.tsv")
    >>> mask.shape
    torch.Size([7, 7])

    Assembly of two chains (N=3, N=4), ESM3 layout:
    BOS[0] res[1-3] EOS[4] BOS[5] res[6-9] EOS[10]

    >>> mask = neighbors_tsv_to_src_mask("1abc.tsv")
    >>> mask.shape
    torch.Size([11, 11])
    """
    neighbors: dict[int, list[int]] = {}
    max_index = 0

    with open(tsv_path) as f:
        for line in f:
            tokens = line.rstrip("\n").split("\t")
            if not tokens or tokens == [""]:
                continue
            indices = [int(t) for t in tokens]
            res_idx, nbrs = indices[0], indices[1:]
            neighbors[res_idx] = nbrs
            max_index = max(max_index, res_idx, *nbrs) if nbrs else max(max_index, res_idx)

    if seq_len is None:
        # max_index is the last residue; EOS sits one slot after it
        seq_len = max_index + 2

    masked_val: bool | float = True if dtype == torch.bool else float("-inf")
    clear_val:  bool | float = False if dtype == torch.bool else 0.0

    mask = torch.full((seq_len, seq_len), masked_val, dtype=dtype)

    # BOS / EOS rows: fully unmasked (attend to all positions)
    residue_positions = set(neighbors.keys())
    for i in range(seq_len):
        if i not in residue_positions:
            mask[i, :] = clear_val  # type: ignore[call-overload]

    # Residue rows: unmask neighbors (and optionally self)
    for res_idx, nbrs in neighbors.items():
        if res_idx >= seq_len:
            continue
        attend_to = set(nbrs)
        if include_self:
            attend_to.add(res_idx)
        for j in attend_to:
            if j < seq_len:
                mask[res_idx, j] = clear_val  # type: ignore[call-overload]

    return mask
