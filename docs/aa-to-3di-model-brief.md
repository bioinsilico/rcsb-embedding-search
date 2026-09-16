# Brief: a model that predicts 3Di from amino acid sequence

Handoff for the development of training scripts. Everything below was established in an
earlier session; the benchmark that produced the numbers is in this repository
(`src/scripts/structure_alignment*.py`, described in
[structure-alignment-benchmark.md](structure-alignment-benchmark.md)).

## 1. The problem this model solves

FoldMatch searches in two stages: an embedding prefilter (FAISS over chain vectors built
from ESM3 residue embeddings), then a pairwise alignment of the query against each candidate
with Smith-Waterman over BLOSUM62, which produces the identity, coverage, E-value and the
alignment shown to users.

On remote homologs the second stage is structurally wrong. PDB chains 3ZI1.A and 1ZSW.A
superimpose at TM-score 0.80, but BLOSUM62 aligns 35 residues of which 19% match the
structural alignment. The prefilter finds the right protein; the alignment then misrepresents
it.

**The decision taken:** score stage 2 with amino acids *plus* the Foldseek 3Di structural
alphabet, selectable so the current behaviour remains available. Targets are PDB entries and
computed structure models, so their 3Di can be computed exactly from coordinates, offline,
at 1 byte per residue. **Queries are often sequence-only, so their 3Di must be predicted —
that is the model to build.**

**Why predict it from ESM3 embeddings:** FoldMatch already runs ESM3 on the query for stage 1.
A head on those embeddings costs ~2 ms per 300 residues against 1.16 s for the ESM3 forward
itself (measured, 1 CPU thread, M1 Max). Any other predictor (ProstT5, ESM-2 3B) means a
second large model per query, and deployment is CPU-only today with a <10 s budget.

## 2. Evidence that it is worth building

Measured on 400 CATH domain pairs and 167 SCOP40 pairs (reference TM ≥ 0.7, median US-align
TM 0.76), using **exact** 3Di on both sides, i.e. the ceiling. Median residue-pair F1 against
US-align reference alignments:

| Sequence identity | n | BLOSUM62 local (today) | 3Di + AA |
|---|---|---|---|
| all | 400 | 0.310 | **0.632** |
| < 10% | 127 | 0.000 | **0.497** |
| 10–20% | 61 | 0.194 | **0.569** |
| ≥ 30% | 27 | 0.792 | **0.863** |

Allowing a ±4 residue shift: 0.429 → 0.867. Median TM-score of the superposition the
alignment implies: 0.237 → 0.606, which is 80% of the reference TM versus 32% today.

**How accurate must the predicted 3Di be?** Degrading only the query's 3Di (targets keep
exact codes, as in deployment), median F1 over 200 CATH pairs:

| Error type | 100% | 90% | 80% | 65% | 50% |
|---|---|---|---|---|---|
| confusion-structured | 0.634 | 0.637 | 0.612 | **0.609** | 0.554 |
| uniform random | 0.634 | 0.572 | 0.514 | 0.394 | 0.305 |

Three conclusions that shape the model:

1. **65% per-residue accuracy suffices** — it retains 96% of the exact-3Di F1 (87% below 20%
   identity). Published heads reach this: ProstT5's CNN reports 41–65%, Johnson et al.'s
   ESM-2 3B head 64.4%.
2. **Error structure matters more than error rate.** At the same 65%, uniform errors give
   0.394 and confusion-structured errors 0.609. Optimize the *kind* of mistake, not just Q20.
3. **Keep amino acids in the score.** With exact 3Di, 3Di-only and 3Di+AA are equal; under
   degradation 3Di+AA wins at every level (0.609 vs 0.544 at 65%).

## 3. Model specification

**Input.** ESM3-open (`esm3_sm_open_v1`, d_model 1536, 48 blocks) per-residue embeddings from
a **sequence-only** forward pass — structure track masked. This matters: embeddings from
structure-conditioned passes (the `build structures` path, or any cached `.pt` residue
tensors) come from a different input distribution and will not transfer.

* Tensors are `(L+2, 1536)` float32 including BOS/EOS; the head must use `[1:-1]`.
* The embedding is the last block's output **before** the final LayerNorm. ESM3's own heads
  read the normalized output, so apply a LayerNorm at the head input, and ablate the layer
  choice (final vs intermediate blocks).
* ESM3's ss8 logits (11 classes) come free from the same forward pass and are worth testing
  as an extra input feature.

**Output.** 20 3Di states as a probability distribution, not an argmax. The soft profile is
what the alignment scorer consumes: for query position *i* and target state *s*, the score is
`Σ_k p_i(k) · M3Di[k, s]`, combined with the amino-acid term. Also emit a per-residue
confidence (max probability or entropy) so low-confidence positions can be masked.

**Architecture.** Start with a 2-layer 1D CNN (kernel 5–7), which is what the published heads
use, then ablate a linear probe and a small transformer head (d=256, 2 layers). No source
found shows attention heads beating CNNs on frozen transformer embeddings for per-residue
structural labels, and NetSurfP-3.0 explicitly found transformer layers "sub-optimal" against
a BiLSTM. Head cost is negligible either way (~2 ms vs ESM3's 1.16 s), so the ablation is
cheap — it just should not be the starting point.

If the head plateaus, the one proven lever is trainable capacity at the top of the backbone:
Johnson et al. went from ~58% to 64.4% by unfreezing ESM-2's last layer. **FoldMatch cannot
fine-tune ESM3** without changing the stage-1 chain vectors and invalidating the FAISS index.
An untested alternative is a trainable *copy* of ESM3's last block on top of frozen
embeddings, costing ~2% more query compute.

**Loss.** Cross-entropy weighted by the 3Di substitution matrix, so a swap to a geometrically
distant state is penalized more than to a neighbouring one — this is what the sweep says
matters most. Johnson et al. weight by the matrix diagonal; going to the full matrix is
untested. Optionally label smoothing and pLDDT weighting (both in DessimozLab/ESM3di).

**Calibration.** No published 3Di predictor reports it, and a soft profile depends on it.
Measure NLL, ECE and top-3 accuracy; consider temperature scaling.

## 4. Training data

**Labels.** `biotite.structure.alphabet.to_3di` (biotite ≥ 1.5, already in this repo's
environment) is a port of the Foldseek encoder with bundled weights, BSD-3 licensed, so no
Foldseek (GPLv3) dependency is needed. Confirm it matches the Foldseek binary letter-for-letter
on a sample first; no published large-scale comparison exists.

**Masking is essential.** Foldseek encodes residues it cannot describe as `d`, which is also a
real state, so they are indistinguishable afterwards. This was 11% of residues in 3ZI1.A and
100% of one 113-residue chain. Exclude from the loss:

* the first and last residue of each observed segment;
* residues missing N, CA or C, and their neighbours (invalidity propagates);
* chain breaks — Foldseek does not detect them, so detect them yourself (CA–CA > ~4.2 Å);
* for predicted models, residues with pLDDT < 70 (the SaProt/ProstT5 convention).

Also note target 3Di exists only for **observed** residues, so it will not line up index-for-index
with a SEQRES-derived sequence.

**Sources available locally** (paths from the earlier session's machine):

| Data | Path | Size |
|---|---|---|
| PDB mmCIF mirror | `/Users/joan/data/pdb` | 32,953 entries |
| CATH domains | `/Users/joan/data/cath_23M/pdb` | 27,265 structures |
| CATH pairs + TM-scores | `/Users/joan/data/cath_23M/cath_23M.csv` | 22.9 M pairs |
| SCOP domains | `/Users/joan/data/scop-zenodo/pdb` | 15,176 structures |
| SCOP pairs + TM-scores | `/Users/joan/data/scop-zenodo/TMfast.dual.csv` | 115 M pairs |

For scale, AFDB Foldseek structural clusters (2.28 M non-singleton representatives, CC-BY)
are the standard source. Train on a mix of PDB chains and high-pLDDT models: 3Di from AF2
models is not distributed like 3Di from experimental structures.

**Scale needed is modest.** ProstT5's CNN was trained on roughly 10k chains (the NetSurfP-2.0
set) and still reached SCOPe40 superfamily AUC 0.47. Start at 10⁴–10⁵ chains.

**Splits.** Split by structural cluster (Foldseek clustering), not randomly within a sequence
cluster. Exclude the folds used by the evaluation benchmarks. Note that Foldseek's 3Di
alphabet and substitution matrix were themselves trained on all of SCOPe40, so SCOPe40
results carry that caveat, and ESM3's pretraining set cannot be controlled.

## 5. Practicalities

* **Embedding cache:** fp16 at 1536-d is 3,072 B per residue, 0.92 MB per 300-residue chain,
  so 92 GB per 100k chains. The head is ~0.2% of ESM3's cost, so caching pays off after the
  first epoch.
* **Generation cost:** measured 0.22 s per 300-residue chain on an M1 Max GPU (MPS) and
  1.16 s on one CPU thread; 100k chains ≈ 6 h on that Mac. On a data-centre GPU with batching
  and bf16, roughly 2–4 GPU-hours per million chains (FLOP-based estimate).
* **Batching gotcha:** `ESM3.logits()` calls forward with `sequence_id=None`, which builds no
  attention mask, so padded batches let real residues attend to padding. Call
  `esm3.forward(..., sequence_id=...)` directly when batching.
* **Getting ss8/structure logits:** `forward_and_sample` discards them. Copy its default-fill
  step and call `esm.utils.generation._batch_forward` to obtain embeddings and all logits from
  one pass.

## 6. Evaluation protocol

Use the benchmark in this repository rather than per-residue accuracy:

```bash
cd src/scripts
python structure_alignment_benchmark.py benchmark \
    --pairs /data/cath_23M/cath_23M.csv --structures /data/cath_23M/pdb \
    --limit 400 --usalign ./USalign
```

Report, stratified by global sequence identity: residue-pair F1 against US-align (exact and
±4), and the TM-score of the superposition the alignment implies. **Target: match the 65%
simulated point — F1 ≈ 0.61 overall and ≈ 0.44 below 20% identity.** Today's baseline is 0.310
and 0.000. Also report per-residue accuracy and calibration, but do not optimize for them.

To evaluate the head end to end, substitute predicted query 3Di for the exact string in the
`Chain` returned by `load_chain` (the `sweep` sub-command already does this with simulated
predictions, so the wiring exists).

## 7. Integration back into FoldMatch (rcsb-embedding-model)

Not part of this repository's work, but it constrains the design:

* Target 3Di goes in the SQLite sequence store alongside the sequence (new column), keeping
  `len(3di) == length`.
* The query head runs in `ChainCompleteModule.predict_step`, next to the stage-1 embedding, so
  there is one ESM3 pass.
* Stage 2 already supports position-specific scoring: `_positional_query` in
  `src/foldmatch/search/alignment.py` builds an L × alphabet substitution matrix over query
  positions, which is exactly where a soft 3Di profile plugs in.
* **E-values must be recalibrated.** The current λ/K are valid only for BLOSUM62 11/1, and a
  3Di+AA score is a different scoring system (one study found Foldseek's E-values off by ~5
  orders of magnitude).
* **Filters change meaning.** Structurally correct alignments span more and report lower
  identity: median coverage 0.37 → 0.89, identity 0.32 → 0.23. The current defaults
  (identity ≥ 0.3, coverage ≥ 0.8) would discard exactly the remote homologs this fixes.
* Structure-built databases have no sequence store, and `query structure` has no stage 2 at
  all; both need plumbing before this reaches structure queries.

## 8. Licensing

ESM3-open weights are now MIT (`biohub/esm3-sm-open-v1`, commit 47f0545, 2026-06-04). Under
the earlier Cambrian Non-Commercial License, a model trained on ESM3 outputs was explicitly a
derivative work (non-commercial only, naming requirements). Record the provenance commit of
the weights used and get a written decision before distributing the head.

## 9. Open questions

* No published 3Di head on ESM3 or ESM C exists, so how ESM3-open embeddings compare with
  ProstT5 or ESM-2 3B for this task is unknown.
* Whether the pre-norm final-block embedding is the best layer is untested.
* Whether a soft 3Di profile beats argmax for *static* structure 3Di is untested (the one
  precedent, ProtProfileMD, used MD-derived profiles and reported small gains).
* A full-matrix confusion-aware loss has not been tried by anyone.
* The sweep models errors as independent per residue; real heads make correlated mistakes over
  loops and disordered regions, which is likely worse.

## 10. Key references

* **ProstT5** (Heinzinger et al. 2024) — AA↔3Di translation; the 2-layer CNN head Foldseek
  ships; 41–65% Q20 yet SCOPe40 superfamily AUC 0.47 vs 0.49 for experimental 3Di.
* **Johnson, Peshwa & Sun** (eLife 2024) — ESM-2 3B → CNN 3Di head, 64.4%; sensitivity drops
  when query and database 3Di come from different methods.
* **Foldseek** (van Kempen et al. 2024) — the 3Di alphabet, its substitution matrix, and the
  1.4/2.1 amino-acid/3Di weighting used here.
* **Phold** — predicted query 3Di against a structure-derived database, the closest published
  setup to ours; masking low-confidence query residues helped.
* **TEA** (Pantolini et al. 2025) — a learned 20-letter alphabet from ESM-2 embeddings;
  complementary to 3Di.
* **SaProt** — masks 3Di from low-pLDDT regions, corroborating the masking rules above.
* **DessimozLab/ESM3di** — LoRA fine-tuning plus linear/CNN/transformer 3Di heads; useful code
  reference, no published accuracy.
