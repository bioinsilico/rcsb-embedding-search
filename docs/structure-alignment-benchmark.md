# Structure-alignment benchmark for sequence-level alignments

## Why

The second stage of the FoldMatch search pipeline aligns a query against each candidate
with Smith-Waterman over BLOSUM62 and reports identity, coverage and an E-value. For remote
homologs the result is correct as an alignment but inconsistent with the structures: PDB
chains 3ZI1.A and 1ZSW.A superimpose at TM-score 0.80 while sharing about 13% identity, and
the BLOSUM62 alignment aligns a different set of residues than the superposition does.

These tools measure that gap, and how far different scoring schemes close it, using only
sequence-level dynamic programming. They exist to answer one design question: **is it worth
training a model that predicts Foldseek 3Di states from ESM3 embeddings, so that queries
without a structure can be aligned structurally?**

## What is measured

Each structure contributes three index-aligned arrays, one entry per residue with a CA atom:
the amino acid sequence, the Foldseek 3Di string (from `biotite.structure.alphabet.to_3di`,
a port of the Foldseek encoder), and the CA coordinates. Alignments are then scored with a
substitution matrix over the combined (amino acid, 3Di) alphabet, so one dynamic-programming
call can weight sequence and structure signal in any proportion.

Schemes compared (`SCHEMES` in `src/scripts/structure_alignment.py`):

| Name | Scoring | Mode |
|---|---|---|
| `blosum_sw` | BLOSUM62, gaps 11/1 | local (what the pipeline does today) |
| `blosum_semi` | BLOSUM62, gaps 11/1 | semi-global, free end gaps |
| `3di_aa_sw` | 1.4 × BLOSUM62 + 2.1 × 3Di, gaps 10/1 | local (Foldseek's combination) |
| `3di_aa_semi` | same | semi-global |
| `3di_sw` | 3Di only | local |

Two quality measures:

* **TM-score of the implied superposition.** Superimpose the two chains using only the
  residue pairs the alignment proposes, and take the TM-score. This needs no reference
  alignment (the idea behind `USalign -I`) and agrees with US-align to within 0.001.
* **Residue-pair precision / recall / F1** against a US-align reference alignment, exact
  and allowing a ±4 residue shift. The shift-tolerant number separates "aligned the wrong
  region" from "aligned the right region, off by a residue or two".

Results are stratified by *global* sequence identity (identical residues over the shorter
chain, from a semi-global alignment). Identity measured over a local alignment's own region
is biased upwards, because a local alignment keeps only the best-matching core.

## Running

US-align is needed for the reference-based measures:

```bash
git clone --depth 1 https://github.com/pylelab/USalign.git && cd USalign && make
# macOS: if the C++ standard headers are not on the default search path
SDK=$(xcrun --show-sdk-path)
clang++ -O3 -ffast-math -isysroot $SDK -cxx-isystem $SDK/usr/include/c++/v1 -o USalign USalign.cpp
```

```bash
cd src/scripts

# one pair, any PDB/mmCIF files (optionally gzipped)
python structure_alignment_benchmark.py pair \
    --query /data/pdb/3ZI1.cif.gz --query-chain A \
    --target /data/pdb/1ZSW.cif  --target-chain A --usalign ./USalign

# many pairs from a TM-score table: id1,id2,tm  or  id1,id2,tm1,tm2
python structure_alignment_benchmark.py benchmark \
    --pairs /data/cath_23M/cath_23M.csv --structures /data/cath_23M/pdb \
    --limit 400 --usalign ./USalign --output cath_benchmark.json

python structure_alignment_benchmark.py benchmark \
    --pairs /data/scop-zenodo/TMfast.dual.csv --structures /data/scop-zenodo/pdb \
    --limit 400 --line-rate 0.01 --usalign ./USalign

# how accurate must a predicted query 3Di be?
python structure_alignment_benchmark.py sweep \
    --pairs /data/cath_23M/cath_23M.csv --structures /data/cath_23M/pdb \
    --limit 200 --usalign ./USalign
```

`--line-rate` skips input lines while reading, which keeps the 115 M-row SCOP table cheap.
`--min-tm` (default 0.7) selects structurally similar pairs, and `--limit` how many are
evaluated.

## Results

Measured on 400 CATH domain pairs and 167 SCOP40 pairs with reference TM ≥ 0.7 (median
US-align TM 0.76), 60–600 residues, plus the 3ZI1.A / 1ZSW.A chain pair.

**Median residue-pair F1 against US-align (400 CATH pairs):**

| Sequence identity | n | `blosum_sw` | `blosum_semi` | `3di_aa_sw` | `3di_sw` |
|---|---|---|---|---|---|
| all | 400 | 0.310 | 0.274 | **0.632** | 0.618 |
| < 10% | 127 | 0.000 | 0.000 | **0.497** | 0.465 |
| 10–20% | 61 | 0.194 | 0.000 | **0.569** | 0.573 |
| 20–30% | 185 | 0.528 | 0.538 | **0.706** | 0.683 |
| ≥ 30% | 27 | 0.792 | 0.798 | **0.863** | 0.796 |

Allowing a ±4 shift, the same comparison reads 0.429 → 0.867 overall and 0.000 → 0.822
below 10% identity. Median TM-score of the implied superposition goes from 0.237 to 0.606,
about 80% of the reference TM against 32% today. SCOP40 agrees: median F1 0.502 → 0.689,
and 0.000 → 0.440 below 10% identity.

**3ZI1.A vs 1ZSW.A** (US-align TM 0.798 over 263 pairs):

| Scheme | TM | Aligned | Identity | F1 | F1 (±4) |
|---|---|---|---|---|---|
| `blosum_sw` | 0.109 | 35 | 0.37 | 0.188 | 0.228 |
| `3di_aa_sw` | 0.647 | 243 | 0.23 | 0.632 | 0.858 |
| `3di_sw` | 0.701 | 251 | 0.12 | 0.685 | 0.895 |

**How accurate must a predicted 3Di be?** Median F1 over 200 CATH pairs, degrading only the
query's 3Di (the target keeps exact codes, as in deployment):

| Scheme | Error type | 100% | 90% | 80% | 65% | 50% |
|---|---|---|---|---|---|---|
| `3di_aa_sw` | confusion-structured | 0.634 | 0.637 | 0.612 | **0.609** | 0.554 |
| `3di_aa_sw` | uniform random | 0.634 | 0.572 | 0.514 | 0.394 | 0.305 |
| `3di_sw` | confusion-structured | 0.629 | 0.625 | 0.572 | 0.544 | 0.504 |
| `3di_sw` | uniform random | 0.629 | 0.553 | 0.444 | 0.308 | 0.209 |

"Confusion-structured" draws wrong states in proportion to the 3Di substitution matrix, so
mistakes land on geometrically similar states, the way real predictors fail. "Uniform" is a
pessimistic floor.

## What this implies for a 3Di head

* **A 65%-accurate head keeps 96% of the exact-3Di F1** (0.609 vs 0.634), and 87% of it
  below 20% identity. Published heads reach that: ProstT5's CNN and Johnson et al.'s ESM-2
  head report 41–65% and 64.4% per-residue accuracy.
* **Error structure matters more than error rate.** At the same 65% accuracy, uniform errors
  drop F1 to 0.394 while confusion-structured errors hold 0.609. The training loss should be
  confusion-aware (weighted by the 3Di substitution matrix), and per-residue accuracy alone
  is the wrong thing to report.
* **Keep amino acids in the score.** With exact 3Di the two are equal, but under degradation
  `3di_aa_sw` beats `3di_sw` at every level: the amino acid term stabilizes a noisy
  structural signal.
* **Accuracy beyond ~80% buys little** (0.612 vs 0.634 at 100%). Calibration and masking
  uncertain positions are the better investment.
* **Local vs global barely matters once 3Di is in the score** (0.606 vs 0.611); it matters
  for BLOSUM62, and in opposite directions (semi-global raises TM but lowers F1 by
  over-aligning).
* **Downstream filters change meaning.** Median coverage rises from 0.37 to 0.89 while
  identity falls from 0.32 to 0.23: a structurally correct alignment spans the whole domain
  and therefore reports *lower* identity. Identity and coverage thresholds tuned for
  BLOSUM62 alignments will not transfer, and E-value statistics calibrated for BLOSUM62
  11/1 do not apply to a 3Di+AA score at all.

## Caveats

* The tables above use **exact** 3Di on both sides, which is the ceiling; the sweep
  estimates the cost of predicting it.
* The sweep model makes errors independently per residue. A real head makes correlated
  mistakes over whole regions (loops, disordered segments), which is likely worse.
* 3Di comes from biotite's port of the Foldseek encoder, not the Foldseek binary; the two
  have not been compared letter-for-letter here.
* Foldseek encodes residues it cannot describe (segment ends, missing backbone atoms, no
  structural neighbour) as `d`, which is also a real state. `Chain.invalid_3di_fraction`
  reports the share; it was 11% for 3ZI1.A. Training labels must mask these.
* CATH and SCOP entries are single domains. The one full-chain pair tested behaved the same.
