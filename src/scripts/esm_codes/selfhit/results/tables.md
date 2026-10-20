## T1. P6 in-database self hits (job 60008416 E_inf), ln E decomposed

ln E = [lnK(m) + ln m + ln N] - lambda(m) S_f + lambda(m) S_r + lambda(m) Q90, N = 2,957,247. Columns: medians; nats = lambda x score; 'need/res' = S' per residue needed for E <= 1e-3; the last four columns count self hits that would reach E <= 1e-3 without S_r, without Q90, without both, or with lambda/lnK extrapolated in ln m below 32.6 aa instead of clamped.

| m | n | S_f/m | S_r | Q90 | lambda | nats S_f | nats S_r | nats Q90 | nats space | need/res | E<=1e-3 | no S_r | no Q90 | neither | lam extrap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 21-100 | 37 | 0.51 | 1.6 | 1.62 | 0.1363 | 3.1 | 0.25 | 0.24 | 4.9 | 1.88 | 3/37 | 3 | 3 | 3 | 3 |
| 101-300 | 25 | 1.31 | 7.5 | 11.87 | 0.0742 | 14.5 | 0.50 | 1.06 | 6.2 | 1.03 | 13/25 | 14 | 14 | 14 | 13 |
| 301-966 | 27 | 2.20 | 15.6 | 41.01 | 0.0426 | 54.2 | 0.68 | 1.87 | 5.7 | 0.55 | 27/27 | 27 | 27 | 27 | 27 |
| >966 | 5 | 2.44 | 19.7 | 40.97 | 0.0416 | 142.0 | 0.82 | 1.71 | 6.3 | 0.23 | 5/5 | 5 | 5 | 5 | 5 |
| 1-20 | 5 | 0.20 | 0.7 | 0.30 | 0.1878 | 0.6 | 0.13 | 0.06 | 4.3 | 3.51 | 0/5 | 0 | 0 | 0 | 0 |


## T2. Self hits by population (queries of 21-966 aa, the calibrated query domain)

Local recomputation (FoldMatch ESM3 + head on MPS, FoldMatch kernel, shipped panel). The last four columns: share of self hits with E <= 1e-3 at N = 10k sample (P6 check) (2.96e+06), PDB unique seqres (4.75e+07), PDB all protein chains (3.21e+08), UniRef50 2026_03 (1.22e+10).

| population | n | median m | S_f/m median | S_f/m P10-P90 | S_r/S_f | Q90 | mean z_ii | LC frac median | LC > 0.1 | E<=1e-3 @ 10k sample (P6 check) | E<=1e-3 @ PDB unique seqres | E<=1e-3 @ PDB all protein chains | E<=1e-3 @ UniRef50 2026_03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P6 queries (200) | 180 | 142 | 1.25 | 0.31-3.10 | 0.029 | 11.67 | 2.21 | 0.00 | 26% | 52% | 47% | 43% | 41% |
| UniRef50 random (800) | 777 | 202 | 1.67 | 0.62-3.18 | 0.023 | 17.69 | 2.67 | 0.04 | 33% | 67% | 61% | 56% | 49% |
| SCOP calibration queries | 1431 | 73 | 1.95 | 0.83-3.14 | 0.034 | 13.14 | 2.95 | 0.00 | 14% | 55% | 43% | 37% | 29% |
| PDB random unique (600) | 566 | 222 | 2.84 | 1.33-3.62 | 0.016 | 29.48 | 3.84 | 0.04 | 21% | 90% | 86% | 82% | 76% |


## T3. By query length

'need/res' = S'/m needed for E <= 1e-3 at the 10k N and at UniRef50 N. (Queries outside 21-966 aa are outside S13 and get ecal = 0; shown for completeness.)

| population | m | n | S_f/m | need/res 10k | need/res U50 | median E 10k | E<=1e-3 10k | E<=1e-3 PDB | E<=1e-3 U50 |
|---|---|---|---|---|---|---|---|---|---|
| P6 queries (200) | 1-20 | 10 | 0.17 | 3.73 | 6.51 | 48 | 0% | 0% | 0% |
| P6 queries (200) | 21-50 | 42 | 0.40 | 1.97 | 3.36 | 21 | 0% | 0% | 0% |
| P6 queries (200) | 51-100 | 32 | 0.86 | 1.57 | 2.63 | 0.45 | 16% | 6% | 0% |
| P6 queries (200) | 101-200 | 32 | 1.49 | 1.11 | 1.80 | 2.3e-05 | 62% | 53% | 44% |
| P6 queries (200) | 201-300 | 18 | 1.57 | 0.98 | 1.60 | 1.6e-06 | 72% | 67% | 50% |
| P6 queries (200) | 301-600 | 32 | 2.15 | 0.68 | 1.13 | 3e-16 | 100% | 94% | 81% |
| P6 queries (200) | 601-966 | 24 | 2.16 | 0.40 | 0.66 | 1.8e-28 | 100% | 100% | 100% |
| P6 queries (200) | >966 | 10 | 2.45 | 0.23 | 0.37 | 6.6e-56 | 100% | 100% | 100% |
| UniRef50 random (800) | 21-50 | 20 | 0.58 | 1.89 | 3.23 | 6 | 0% | 0% | 0% |
| UniRef50 random (800) | 51-100 | 133 | 1.02 | 1.51 | 2.52 | 0.15 | 14% | 7% | 1% |
| UniRef50 random (800) | 101-200 | 235 | 1.49 | 1.10 | 1.79 | 5.3e-05 | 61% | 51% | 32% |
| UniRef50 random (800) | 201-300 | 147 | 2.00 | 0.97 | 1.60 | 6.1e-09 | 88% | 82% | 62% |
| UniRef50 random (800) | 301-600 | 183 | 2.37 | 0.71 | 1.18 | 8.8e-16 | 94% | 93% | 87% |
| UniRef50 random (800) | 601-966 | 59 | 2.02 | 0.40 | 0.67 | 1.5e-25 | 93% | 92% | 90% |
| UniRef50 random (800) | >966 | 22 | 1.98 | 0.27 | 0.44 | 1.2e-42 | 95% | 95% | 95% |
| SCOP calibration queries | 21-50 | 404 | 1.63 | 1.90 | 3.24 | 0.048 | 18% | 2% | 0% |
| SCOP calibration queries | 51-100 | 601 | 1.80 | 1.57 | 2.64 | 0.0012 | 49% | 33% | 8% |
| SCOP calibration queries | 101-200 | 131 | 2.61 | 1.08 | 1.76 | 2.6e-10 | 95% | 90% | 75% |
| SCOP calibration queries | 201-300 | 108 | 2.93 | 0.97 | 1.60 | 1.2e-13 | 99% | 97% | 88% |
| SCOP calibration queries | 301-600 | 132 | 2.50 | 0.69 | 1.14 | 8.3e-17 | 99% | 99% | 94% |
| SCOP calibration queries | 601-966 | 55 | 2.00 | 0.40 | 0.66 | 5.2e-24 | 100% | 100% | 98% |
| PDB random unique (600) | 1-20 | 33 | 0.17 | 4.49 | 7.90 | 37 | 0% | 0% | 0% |
| PDB random unique (600) | 21-50 | 19 | 0.48 | 1.91 | 3.25 | 13 | 11% | 5% | 0% |
| PDB random unique (600) | 51-100 | 48 | 1.55 | 1.50 | 2.50 | 0.0016 | 46% | 33% | 12% |
| PDB random unique (600) | 101-200 | 176 | 2.46 | 1.11 | 1.81 | 3.6e-09 | 94% | 85% | 63% |
| PDB random unique (600) | 201-300 | 135 | 3.08 | 0.98 | 1.60 | 7e-15 | 99% | 98% | 95% |
| PDB random unique (600) | 301-600 | 156 | 3.39 | 0.72 | 1.19 | 7.9e-21 | 99% | 99% | 97% |
| PDB random unique (600) | 601-966 | 32 | 2.76 | 0.42 | 0.69 | 1.3e-34 | 100% | 100% | 100% |


## T4. UniRef50 (1,000 sequences, 21-966 aa): Spearman correlation with S_f/m

| feature | rho | p |
|---|---|---|
| length (ln) | +0.54 | 3.8e-72 |
| LC fraction | -0.19 | 2.8e-09 |
| TOP-IDP (disorder) | -0.20 | 2.4e-10 |
| composition entropy | +0.52 | 2.1e-68 |
| 3-mer redundancy | +0.10 | 1.5e-03 |
| BLOSUM62 internal repeat | +0.15 | 1.6e-06 |
| 'uncharacterized' | -0.55 | 3.1e-77 |
| 'fragment' | -0.07 | 3.9e-02 |
| cluster size n | +0.34 | 8.3e-27 |
| mean s_ii | +0.07 | 2.9e-02 |
| mean row mean mu_i | -0.86 | 2.4e-282 |
| mean row std sigma_i | -0.75 | 1.6e-176 |
| share of residues out-scored in their row | -0.86 | 1.8e-281 |
| off-diagonal z>1 share | +0.87 | 9.3e-294 |
| code homogeneity h = |mean code| / 8 | -0.95 | 0.0e+00 |
| mean similarity to the panel background mu_i^P | -0.80 | 1.0e-214 |


Variance of ln(S_f/m) explained (least squares, UniRef50 21-966 aa):

| predictors | R^2 |
|---|---|
| length (ln m) | 0.21 |
| homogeneity h | 0.74 |
| h + ln m | 0.76 |
| sequence features only (ln m, LC, TOP-IDP, entropy, uncharacterized, ln n) | 0.57 |
| all of the above + h | 0.82 |


## T5. The self matrix, per residue (medians over sequences of per-sequence means; 21-966 aa)

h = |mean residue code| / 8 (1 = all residues share one code; the panel's 355,626 residues give 0.282); s_ii = diagonal similarity cos(q_i, d_i); mu_i, sigma_i = mean and std of row i of the self matrix (S7); mu_i^P = mean similarity of residue i to the panel's codes; z_ii = the diagonal rowcol z; z_ii^P = the diagonal standardised against the panel background; 'z_ii<=1' = share of diagonal cells whose SW cell score z-1 is <= 0; 'beaten' = share of residues i with some j != i where s_ij > s_ii; 'offdiag+' = share of cells |i-j|>3 with z > 1 (internal repeats).

| group | n | median m | S_f/m | h | s_ii | mu_i | mu_i^P | sigma_i | z_ii | z_ii^P | z_ii<=1 | beaten | offdiag+ | S_r/S_f | Q90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UniRef50 LC = 0 | 431 | 134 | 1.58 | 0.412 | 0.519 | 0.250 | 0.099 | 0.150 | 2.56 | 2.50 | 0.02 | 0.45 | 0.060 | 0.027 | 13.50 |
| UniRef50 LC (0, 0.1] | 221 | 354 | 2.25 | 0.360 | 0.476 | 0.097 | 0.063 | 0.128 | 3.24 | 2.67 | 0.02 | 0.45 | 0.095 | 0.015 | 30.52 |
| UniRef50 LC > 0.1 | 305 | 187 | 1.22 | 0.459 | 0.467 | 0.238 | 0.110 | 0.154 | 2.19 | 2.21 | 0.11 | 0.73 | 0.063 | 0.030 | 13.38 |
| UniRef50 LC > 0.3 | 68 | 130 | 0.42 | 0.536 | 0.442 | 0.335 | 0.124 | 0.168 | 1.39 | 2.00 | 0.28 | 0.89 | 0.046 | 0.069 | 2.40 |
| UniRef50 TOP-IDP <= 0.0673 | 353 | 174 | 1.80 | 0.393 | 0.515 | 0.200 | 0.088 | 0.141 | 2.80 | 2.58 | 0.01 | 0.46 | 0.073 | 0.023 | 15.56 |
| UniRef50 TOP-IDP > 0.0998 | 360 | 188 | 1.25 | 0.449 | 0.469 | 0.243 | 0.108 | 0.153 | 2.23 | 2.22 | 0.10 | 0.68 | 0.063 | 0.030 | 14.31 |
| UniRef50 'uncharacterized' | 363 | 123 | 0.97 | 0.477 | 0.512 | 0.339 | 0.129 | 0.167 | 1.96 | 2.28 | 0.06 | 0.70 | 0.048 | 0.035 | 6.24 |
| UniRef50 annotated (not uncharacterized) | 594 | 252 | 2.10 | 0.361 | 0.487 | 0.104 | 0.059 | 0.132 | 3.10 | 2.67 | 0.02 | 0.40 | 0.087 | 0.019 | 26.32 |
| UniRef50 'fragment' | 59 | 125 | 1.31 | 0.449 | 0.505 | 0.319 | 0.127 | 0.158 | 2.31 | 2.43 | 0.03 | 0.59 | 0.053 | 0.031 | 8.85 |
| UniRef50 singleton cluster (n=1) | 615 | 156 | 1.36 | 0.435 | 0.502 | 0.263 | 0.113 | 0.152 | 2.34 | 2.38 | 0.03 | 0.59 | 0.060 | 0.028 | 12.67 |
| UniRef50 cluster n >= 10 | 66 | 310 | 2.71 | 0.302 | 0.503 | 0.056 | 0.038 | 0.126 | 3.71 | 3.12 | 0.01 | 0.22 | 0.104 | 0.015 | 36.49 |
| SCOP calibration queries | 1431 | 73 | 1.95 | 0.349 | 0.508 | 0.123 | 0.051 | 0.142 | 2.95 | 2.70 | 0.00 | 0.25 | 0.050 | 0.034 | 13.14 |
| PDB random unique (600) | 566 | 222 | 2.84 | 0.281 | 0.508 | 0.019 | 0.016 | 0.129 | 3.84 | 3.45 | 0.00 | 0.20 | 0.099 | 0.016 | 29.48 |
| all populations, h <= 0.309 | 985 | 203 | 3.07 | 0.260 | 0.516 | 0.012 | 0.004 | 0.128 | 4.07 | 3.76 | 0.00 | 0.09 | 0.101 | 0.017 | 31.89 |
| all populations, h in (0.309, 0.409] | 984 | 114 | 2.00 | 0.356 | 0.485 | 0.133 | 0.061 | 0.140 | 3.00 | 2.62 | 0.00 | 0.32 | 0.066 | 0.028 | 17.24 |
| all populations, h > 0.409 | 985 | 91 | 1.03 | 0.472 | 0.513 | 0.322 | 0.121 | 0.169 | 2.02 | 2.32 | 0.05 | 0.68 | 0.043 | 0.038 | 7.64 |


## T5b. P6 in-database self hits: the 12 weakest (21-966 aa) and the 4 strongest

| id | m | name | E (P6) | S_f/m | h | LC | TOP-IDP | mu_i | z_ii |
|---|---|---|---|---|---|---|---|---|---|
| UniRef50_A0A4V3D5N2 | 34 | Uncharacterized protein | 1.5e+02 | 0.00 | 0.67 | 0.00 | 0.23 | 0.45 | 0.73 |
| UniRef50_A0A509EDP0 | 67 | Uncharacterized protein | 1.5e+02 | 0.06 | 0.66 | 0.00 | 0.17 | 0.49 | 0.97 |
| UniRef50_A0A518B6S0 | 52 | Uncharacterized protein | 1.2e+02 | 0.05 | 0.72 | 0.00 | 0.04 | 0.50 | 0.68 |
| UniRef50_A0ABZ1TTB4 | 54 | Uncharacterized protein | 62 | 0.20 | 0.67 | 0.96 | 0.11 | 0.55 | 1.18 |
| UniRef50_A0A816GV12 | 23 | Uncharacterized protein (Fragment) | 55 | 0.19 | 0.57 | 0.00 | 0.07 | 0.42 | 1.14 |
| UniRef50_A3IHS3 | 24 | Uncharacterized protein | 53 | 0.21 | 0.56 | 0.00 | -0.09 | 0.36 | 1.14 |
| UniRef50_A0AAV4H412 | 91 | Uncharacterized protein | 49 | 0.24 | 0.62 | 0.63 | 0.10 | 0.40 | 1.09 |
| UniRef50_C1P600 | 21 | Uncharacterized protein YnbG | 49 | 0.21 | 0.52 | 0.00 | 0.02 | 0.35 | 1.18 |
| UniRef50_A0AAU9ENM7 | 46 | ATP synthase F0 subunit 8 | 47 | 0.24 | 0.63 | 0.37 | -0.14 | 0.51 | 1.23 |
| UniRef50_A0A392VFA7 | 25 | Uncharacterized protein (Fragment) | 43 | 0.25 | 0.60 | 0.00 | 0.05 | 0.44 | 1.24 |
| UniRef50_A0A5S5DUJ2 | 26 | Uncharacterized protein (Fragment) | 42 | 0.24 | 0.57 | 0.00 | -0.02 | 0.36 | 1.21 |
| UniRef50_A0ABV1YGV4 | 22 | Fido domain-containing protein | 33 | 0.37 | 0.52 | 0.00 | 0.08 | 0.25 | 1.37 |
| UniRef50_D7FVL5 | 715 | UTP--glucose-1-phosphate uridylyltransfera | 3.6e-37 | 3.14 | 0.31 | 0.15 | 0.12 | -0.01 | 4.14 |
| UniRef50_A0AAN4VXG8 | 779 | non-specific protein-tyrosine kinase | 8.6e-33 | 2.61 | 0.33 | 0.02 | 0.06 | 0.08 | 3.61 |
| UniRef50_A0A9J6RIV1 | 628 | Periplasmic chaperone PpiD | 1.5e-30 | 2.90 | 0.33 | 0.05 | 0.11 | 0.07 | 3.90 |
| UniRef50_A0AAW0WB51 | 656 | Alkaline phosphatase | 2.5e-30 | 2.90 | 0.35 | 0.02 | 0.09 | -0.07 | 3.90 |


## T6. Share of self hits at E <= 1e-3 as the database grows (E is proportional to N: every factor 10 in N costs 2.3 nats)

| population | m | n | 10k sample (P6 check) | PDB unique seqres | PDB all protein chains | UniRef50 2026_03 |
|---|---|---|---|---|---|---|
| P6 queries (200) | 21-100 | 74 | 7% | 3% | 3% | 0% |
| P6 queries (200) | 101-300 | 50 | 66% | 58% | 46% | 46% |
| P6 queries (200) | 301-966 | 56 | 100% | 96% | 93% | 89% |
| UniRef50 random (800) | 21-100 | 153 | 12% | 6% | 3% | 1% |
| UniRef50 random (800) | 101-300 | 382 | 71% | 63% | 54% | 43% |
| UniRef50 random (800) | 301-966 | 242 | 94% | 93% | 92% | 88% |
| SCOP calibration queries | 21-100 | 1005 | 37% | 21% | 13% | 5% |
| SCOP calibration queries | 101-300 | 239 | 97% | 93% | 91% | 81% |
| SCOP calibration queries | 301-966 | 187 | 99% | 99% | 98% | 95% |
| PDB random unique (600) | 21-100 | 67 | 36% | 25% | 18% | 9% |
| PDB random unique (600) | 101-300 | 311 | 96% | 91% | 86% | 77% |
| PDB random unique (600) | 301-966 | 188 | 99% | 99% | 99% | 97% |


## T7. Information per residue of a self hit: esm-codes-v1 bits/m vs BLOSUM62 bits/m (BLOSUM62: full-length self score, FoldMatch's 11/1 Karlin-Altschul parameters, no composition correction)

| population | m | n | esm-codes bits/res | BLOSUM62 bits/res | esm-codes E<=1e-3 @ U50 | BLOSUM62 E<=1e-3 @ U50 |
|---|---|---|---|---|---|---|
| P6 queries (200) | 21-100 | 74 | 0.559 | 2.14 | 0% | 100% |
| P6 queries (200) | 101-300 | 50 | 0.276 | 2.07 | 46% | 100% |
| P6 queries (200) | 301-966 | 56 | 0.168 | 2.06 | 89% | 100% |
| UniRef50 random (800) | 21-100 | 153 | 0.422 | 2.11 | 1% | 100% |
| UniRef50 random (800) | 101-300 | 382 | 0.279 | 2.08 | 43% | 100% |
| UniRef50 random (800) | 301-966 | 242 | 0.184 | 2.06 | 88% | 100% |
| SCOP calibration queries | 21-100 | 1005 | 0.609 | 2.16 | 5% | 100% |
| SCOP calibration queries | 101-300 | 239 | 0.336 | 2.07 | 81% | 100% |
| SCOP calibration queries | 301-966 | 187 | 0.185 | 2.07 | 95% | 100% |
| PDB random unique (600) | 21-100 | 67 | 0.509 | 2.11 | 9% | 100% |
| PDB random unique (600) | 101-300 | 311 | 0.350 | 2.09 | 77% | 100% |
| PDB random unique (600) | 301-966 | 188 | 0.245 | 2.07 | 97% | 100% |


## T8. Reversed-query cost by low-complexity fraction (all sets, 21-966 aa)

| LC fraction | n | S_r/S_f median (P90) | lambda S_r nats median (P90) | self hits dropping S_r lifts over 1e-3 (10k / U50) |
|---|---|---|---|---|
| 0 | 1740 | 0.033 (0.055) | 0.53 (0.78) | 44 / 27 |
| (0, 0.1] | 587 | 0.014 (0.025) | 0.62 (0.85) | 2 / 2 |
| (0.1, 0.3] | 498 | 0.026 (0.053) | 0.53 (0.84) | 9 / 10 |
| (0.3, 0.6] | 102 | 0.054 (0.178) | 0.34 (0.74) | 2 / 1 |
| > 0.6 | 27 | 0.091 (0.857) | 0.25 (0.59) | 0 / 0 |
| all | 2954 | 0.028 (0.055) | 0.54 (0.80) | 57 / 40 |


## T9. Code homogeneity of the shipped panel's strata

| panel stratum | n | median h | h > 0.409 |
|---|---|---|---|
| LC 0 | 326 | 0.306 | 23% |
| LC (0, 0.1] | 383 | 0.287 | 13% |
| LC > 0.1 | 291 | 0.381 | 40% |
| D0 (ordered) | 339 | 0.302 | 20% |
| D1 | 335 | 0.275 | 15% |
| D2 (disordered) | 326 | 0.379 | 37% |
| 30-134 aa | 128 | 0.428 | 60% |
| >= 700 aa | 125 | 0.336 | 27% |
