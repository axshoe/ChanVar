"""
validate_bootstrap.py
----------------------
Computes:
1. Primary AUC with 95% bootstrap CI (FoldX enabled, f6 excluded)
2. Paired bootstrap test for FoldX improvement significance
3. 5-fold cross-validation

Run after: chanvar batch with and without FoldX on clinvar_validation.csv
Uses hardcoded results from the two batch runs.

Usage: python validate_bootstrap.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# ── FoldX-enabled CPS scores (from results\clinvar_foldx_real\batch_summary.json)
RESULTS_FX = [
  ("R192Q",0.8159,1),("T665M",0.7227,1),("I1809L",0.6174,1),("R1660H",0.7187,1),
  ("G293R",0.7664,1),("Y1383C",0.7388,1),("S218L",0.8167,1),("R582Q",0.6433,1),
  ("I1708T",0.7051,1),("R1345Q",0.7155,1),("T500M",0.6184,1),("R1663Q",0.7275,1),
  ("R1666W",0.7181,1),("H253Y",0.6473,1),("V1392M",0.6835,1),("D302N",0.8129,1),
  ("A712T",0.7295,1),("R1348Q",0.7245,1),("S1798L",0.6377,1),("D1643N",0.7396,1),
  ("A1507D",0.7208,1),("P1360Q",0.7421,1),("P1352L",0.7600,1),("R1348W",0.7232,1),
  ("E532K",0.6095,1),("D1633N",0.5929,1),("R1358W",0.7225,1),("R1351L",0.7068,1),
  ("I711M",0.8339,1),("G1754R",0.6962,1),("A1807S",0.6219,1),("E1263K",0.6568,1),
  ("R279C",0.7183,1),("G297R",0.7900,1),("E147K",0.6188,1),("S1798P",0.7022,1),
  ("C272R",0.7463,1),("R1666P",0.7696,1),("I613M",0.6632,1),("I711V",0.8198,1),
  ("V215A",0.7350,1),("S615N",0.7174,1),("V1455M",0.7085,1),("R1351Q",0.7046,1),
  ("T1355N",0.7342,1),("E667K",0.7098,1),("E101K",0.7278,1),("S1468L",0.7429,1),
  ("P202L",0.7180,1),("G676R",0.7789,1),("R1660C",0.7855,1),("V1392A",0.7028,1),
  ("G539R",0.7275,1),("V1808I",0.5611,1),("C272Y",0.7502,1),("D302H",0.8623,1),
  ("A1507T",0.6780,1),("L1344P",0.7544,1),("V713M",0.7353,1),("Y62H",0.6896,1),
  ("Y62C",0.7300,1),
  ("A453T",0.6885,0),("E917D",0.5746,0),("E992V",0.5824,0),("G1104S",0.6287,0),
  ("E731A",0.5461,0),("P913S",0.5284,0),("P1010A",0.5700,0),("P1137A",0.5375,0),
  ("R893Q",0.5723,0),("G266S",0.6202,0),("G2023S",0.5861,0),("A21V",0.5994,0),
  ("R2195Q",0.5198,0),("Q1153E",0.5623,0),("R1234H",0.5542,0),("E2021K",0.5031,0),
  ("A2431T",0.5043,0),("D1102N",0.5354,0),("G2425D",0.5888,0),("P1103L",0.5441,0),
  ("G2261D",0.5763,0),("I1631V",0.5562,0),("A920G",0.6052,0),("P896L",0.5891,0),
  ("H2480Q",0.5116,0),("G876S",0.5988,0),("P2421L",0.5212,0),("P2421A",0.5232,0),
  ("A2443V",0.5739,0),("D800E",0.5852,0),("E842K",0.5309,0),("G1151S",0.6187,0),
]

# ── No-FoldX CPS scores (from results\clinvar_full\batch_summary.json)
RESULTS_NOFX = [
  ("R192Q",0.8137,1),("T665M",0.7139,1),("I1809L",0.6519,1),("R1660H",0.6831,1),
  ("G293R",0.7495,1),("Y1383C",0.6854,1),("S218L",0.8496,1),("R582Q",0.6444,1),
  ("I1708T",0.6492,1),("R1345Q",0.6882,1),("T500M",0.6575,1),("R1663Q",0.7196,1),
  ("R1666W",0.7094,1),("H253Y",0.6556,1),("V1392M",0.6556,1),("D302N",0.7744,1),
  ("A712T",0.6937,1),("R1348Q",0.6882,1),("S1798L",0.6685,1),("D1643N",0.6809,1),
  ("A1507D",0.6556,1),("P1360Q",0.7003,1),("P1352L",0.7235,1),("R1348W",0.7094,1),
  ("E532K",0.6378,1),("D1633N",0.6264,1),("R1358W",0.7094,1),("R1351L",0.7411,1),
  ("I711M",0.8039,1),("G1754R",0.6616,1),("A1807S",0.6407,1),("E1263K",0.6378,1),
  ("R279C",0.6805,1),("G297R",0.7299,1),("E147K",0.6489,1),("S1798P",0.6440,1),
  ("C272R",0.6805,1),("R1666P",0.7118,1),("I613M",0.6556,1),("I711V",0.7753,1),
  ("V215A",0.7431,1),("S615N",0.7321,1),("V1455M",0.6556,1),("R1351Q",0.7196,1),
  ("T1355N",0.7118,1),("E667K",0.7047,1),("E101K",0.6958,1),("S1468L",0.6796,1),
  ("P202L",0.7431,1),("G676R",0.7181,1),("R1660C",0.7382,1),("V1392A",0.6556,1),
  ("G539R",0.6616,1),("V1808I",0.6137,1),("C272Y",0.6854,1),("D302H",0.8052,1),
  ("A1507T",0.6496,1),("L1344P",0.7118,1),("V713M",0.7235,1),("Y62H",0.6259,1),
  ("Y62C",0.6557,1),
  ("A453T",0.6354,0),("E917D",0.5556,0),("E992V",0.5556,0),("G1104S",0.5667,0),
  ("E731A",0.5556,0),("P913S",0.5551,0),("P1010A",0.5556,0),("P1137A",0.5667,0),
  ("R893Q",0.5444,0),("G266S",0.5667,0),("G2023S",0.5519,0),("A21V",0.5259,0),
  ("R2195Q",0.5296,0),("Q1153E",0.5667,0),("R1234H",0.5396,0),("E2021K",0.5341,0),
  ("A2431T",0.5348,0),("D1102N",0.5376,0),("G2425D",0.5519,0),("P1103L",0.5667,0),
  ("G2261D",0.5519,0),("I1631V",0.5396,0),("A920G",0.5667,0),("P896L",0.5667,0),
  ("H2480Q",0.5407,0),("G876S",0.5667,0),("P2421L",0.5519,0),("P2421A",0.5519,0),
  ("A2443V",0.5519,0),("D800E",0.5667,0),("E842K",0.5489,0),("G1151S",0.5667,0),
]

W_TOTAL = 14.0
W_F6 = 1.5
W_NO_F6 = W_TOTAL - W_F6

labels = np.array([r[2] for r in RESULTS_FX])
f6v = labels.astype(float)

s_fx   = (np.array([r[1] for r in RESULTS_FX])   * W_TOTAL - W_F6*f6v) / W_NO_F6
s_nofx = (np.array([r[1] for r in RESULTS_NOFX]) * W_TOTAL - W_F6*f6v) / W_NO_F6

auc_fx   = roc_auc_score(labels, s_fx)
auc_nofx = roc_auc_score(labels, s_nofx)
delta    = auc_fx - auc_nofx

np.random.seed(42)
n = len(labels)
N_BOOT = 10000

# Primary AUC CI
boot_fx = []
for _ in range(N_BOOT):
    idx = np.random.choice(n, n, replace=True)
    try: boot_fx.append(roc_auc_score(labels[idx], s_fx[idx]))
    except: pass
boot_fx = np.array(boot_fx)

# Paired delta CI
boot_deltas = []
for _ in range(N_BOOT):
    idx = np.random.choice(n, n, replace=True)
    try:
        d = roc_auc_score(labels[idx], s_fx[idx]) - roc_auc_score(labels[idx], s_nofx[idx])
        boot_deltas.append(d)
    except: pass
boot_deltas = np.array(boot_deltas)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = [roc_auc_score(labels[ti], s_fx[ti])
           for _, ti in skf.split(s_fx.reshape(-1,1), labels)
           if len(np.unique(labels[ti])) == 2]

print("=" * 60)
print("CHANVAR VALIDATION STATISTICS")
print("=" * 60)
print(f"n = {n} (61 P/LP, 32 B/LB, >=2 ClinVar stars)")
print()
print(f"Primary AUROC (FoldX, f6 excl):  {auc_fx:.4f}")
print(f"Bootstrap 95% CI:                [{np.percentile(boot_fx,2.5):.4f}, {np.percentile(boot_fx,97.5):.4f}]")
print(f"5-fold CV mean +/- SD:           {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
print()
print(f"No-FoldX AUROC (f6 excl):        {auc_nofx:.4f}")
print(f"Delta (FoldX improvement):       {delta:+.4f}")
print(f"Bootstrap 95% CI of delta:       [{np.percentile(boot_deltas,2.5):.4f}, {np.percentile(boot_deltas,97.5):.4f}]")
print(f"P(delta <= 0) one-sided:         {np.mean(boot_deltas <= 0):.4f}")
print()
if np.percentile(boot_deltas, 2.5) > 0:
    print("RESULT: FoldX improvement is statistically significant (CI excludes 0)")
elif np.mean(boot_deltas <= 0) < 0.10:
    print("RESULT: FoldX improvement is a non-significant trend (p < 0.10, CI includes 0)")
else:
    print("RESULT: FoldX improvement is not statistically significant")
