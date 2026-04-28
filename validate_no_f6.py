"""
Honest validation: CPS excluding f6 (ClinVar prior) to avoid circular evaluation.
Run after: chanvar batch --input clinvar_validation.csv --output results/clinvar_full/
"""
import json, csv, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

W_TOTAL = 14.0
W_F6 = 1.5

with open('results/clinvar_full/batch_summary.json', encoding='utf-8') as f:
    results = json.load(f)

gt = {}
with open('clinvar_validation.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        sig = row['clinvar_sig'].lower()
        if 'pathogenic' in sig: gt[row['aa_change']] = 1
        elif 'benign' in sig:   gt[row['aa_change']] = 0

matched = [(r['cps'], gt[r['aa_change']]) for r in results if r['aa_change'] in gt]
cps_full   = np.array([m[0] for m in matched])
labels     = np.array([m[1] for m in matched])
f6_vals    = labels.astype(float)  # P/LP->1.0, B/LB->0.0
cps_no_f6  = (cps_full * W_TOTAL - W_F6 * f6_vals) / (W_TOTAL - W_F6)

auc_full  = roc_auc_score(labels, cps_full)
auc_no_f6 = roc_auc_score(labels, cps_no_f6)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = [roc_auc_score(labels[ti], cps_no_f6[ti])
           for _, ti in skf.split(cps_no_f6.reshape(-1,1), labels)
           if len(np.unique(labels[ti])) == 2]

print(f"n = {len(matched)} ({sum(labels)} P/LP, {len(labels)-sum(labels)} B/LB)")
print(f"AUC with f6 (circular, do not use as primary result): {auc_full:.4f}")
print(f"AUC without f6 (independent):                        {auc_no_f6:.4f}")
print(f"5-fold CV AUC without f6:                            {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")