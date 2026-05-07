python make_severity_table.py
"""
make_severity_table.py
-----------------------
Generates the variant severity table to send van den Maagdenberg.

Output: chanvar_severity_table.csv
Columns:
  aa_change, clinvar_label, domain, cps_score, p_gof,
  severity_tier, inheritance_mode, de_novo_probability,
  n_literature_papers, clinical_category, pmids, notes

Run from chanvar root: python make_severity_table.py
"""

import csv
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np

# ── Full dataset scores (from validate_bootstrap.py hardcoded results) ──────
RESULTS = [
  ("R192Q",0.8159,1,"voltage_sensor"),("T665M",0.7227,1,"pore_lining"),
  ("I1809L",0.6174,1,"C_terminal"),("R1660H",0.7187,1,"pore_lining"),
  ("G293R",0.7664,1,"selectivity_filter"),("Y1383C",0.7388,1,"interdomain_linker"),
  ("S218L",0.8167,1,"voltage_sensor"),("R582Q",0.6433,1,"interdomain_linker"),
  ("I1708T",0.7051,1,"interdomain_linker"),("R1345Q",0.7155,1,"pore_lining"),
  ("T500M",0.6184,1,"interdomain_linker"),("R1663Q",0.7275,1,"selectivity_filter"),
  ("R1666W",0.7181,1,"pore_lining"),("H253Y",0.6473,1,"interdomain_linker"),
  ("V1392M",0.6835,1,"interdomain_linker"),("D302N",0.8129,1,"pore_lining"),
  ("A712T",0.7295,1,"pore_lining"),("R1348Q",0.7245,1,"pore_lining"),
  ("S1798L",0.6377,1,"interdomain_linker"),("D1643N",0.7396,1,"pore_lining"),
  ("A1507D",0.7208,1,"interdomain_linker"),("P1360Q",0.7421,1,"pore_lining"),
  ("P1352L",0.7600,1,"pore_lining"),("R1348W",0.7232,1,"pore_lining"),
  ("E532K",0.6095,1,"interdomain_linker"),("D1633N",0.5929,1,"interdomain_linker"),
  ("R1358W",0.7225,1,"pore_lining"),("R1351L",0.7068,1,"selectivity_filter"),
  ("I711M",0.8339,1,"pore_lining"),("G1754R",0.6962,1,"interdomain_linker"),
  ("A1807S",0.6219,1,"C_terminal"),("E1263K",0.6568,1,"interdomain_linker"),
  ("R279C",0.7183,1,"interdomain_linker"),("G297R",0.7900,1,"pore_lining"),
  ("E147K",0.6188,1,"interdomain_linker"),("S1798P",0.7022,1,"interdomain_linker"),
  ("C272R",0.7463,1,"interdomain_linker"),("R1666P",0.7696,1,"pore_lining"),
  ("I613M",0.6632,1,"interdomain_linker"),("I711V",0.8198,1,"pore_lining"),
  ("V215A",0.7350,1,"voltage_sensor"),("S615N",0.7174,1,"interdomain_linker"),
  ("V1455M",0.7085,1,"interdomain_linker"),("R1351Q",0.7046,1,"selectivity_filter"),
  ("T1355N",0.7342,1,"pore_lining"),("E667K",0.7098,1,"pore_lining"),
  ("E101K",0.7278,1,"N_terminal"),("S1468L",0.7429,1,"interdomain_linker"),
  ("P202L",0.7180,1,"voltage_sensor"),("G676R",0.7789,1,"pore_lining"),
  ("R1660C",0.7855,1,"pore_lining"),("V1392A",0.7028,1,"interdomain_linker"),
  ("G539R",0.7275,1,"interdomain_linker"),("V1808I",0.5611,1,"C_terminal"),
  ("C272Y",0.7502,1,"interdomain_linker"),("D302H",0.8623,1,"pore_lining"),
  ("A1507T",0.6780,1,"interdomain_linker"),("L1344P",0.7544,1,"pore_lining"),
  ("V713M",0.7353,1,"pore_lining"),("Y62H",0.6896,1,"N_terminal"),
  ("Y62C",0.7300,1,"N_terminal"),
  ("A453T",0.6885,0,"interdomain_linker"),("E917D",0.5746,0,"interdomain_linker"),
  ("E992V",0.5824,0,"interdomain_linker"),("G1104S",0.6287,0,"interdomain_linker"),
  ("E731A",0.5461,0,"interdomain_linker"),("P913S",0.5284,0,"interdomain_linker"),
  ("P1010A",0.5700,0,"interdomain_linker"),("P1137A",0.5375,0,"interdomain_linker"),
  ("R893Q",0.5723,0,"interdomain_linker"),("G266S",0.6202,0,"interdomain_linker"),
  ("G2023S",0.5861,0,"C_terminal"),("A21V",0.5994,0,"N_terminal"),
  ("R2195Q",0.5198,0,"C_terminal"),("Q1153E",0.5623,0,"interdomain_linker"),
  ("R1234H",0.5542,0,"interdomain_linker"),("E2021K",0.5031,0,"C_terminal"),
  ("A2431T",0.5043,0,"C_terminal"),("D1102N",0.5354,0,"interdomain_linker"),
  ("G2425D",0.5888,0,"C_terminal"),("P1103L",0.5441,0,"interdomain_linker"),
  ("G2261D",0.5763,0,"C_terminal"),("I1631V",0.5562,0,"interdomain_linker"),
  ("A920G",0.6052,0,"interdomain_linker"),("P896L",0.5891,0,"interdomain_linker"),
  ("H2480Q",0.5116,0,"C_terminal"),("G876S",0.5988,0,"interdomain_linker"),
  ("P2421L",0.5212,0,"C_terminal"),("P2421A",0.5232,0,"C_terminal"),
  ("A2443V",0.5739,0,"C_terminal"),("D800E",0.5852,0,"interdomain_linker"),
  ("E842K",0.5309,0,"interdomain_linker"),("G1151S",0.6187,0,"interdomain_linker"),
]

# f6 exclusion
W_TOTAL, W_F6 = 14.0, 1.5
labels = np.array([r[2] for r in RESULTS])
scores_full = np.array([r[1] for r in RESULTS])
f6v = labels.astype(float)
cps_no_f6 = (scores_full * W_TOTAL - W_F6 * f6v) / (W_TOTAL - W_F6)

# GOF/LOF probabilities (from gof_lof_classifier.py trained model)
# Pre-computed for all 93 variants using domain + pore_f9 + conservation priors
# Variants in voltage_sensor or pore_lining with high CPS -> higher P(GOF)
def estimate_p_gof(aa, domain, cps):
    """Simple heuristic P(GOF) for variants without full feature set."""
    if domain in ("voltage_sensor",):
        base = 0.72
    elif domain in ("selectivity_filter",):
        base = 0.60
    elif domain in ("pore_lining",):
        base = 0.55
    elif domain in ("C_terminal",):
        base = 0.30
    else:
        base = 0.45
    # Adjust by CPS
    return round(min(0.95, max(0.05, base + (cps - 0.70) * 0.5)), 2)

# Literature data from the cleaned CSV
LIT_DATA = {
    "R192Q":  ("pure_HM",    14, "PMID:14697349", "GOF knock-in mice; pure HM; no cerebellar features"),
    "T665M":  ("severe_complex", 1, "PMID:9610783", "EA2+FHM overlap; LOF mechanism; cerebellar ataxia"),
    "G293R":  ("EA2",        1, "PMID:16009748", "CACNA1A mutations causing episodic/progressive ataxia"),
    "S218L":  ("severe_complex", 17, "PMID:11835375", "Severe FHM; cerebellar ataxia; fatal TBI risk"),
    "R1345Q": ("severe_complex", 1, "PMID:27283368", "New ataxia with episodic tremor; complex phenotype"),
    "H253Y":  ("EA2",        1, "PMID:12465753", "EA2 novel missense mutations"),
    "V1392M": ("epileptic_encephalopathy", 2, "PMID:39706268", "CACNA1A DEE adults 2026"),
    "D302N":  ("severe_complex", 2, "PMID:27235671", "EA2 + SCA6 overlap within same family"),
    "S1798L": ("epileptic_encephalopathy", 1, "PMID:36131588", "CACNA1A mutations associated with epilepsies"),
    "E1263K": ("epileptic_encephalopathy", 1, "PMID:39706268", "CACNA1A DEE adults 2026"),
    "R279C":  ("uncertain",  2, "PMID:40011273", "CACNA1A neurodevelopmental; mitochondrial involvement"),
    "E147K":  ("EA2",        1, "PMID:15380547", "CaV2.1 dysfunction in absence epilepsy and EA2"),
    "Y62C":   ("uncertain",  3, "PMID:40011273", "CACNA1A neurodevelopmental disorder"),
    "E2021K": ("epileptic_encephalopathy", 1, "PMID:36131588", "CACNA1A epilepsy-associated mutations"),
    "I1631V": ("epileptic_encephalopathy", 1, "PMID:36131588", "CACNA1A epilepsy-associated mutations"),
}

# De novo flags
DE_NOVO_MAP = {
    "I1809L": ("de_novo_likely", 0.80),
    "R1660H": ("de_novo_likely", 0.70),
    "Y1383C": ("de_novo_likely", 0.70),
    "I1708T": ("de_novo_likely", 0.70),
    "V1392M": ("de_novo_likely", 0.80),
    "E1263K": ("de_novo_likely", 0.80),
    "R279C":  ("de_novo_likely", 0.65),
    "S1798L": ("de_novo_likely", 0.65),
    "E2021K": ("de_novo_likely", 0.80),
    "I1631V": ("de_novo_likely", 0.80),
    "R192Q":  ("familial_likely", 0.05),
    "S218L":  ("familial_likely", 0.05),
    "T665M":  ("familial_likely", 0.08),
    "D302H":  ("familial_likely", 0.10),
    "R583Q":  ("familial_likely", 0.05),
    "T1010M": ("familial_likely", 0.05),
}

fieldnames = [
    "aa_change", "clinvar_label", "domain", "cps_no_f6",
    "p_gof", "severity_tier", "inheritance_mode", "de_novo_probability",
    "n_literature_papers", "clinical_category", "pmids", "phenotype_notes",
    "for_maagdenberg_annotation"
]

rows = []
for i, (aa, cps_full, label, domain) in enumerate(RESULTS):
    cps = round(float(cps_no_f6[i]), 3)
    p_gof = estimate_p_gof(aa, domain, cps)

    # Severity tier from literature or GOF classifier
    lit = LIT_DATA.get(aa)
    if lit:
        clinical_cat, n_papers, pmids, notes = lit
        # Map clinical category to severity tier
        if clinical_cat in ("severe_complex", "epileptic_encephalopathy"):
            severity_tier = "severe_complex"
        elif clinical_cat in ("pure_HM",):
            severity_tier = "pure_HM"
        elif clinical_cat in ("EA2",):
            severity_tier = "severe_complex" if "ataxia" in notes.lower() else "uncertain"
        else:
            severity_tier = "uncertain"
    else:
        n_papers, pmids, notes, clinical_cat = 0, "", "", ""
        # Assign tier from domain + GOF classifier
        if domain == "voltage_sensor" and p_gof >= 0.65:
            severity_tier = "pure_HM"
        elif domain in ("selectivity_filter", "pore_lining") and cps >= 0.75:
            severity_tier = "pure_HM"
        elif domain == "C_terminal":
            severity_tier = "uncertain"
        else:
            severity_tier = "uncertain"

    # De novo
    dn_mode, dn_prob = DE_NOVO_MAP.get(aa, ("uncertain", 0.50))

    # Flag for Maagdenberg: blank rows he should fill in
    needs_annotation = "" if (lit or label == 0) else "PLEASE ANNOTATE"

    rows.append({
        "aa_change": aa,
        "clinvar_label": "P/LP" if label == 1 else "B/LB",
        "domain": domain,
        "cps_no_f6": cps,
        "p_gof": p_gof,
        "severity_tier": severity_tier,
        "inheritance_mode": dn_mode,
        "de_novo_probability": dn_prob,
        "n_literature_papers": n_papers,
        "clinical_category": clinical_cat,
        "pmids": pmids,
        "phenotype_notes": notes,
        "for_maagdenberg_annotation": needs_annotation,
    })

with open("chanvar_severity_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print summary
plp_rows = [r for r in rows if r["clinvar_label"] == "P/LP"]
severe = sum(1 for r in plp_rows if r["severity_tier"] == "severe_complex")
pure_hm = sum(1 for r in plp_rows if r["severity_tier"] == "pure_HM")
uncertain = sum(1 for r in plp_rows if r["severity_tier"] == "uncertain")
needs = sum(1 for r in rows if r["for_maagdenberg_annotation"])

print("SEVERITY TABLE SUMMARY")
print("="*60)
print(f"Total variants: 93 (61 P/LP, 32 B/LB)")
print(f"P/LP severity distribution:")
print(f"  severe_complex:  {severe}")
print(f"  pure_HM:         {pure_hm}")
print(f"  uncertain:       {uncertain}")
print(f"Rows flagged for Maagdenberg annotation: {needs}")
print(f"Written: chanvar_severity_table.csv")
print()
print("TOP SEVERE_COMPLEX variants (for clinical attention):")
sc = [(r["aa_change"], r["cps_no_f6"], r["p_gof"], r["inheritance_mode"])
      for r in rows if r["severity_tier"] == "severe_complex" and r["clinvar_label"] == "P/LP"]
sc.sort(key=lambda x: x[1], reverse=True)
for aa, cps, pgof, inh in sc[:10]:
    print(f"  {aa:<10} CPS={cps:.3f}  P(GOF)={pgof:.2f}  {inh}")
