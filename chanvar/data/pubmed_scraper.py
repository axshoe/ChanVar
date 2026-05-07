"""
chanvar/data/pubmed_scraper.py
--------------------------------
Priority 2: Literature scraper for CACNA1A validation variants.

For each variant in the ClinVar validation set, searches PubMed for:
  1. Case reports with clinical phenotype (pure HM, severe complex, EA2, etc.)
  2. Functional studies with electrophysiology data

Output: chanvar_literature_annotations.csv
  Columns: aa_change, clinvar_sig, pubmed_ids, clinical_category,
           activation_v12_shift_mv, peak_current_ratio,
           inactivation_v12_shift_mv, mechanism, source_papers

Usage:
  python chanvar/data/pubmed_scraper.py

This script generates the spreadsheet you send to van den Maagdenberg
for collaborative validation. Run once; extend manually with additional
literature as you find it.
"""

import csv
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path


VALIDATION_VARIANTS = [
    "R192Q","T665M","I1809L","R1660H","G293R","Y1383C","S218L","R582Q",
    "I1708T","R1345Q","T500M","R1663Q","R1666W","H253Y","V1392M","D302N",
    "A712T","R1348Q","S1798L","D1643N","A1507D","P1360Q","P1352L","R1348W",
    "E532K","D1633N","R1358W","R1351L","I711M","G1754R","A1807S","E1263K",
    "R279C","G297R","E147K","S1798P","C272R","R1666P","I613M","I711V",
    "V215A","S615N","V1455M","R1351Q","T1355N","E667K","E101K","S1468L",
    "P202L","G676R","R1660C","V1392A","G539R","V1808I","C272Y","D302H",
    "A1507T","L1344P","V713M","Y62H","Y62C",
    "A453T","E917D","E992V","G1104S","E731A","P913S","P1010A","P1137A",
    "R893Q","G266S","G2023S","A21V","R2195Q","Q1153E","R1234H","E2021K",
    "A2431T","D1102N","G2425D","P1103L","G2261D","I1631V","A920G","P896L",
    "H2480Q","G876S","P2421L","P2421A","A2443V","D800E","E842K","G1151S",
]

CLINVAR_LABELS = {
    "R192Q":1,"T665M":1,"I1809L":1,"R1660H":1,"G293R":1,"Y1383C":1,"S218L":1,
    "R582Q":1,"I1708T":1,"R1345Q":1,"T500M":1,"R1663Q":1,"R1666W":1,"H253Y":1,
    "V1392M":1,"D302N":1,"A712T":1,"R1348Q":1,"S1798L":1,"D1643N":1,"A1507D":1,
    "P1360Q":1,"P1352L":1,"R1348W":1,"E532K":1,"D1633N":1,"R1358W":1,"R1351L":1,
    "I711M":1,"G1754R":1,"A1807S":1,"E1263K":1,"R279C":1,"G297R":1,"E147K":1,
    "S1798P":1,"C272R":1,"R1666P":1,"I613M":1,"I711V":1,"V215A":1,"S615N":1,
    "V1455M":1,"R1351Q":1,"T1355N":1,"E667K":1,"E101K":1,"S1468L":1,"P202L":1,
    "G676R":1,"R1660C":1,"V1392A":1,"G539R":1,"V1808I":1,"C272Y":1,"D302H":1,
    "A1507T":1,"L1344P":1,"V713M":1,"Y62H":1,"Y62C":1,
    "A453T":0,"E917D":0,"E992V":0,"G1104S":0,"E731A":0,"P913S":0,"P1010A":0,
    "P1137A":0,"R893Q":0,"G266S":0,"G2023S":0,"A21V":0,"R2195Q":0,"Q1153E":0,
    "R1234H":0,"E2021K":0,"A2431T":0,"D1102N":0,"G2425D":0,"P1103L":0,
    "G2261D":0,"I1631V":0,"A920G":0,"P896L":0,"H2480Q":0,"G876S":0,
    "P2421L":0,"P2421A":0,"A2443V":0,"D800E":0,"E842K":0,"G1151S":0,
}


def pubmed_search(query: str, max_results: int = 5) -> list:
    """Search PubMed and return list of PMIDs."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ChanVar/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def pubmed_fetch_titles(pmids: list) -> dict:
    """Fetch titles for a list of PMIDs."""
    if not pmids:
        return {}
    ids_str = ",".join(pmids)
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
           f"?db=pubmed&id={ids_str}&retmode=json")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ChanVar/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        result = data.get("result", {})
        return {
            pmid: result.get(pmid, {}).get("title", "")
            for pmid in pmids
        }
    except Exception:
        return {}


def search_variant(aa_change: str) -> dict:
    """
    Search PubMed for case reports and functional studies on a CACNA1A variant.

    Returns dict with pmids and titles for manual review.
    """
    # Two queries: clinical/case reports + electrophysiology
    query_clinical = (
        f'CACNA1A[gene] AND "{aa_change}"[tiab] AND '
        f'(migraine OR hemiplegic OR ataxia OR "episodic ataxia")[tiab]'
    )
    query_electro = (
        f'CACNA1A[gene] AND "{aa_change}"[tiab] AND '
        f'(electrophysiology OR "patch clamp" OR "activation" OR '
        f'"current density" OR "gain of function")[tiab]'
    )

    pmids_clinical = pubmed_search(query_clinical, max_results=5)
    time.sleep(0.35)
    pmids_electro = pubmed_search(query_electro, max_results=5)
    time.sleep(0.35)

    all_pmids = list(set(pmids_clinical + pmids_electro))
    titles = pubmed_fetch_titles(all_pmids) if all_pmids else {}

    return {
        "pmids_clinical": pmids_clinical,
        "pmids_electro": pmids_electro,
        "all_pmids": all_pmids,
        "titles": titles,
        "n_papers": len(all_pmids),
    }


def run_full_search(output_path: str = "chanvar_literature_annotations.csv"):
    """
    Run PubMed search for all 93 validation variants and write CSV.

    The CSV is the spreadsheet to send van den Maagdenberg.
    Manually fill in clinical_category and electrophysiology columns
    after reviewing the PMIDs.
    """
    import sys
    from chanvar.data.severity_tier import get_severity_tier
    from chanvar.data.biophysical_severity import (
        get_electrophys, compute_biophysical_severity
    )

    fieldnames = [
        "aa_change",
        "clinvar_label",           # 1=P/LP, 0=B/LB
        "severity_tier",           # from literature (auto-filled where known)
        "n_papers_found",
        "pmids",
        "sample_titles",
        # To fill manually after review:
        "clinical_category",       # pure_HM / severe_complex / EA2 / SCA6 / epileptic_encephalopathy / uncertain
        "activation_v12_shift_mv", # from patch-clamp
        "peak_current_ratio",      # mutant/WT
        "inactivation_v12_shift_mv",
        "mechanism",               # GOF / LOF / mixed / uncertain
        "expression_system",
        "notes",
    ]

    rows = []
    total = len(VALIDATION_VARIANTS)

    for i, aa in enumerate(VALIDATION_VARIANTS):
        print(f"[{i+1}/{total}] Searching PubMed for {aa}...", end=" ", flush=True)
        results = search_variant(aa)

        tier = get_severity_tier(aa)
        ep = get_electrophys(aa)
        bs = compute_biophysical_severity(aa)

        title_sample = " | ".join(list(results["titles"].values())[:2])
        pmid_str = ",".join(results["all_pmids"])

        row = {
            "aa_change": aa,
            "clinvar_label": CLINVAR_LABELS.get(aa, ""),
            "severity_tier": tier.tier,
            "n_papers_found": results["n_papers"],
            "pmids": pmid_str,
            "sample_titles": title_sample[:200],
            # Pre-fill from biophysical database where available
            "clinical_category": tier.tier if tier.tier != "uncertain" else "",
            "activation_v12_shift_mv": ep.activation_v12_shift_mv if ep else "",
            "peak_current_ratio": ep.peak_current_ratio if ep else "",
            "inactivation_v12_shift_mv": ep.inactivation_v12_shift_mv if ep else "",
            "mechanism": ep.mechanism if ep else "",
            "expression_system": ep.expression_system if ep else "",
            "notes": "",
        }
        rows.append(row)
        print(f"{results['n_papers']} papers found")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSpreadsheet written to: {output_path}")
    print(f"Total variants: {total}")
    pre_filled = sum(1 for r in rows if r["clinical_category"])
    print(f"Pre-filled from database: {pre_filled}")
    print(f"Needs manual review: {total - pre_filled}")
    print(f"\nSend this CSV to van den Maagdenberg for collaborative annotation.")


if __name__ == "__main__":
    run_full_search()
