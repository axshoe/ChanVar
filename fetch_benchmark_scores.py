
"""
fetch_benchmark_scores.py
-------------------------
Fetches AlphaMissense and CADD scores for the 93 ClinVar validation variants
using Ensembl VEP REST API.

Run: python fetch_benchmark_scores.py
Output: benchmark_scores.json
"""

import csv, json, time, urllib.request

VARIANTS = [
    # (aa_change, hgvs_protein)
    ("R192Q", "ENSP00000261996.2:p.Arg192Gln"),
    ("S218L", "ENSP00000261996.2:p.Ser218Leu"),
    ("T665M", "ENSP00000261996.2:p.Thr665Met"),
    ("D302H", "ENSP00000261996.2:p.Asp302His"),
    # Add all 93... or use genomic HGVS from variant_id field
]

# Better: use genomic HGVS notation from the variant_id
# variant_id format: "19-13392445-G-A" -> "19:g.13392445G>A"

def vep_lookup(hgvs_notation, retries=3):
    url = f"https://rest.ensembl.org/vep/human/hgvs/{urllib.request.quote(hgvs_notation)}"
    url += "?content-type=application/json&AlphaMissense=1&CADD=1"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt < retries-1:
                time.sleep(2)
            else:
                return None

# Load variant IDs from clinvar_validation.csv
results = {}
with open("clinvar_validation.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        aa = row["aa_change"]
        vid = row.get("variant_id", "")
        if "-" in vid:
            parts = vid.split("-")
            if len(parts) == 4:
                chrom, pos, ref, alt = parts
                hgvs = f"{chrom}:g.{pos}{ref}>{alt}"
                print(f"Looking up {aa} ({hgvs})...")
                data = vep_lookup(hgvs)
                if data:
                    # Extract AlphaMissense and CADD scores
                    am_score = None
                    cadd_score = None
                    for hit in (data if isinstance(data, list) else [data]):
                        for tc in hit.get("transcript_consequences", []):
                            if tc.get("gene_symbol") == "CACNA1A":
                                am_score = tc.get("alphamissense", {}).get("am_pathogenicity")
                                cadd_score = tc.get("cadd_phred")
                                break
                    results[aa] = {"alphamissense": am_score, "cadd_phred": cadd_score}
                    print(f"  AlphaMissense={am_score}, CADD={cadd_score}")
                time.sleep(0.5)  # rate limit

with open("benchmark_scores.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Written {len(results)} results to benchmark_scores.json")
