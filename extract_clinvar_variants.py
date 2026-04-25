"""
extract_clinvar_variants.py
----------------------------
Extracts CACNA1A missense variants from the ClinVar variant_summary.txt.gz file
and builds the input CSV for chanvar batch.

Download the input file from:
  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

Save it to your chanvar/ project folder, then run:
  python extract_clinvar_variants.py

Outputs:
  clinvar_validation.csv  - input for: chanvar batch --input clinvar_validation.csv
"""

import gzip
import csv
import re
import os

INPUT_FILE = "variant_summary.txt.gz"
OUTPUT_FILE = "clinvar_validation.csv"
MIN_STARS = 2

# ClinVar review status -> star rating
REVIEW_STARS = {
    "no assertion provided": 0,
    "no assertion criteria provided": 0,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "reviewed by expert panel": 3,
    "practice guideline": 4,
}

# Amino acid 3-letter to 1-letter
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}


def extract_aa_change(name_field):
    """
    Extract short amino acid change from the ClinVar Name column.

    Examples of Name field values:
      NM_023035.2(CACNA1A):c.575G>A (p.Arg192Gln)
      NM_023035.2(CACNA1A):c.653C>T (p.Ser218Leu)
      NM_023035.2(CACNA1A):c.1997C>T (p.Thr666Met)

    Returns e.g. "R192Q", "S218L", "T666M", or "" if parsing fails.
    """
    # Try p. notation first (most informative)
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*|Ter)', name_field)
    if match:
        ref3, pos, alt3 = match.groups()
        ref1 = AA3_TO_1.get(ref3, "?")
        alt1 = AA3_TO_1.get(alt3, "*") if alt3 != "*" else "*"
        return f"{ref1}{pos}{alt1}"
    return ""


def get_star_rating(review_status):
    """Convert ClinVar review status string to star integer."""
    status_lower = review_status.lower().strip()
    for key, stars in REVIEW_STARS.items():
        if key in status_lower:
            return stars
    return 0


def classify_significance(clnsig):
    """Return canonical category from ClinVar significance string."""
    s = clnsig.lower()
    if "pathogenic" in s and "likely" in s:
        return "Likely pathogenic"
    elif "pathogenic" in s:
        return "Pathogenic"
    elif "benign" in s and "likely" in s:
        return "Likely benign"
    elif "benign" in s:
        return "Benign"
    elif "uncertain" in s or "vus" in s:
        return "VUS"
    return "other"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found in current directory.")
        print("Download it from:")
        print("  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz")
        return

    print(f"Reading {INPUT_FILE}...")

    positives = []  # P/LP
    negatives = []  # B/LB
    vus = []        # VUS (inference targets)
    skipped_no_aa = 0
    skipped_stars = 0
    skipped_not_missense = 0
    total = 0

    opener = gzip.open if INPUT_FILE.endswith(".gz") else open

    with opener(INPUT_FILE, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total += 1
            if total % 50000 == 0:
                print(f"  Processed {total:,} rows...")

            # Filter to CACNA1A
            if row.get("GeneSymbol", "") != "CACNA1A":
                continue

            # Filter to GRCh38
            if row.get("Assembly", "") != "GRCh38":
                continue

            # Filter to missense (single nucleotide variant with protein change)
            variant_type = row.get("Type", "")
            name = row.get("Name", "")
            if "single nucleotide variant" not in variant_type.lower():
                skipped_not_missense += 1
                continue

            # Must have a protein change in the name
            if "p." not in name:
                skipped_not_missense += 1
                continue

            # Extract amino acid change
            aa_change = extract_aa_change(name)
            if not aa_change or "?" in aa_change or "*" in aa_change:
                skipped_no_aa += 1
                continue

            # Clinical significance
            clnsig = row.get("ClinicalSignificance", "")
            category = classify_significance(clnsig)

            # Review status / star rating
            review_status = row.get("ReviewStatus", "")
            stars = get_star_rating(review_status)

            # Phenotype
            phenotype = row.get("PhenotypeList", "")

            # Genomic position
            chrom = row.get("Chromosome", "")
            start = row.get("Start", "")
            ref = row.get("ReferenceAllele", "")
            alt = row.get("AlternateAllele", "")
            variant_id = f"{chrom}-{start}-{ref}-{alt}" if all([chrom, start, ref, alt]) else name[:50]

            record = {
                "aa_change": aa_change,
                "clinvar_sig": clnsig,
                "review_stars": stars,
                "category": category,
                "phenotype": phenotype[:80] if phenotype else "",
                "variant_id": variant_id,
                "name": name[:100],
                "af": "",    # will be queried from gnomAD
                "cadd": "",  # optional
            }

            if category in ("Pathogenic", "Likely pathogenic"):
                if stars >= MIN_STARS:
                    positives.append(record)
                else:
                    skipped_stars += 1
            elif category in ("Benign", "Likely benign"):
                if stars >= MIN_STARS:
                    negatives.append(record)
                else:
                    skipped_stars += 1
            elif category == "VUS":
                vus.append(record)

    print(f"\nTotal rows scanned: {total:,}")
    print(f"Skipped (not missense / no p. notation): {skipped_not_missense}")
    print(f"Skipped (below {MIN_STARS}-star threshold): {skipped_stars}")
    print(f"Skipped (aa_change parse failed): {skipped_no_aa}")
    print()
    print(f"P/LP with >= {MIN_STARS} stars: {len(positives)}")
    print(f"B/LB with >= {MIN_STARS} stars: {len(negatives)}")
    print(f"VUS (all star ratings): {len(vus)}")
    print(f"Total usable for validation (P/LP + B/LB): {len(positives) + len(negatives)}")

    if positives:
        print(f"\nExample P/LP variants:")
        for r in positives[:8]:
            print(f"  {r['aa_change']:8s} | {r['clinvar_sig'][:35]:35s} | {r['review_stars']} stars")

    if negatives:
        print(f"\nExample B/LB variants:")
        for r in negatives[:8]:
            print(f"  {r['aa_change']:8s} | {r['clinvar_sig'][:35]:35s} | {r['review_stars']} stars")

    # Write validation CSV (P/LP + B/LB only)
    all_for_validation = positives + negatives
    if not all_for_validation:
        print("\nERROR: No usable variants found. Check that variant_summary.txt.gz is the correct file.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["aa_change", "clinvar_sig", "review_stars", "category",
                      "phenotype", "variant_id", "af", "cadd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_for_validation:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nWritten {len(all_for_validation)} variants to {OUTPUT_FILE}")
    print(f"Ready to run:")
    print(f"  chanvar batch --input {OUTPUT_FILE} --output results\\clinvar_full\\")

    # Also write VUS separately for inference (optional)
    vus_file = "clinvar_vus.csv"
    with open(vus_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["aa_change", "clinvar_sig", "review_stars", "category",
                      "phenotype", "variant_id", "af", "cadd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in vus:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"Also written {len(vus)} VUS variants to {vus_file}")
    print("(VUS are your inference targets -- run the batch on these too after validation)")


if __name__ == "__main__":
    main()