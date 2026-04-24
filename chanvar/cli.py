"""
chanvar/cli.py
--------------
Command-line interface for ChanVar variant annotation.

Usage examples:
  chanvar annotate --variant R192Q
  chanvar annotate --variant R192Q --af 0.0 --ddg 3.2 --cadd 35.0
  chanvar batch --input variants.csv --output results.csv
  chanvar structure --pdb AF-O00555-F1-model_v4.pdb --variant R192Q
  chanvar train --clinvar clinvar_cacna1a.vcf
  chanvar validate

All commands produce JSON output by default; add --format markdown or --format html for reports.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("chanvar")


def cmd_annotate(args):
    """Annotate a single CACNA1A variant."""
    from chanvar.scoring.features import build_feature_vector
    from chanvar.scoring.cps import compute_cps
    from chanvar.report.generator import generate_report

    # Load conservation scores if available
    conservation_scores = None
    if args.conservation and Path(args.conservation).exists():
        from chanvar.evolution.conservation import load_conservation_scores
        conservation_scores = load_conservation_scores(args.conservation)

    features = build_feature_vector(
        variant_id=args.variant_id or f"19-unknown-{args.variant}",
        aa_change=args.variant,
        gnomad_af=args.af,
        ddg_foldx=args.ddg,
        rmsd=args.rmsd,
        conservation_scores=conservation_scores,
        alphamissense_score=args.alphamissense,
        cadd_phred=args.cadd,
        clinvar_sig=args.clinvar_sig,
    )

    result = compute_cps(features, bootstrap_n=args.bootstrap_n)

    print(f"\n{result.summary_line()}\n")
    print("Feature contributions:")
    for fname, contrib in result.feature_contributions.items():
        bar = "█" * int(contrib * 30)
        print(f"  {fname}: {contrib:.3f} {bar}")
    print(f"\nData completeness: {features.data_completeness:.0%}")

    if args.output:
        paths = generate_report(
            result, features,
            output_dir=args.output,
            formats=args.format.split(",") if args.format else None,
        )
        print(f"\nReport written: {list(paths.values())}")

    return result


def cmd_batch(args):
    """Score a CSV of variants, writing per-variant reports to an output directory."""
    import csv
    from chanvar.scoring.features import batch_build_features
    from chanvar.scoring.cps import batch_score
    from chanvar.report.generator import generate_report

    # Load input CSV
    variants = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append(row)

    logger.info("Loaded %d variants from %s", len(variants), args.input)

    conservation_scores = None
    if args.conservation and Path(args.conservation).exists():
        from chanvar.evolution.conservation import load_conservation_scores
        conservation_scores = load_conservation_scores(args.conservation)

    features_list = batch_build_features(variants, conservation_scores=conservation_scores)
    results = batch_score(features_list, bootstrap_n=500)

    # Output directory — strip trailing slash/backslash so Path works on Windows
    out_dir = args.output.rstrip("/\\") if args.output else "batch_results"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Write per-variant reports
    all_summaries = []
    for fv, res in zip(features_list, results):
        print(f"\n{res.summary_line()}")
        print(f"  Data completeness: {fv.data_completeness:.0%}")
        try:
            generate_report(res, fv, output_dir=out_dir,
                            formats=["json", "markdown", "html"])
        except Exception as exc:
            logger.warning("Report failed for %s: %s", res.aa_change, exc)
        all_summaries.append({
            "aa_change": res.aa_change,
            "cps": round(res.cps, 4),
            "ci_lower": round(res.ci_lower, 4),
            "ci_upper": round(res.ci_upper, 4),
            "classification": res.classification,
            "domain": res.domain,
            "data_completeness": round(fv.data_completeness, 3),
            "confidence_flag": res.confidence_flag or "",
        })

    # Write batch_summary.json
    summary_path = Path(out_dir) / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n--- Batch complete: {len(results)} variants ---")
    print(f"Reports written to: {out_dir}\\")
    print(f"Summary: {summary_path}")


def cmd_train(args):
    """Train CPS weights from ClinVar data."""
    from chanvar.data.clinvar_parser import parse_clinvar_vcf, get_training_sets
    from chanvar.scoring.features import build_feature_vector
    from chanvar.scoring.cps import train_logistic_weights

    print(f"Parsing ClinVar VCF: {args.clinvar}")
    records = parse_clinvar_vcf(args.clinvar)
    positives, negatives = get_training_sets(records, min_stars=2)
    print(f"Training set: {len(positives)} P/LP, {len(negatives)} B/LB")

    if len(positives) < 20:
        print("Warning: Training set too small for reliable logistic regression. Using prior weights.")
        return

    features_list = []
    labels = []
    for rec in positives + negatives:
        try:
            fv = build_feature_vector(
                variant_id=rec.variant_id,
                aa_change=rec.aa_change,
                clinvar_sig=rec.raw_classification,
            )
            features_list.append(fv)
            labels.append(1 if rec.classification == "pathogenic" else 0)
        except Exception as exc:
            logger.debug("Skipping %s: %s", rec.aa_change, exc)

    weights, auroc = train_logistic_weights(features_list, labels)
    print(f"Learned weights: {json.dumps(weights, indent=2)}")
    print(f"CV AUROC: {auroc:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"weights": weights, "cv_auroc": auroc}, f, indent=2)
        print(f"Weights saved to {args.output}")


def cmd_validate(args):
    """Run unit tests on all scoring functions."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        prog="chanvar",
        description="ChanVar: In Silico Functional Annotation of CACNA1A Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chanvar annotate --variant R192Q
  chanvar annotate --variant S218L --af 0.0 --ddg 2.8 --cadd 38 --alphamissense 0.95
  chanvar batch --input variants.csv --output results.csv
  chanvar train --clinvar clinvar_cacna1a.vcf --output weights.json
  chanvar validate

Documentation: https://github.com/axshoe/chanvar
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # annotate
    ann = subparsers.add_parser("annotate", help="Annotate a single CACNA1A variant")
    ann.add_argument("--variant", required=True, help="Amino acid change, e.g. R192Q")
    ann.add_argument("--variant-id", dest="variant_id", help="gnomAD variant ID")
    ann.add_argument("--af", type=float, help="gnomAD allele frequency")
    ann.add_argument("--ddg", type=float, help="FoldX dDDG (kcal/mol)")
    ann.add_argument("--rmsd", type=float, help="Local backbone RMSD (Å)")
    ann.add_argument("--alphamissense", type=float, help="AlphaMissense score [0-1]")
    ann.add_argument("--cadd", type=float, help="CADD Phred score")
    ann.add_argument("--clinvar-sig", dest="clinvar_sig", help="ClinVar classification")
    ann.add_argument("--conservation", help="Path to cacna1a_conservation.tsv")
    ann.add_argument("--output", help="Output directory for reports")
    ann.add_argument("--format", default="json,markdown,html", help="Report format(s)")
    ann.add_argument("--bootstrap-n", dest="bootstrap_n", type=int, default=1000)
    ann.set_defaults(func=cmd_annotate)

    # batch
    bat = subparsers.add_parser("batch", help="Score a CSV of variants")
    bat.add_argument("--input", required=True, help="Input CSV with aa_change column")
    bat.add_argument("--output", help="Output CSV path")
    bat.add_argument("--conservation", help="Path to cacna1a_conservation.tsv")
    bat.set_defaults(func=cmd_batch)

    # train
    trn = subparsers.add_parser("train", help="Train CPS weights from ClinVar")
    trn.add_argument("--clinvar", required=True, help="ClinVar VCF for CACNA1A")
    trn.add_argument("--output", help="Output JSON for learned weights")
    trn.set_defaults(func=cmd_train)

    # validate
    val = subparsers.add_parser("validate", help="Run unit tests")
    val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()