"""
chanvar/report/generator.py
----------------------------
Markdown and HTML report generation for ChanVar variant annotation.

Generates three output formats:
  1. Machine-readable JSON (full feature values + CPS)
  2. Markdown report (human-readable, suitable for lab notebook)
  3. HTML report (with embedded visualizations, for lab website)

Each report section:
  - Variant identity (gnomAD ID, HGVS notation, amino acid change)
  - CPS with confidence interval and interpretation bar
  - ClinVar context (if available)
  - Feature breakdown table (f1-f8 with explanations)
  - Domain context (which part of the channel is affected)
  - Structural context (pLDDT, dDDG, RMSD)
  - Conservation context (rate4site, PhyloP, dominant species)
  - Limitations and caveats (domain-specific and global)
  - Academic citations for all methods used
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from chanvar.scoring.cps import CPSResult
from chanvar.scoring.features import VariantFeatures

logger = logging.getLogger(__name__)

CHANVAR_VERSION = "1.0.0"
TODAY = datetime.today().strftime("%B %d, %Y")

DOMAIN_DESCRIPTIONS = {
    "voltage_sensor": (
        "Voltage sensor (S4 segment). S4 contains positively charged arginine and lysine "
        "residues that move outward during membrane depolarization, physically opening the "
        "channel pore. Missense variants at S4 arginine positions are the most common "
        "mechanism of FHM1 pathogenesis (Ophoff et al. 1996). ChanVar applies maximum "
        "domain weight (1.0) to variants at this position."
    ),
    "selectivity_filter": (
        "Selectivity filter (EEEE locus). Four glutamate residues — one per domain — "
        "form the ion selectivity filter that allows Ca2+ entry while excluding Na+ and K+. "
        "Variants at EEEE positions are expected to abolish calcium selectivity, fundamentally "
        "disrupting Cav2.1 function. Domain weight: 1.0."
    ),
    "pore_lining": (
        "Pore-lining segment (S5-S6). The S5-S6 segments form the physical channel pore "
        "and gate. Variants here affect pore geometry and calcium conductance. Less clinically "
        "characterized than S4 variants but biologically important. Domain weight: 0.8. "
        "Note: dDDG reliability is reduced for transmembrane variants."
    ),
    "IIIIV_linker": (
        "Domain III-IV linker (inactivation gate). This cytoplasmic linker contains the "
        "inactivation particle that blocks the channel pore during sustained depolarization "
        "(fast inactivation). Variants here affect inactivation kinetics rather than channel "
        "gating per se, and may produce subtler clinical phenotypes. Domain weight: 0.6."
    ),
    "C_terminal": (
        "C-terminal tail. Contains SNARE-interaction domains (syntaxin, SNAP-25, "
        "synaptotagmin binding sites) that couple channel opening to vesicle fusion. "
        "Variants here may impair neurotransmitter release coupling without disrupting "
        "channel gating. Domain weight: 0.4. dDDG calculations more reliable here than "
        "in transmembrane segments."
    ),
    "N_terminal": (
        "N-terminal domain. Poorly characterized functionally in CACNA1A. "
        "Domain weight: 0.3."
    ),
    "interdomain_linker": (
        "Interdomain linker region (S1-S3 segments or inter-repeat linkers). "
        "These connect transmembrane helices within and between domains. Variants here "
        "may affect channel assembly, trafficking, or auxiliary subunit interactions. "
        "Domain weight: 0.5."
    ),
}

FEATURE_DESCRIPTIONS = {
    "f1": ("gnomAD Population Frequency", "Variant rarity in 730,000 sequenced individuals"),
    "f2": ("Thermodynamic Stability (dDDG)", "FoldX/EvoEF2 predicted destabilization (kcal/mol)"),
    "f3": ("Local Structure RMSD", "Backbone displacement in mutant vs wildtype structure (Å)"),
    "f4": ("Evolutionary Conservation", "rate4site normalized evolutionary rate at this position"),
    "f5": ("Functional Domain Weight", "Biological importance of the affected channel domain"),
    "f6": ("AlphaMissense", "Deep learning variant effect prediction (Cheng et al. 2023)"),
    "f7": ("CADD Score", "Combined Annotation-Dependent Depletion (Kircher et al. 2014)"),
    "f8": ("Physicochemical Change", "Grantham distance (amino acid property dissimilarity)"),
}

CITATIONS = {
    "ophoff1996": "Ophoff RA et al. (1996). Familial hemiplegic migraine and episodic ataxia type-2 are caused by mutations in the Ca2+ channel gene CACNA1A. Cell, 87(3), 543–552.",
    "lek2016": "Lek M et al. (2016). Analysis of protein-coding genetic variation in 60,706 humans. Nature, 536, 285–291.",
    "karczewski2020": "Karczewski KJ et al. (2020). The mutational constraint spectrum from variation in 141,456 humans. Nature, 581, 434–443.",
    "ashkenazy2010": "Ashkenazy H et al. (2010). ConSurf 2010: calculating evolutionary conservation in sequence and structure of proteins and nucleic acids. Nucleic Acids Research, 38, W529–W533.",
    "cheng2023": "Cheng J et al. (2023). Accurate proteome-wide missense variant effect prediction with AlphaMissense. Science, 381, eadg7492.",
    "schymkowitz2005": "Schymkowitz J et al. (2005). The FoldX web server: an online force field. Nucleic Acids Research, 33, W382–W388.",
    "kircher2014": "Kircher M et al. (2014). A general framework for estimating the relative pathogenicity of human genetic variants. Nature Genetics, 46, 310–315.",
    "pejaver2022": "Pejaver V et al. (2022). Calibration of computational tools to assess single-nucleotide variant pathogenicity using ClinVar. American Journal of Human Genetics, 109(12), 2163–2177.",
    "grantham1974": "Grantham R. (1974). Amino acid difference formula to help explain protein evolution. Science, 185(4154), 862–864.",
    "jumper2021": "Jumper J et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583–589.",
}


def generate_report(
    cps_result: CPSResult,
    features: VariantFeatures,
    output_dir: str = "reports",
    formats: list = None,
) -> dict[str, str]:
    """
    Generate ChanVar annotation report in multiple formats.

    Parameters
    ----------
    cps_result : CPSResult
        CPS computation result.
    features : VariantFeatures
        Full feature vector (for detailed breakdown).
    output_dir : str
        Directory to write report files.
    formats : list of str
        Formats to generate: ['json', 'markdown', 'html'].
        Defaults to all three.

    Returns
    -------
    dict
        Mapping format -> output file path.
    """
    if formats is None:
        formats = ["json", "markdown", "html"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_paths = {}

    slug = cps_result.aa_change.replace("/", "_").replace(":", "_")

    if "json" in formats:
        json_path = str(Path(output_dir) / f"chanvar_{slug}.json")
        _write_json(cps_result, features, json_path)
        output_paths["json"] = json_path

    if "markdown" in formats:
        md_path = str(Path(output_dir) / f"chanvar_{slug}.md")
        _write_markdown(cps_result, features, md_path)
        output_paths["markdown"] = md_path

    if "html" in formats:
        html_path = str(Path(output_dir) / f"chanvar_{slug}.html")
        _write_html(cps_result, features, html_path)
        output_paths["html"] = html_path

    logger.info("Reports written for %s: %s", cps_result.aa_change, list(output_paths.values()))
    return output_paths


def _write_json(cps_result: CPSResult, features: VariantFeatures, path: str) -> None:
    payload = {
        "chanvar_version": CHANVAR_VERSION,
        "generated": TODAY,
        "variant": {
            "variant_id": cps_result.variant_id,
            "aa_change": cps_result.aa_change,
            "ref_aa": features.ref_aa,
            "alt_aa": features.alt_aa,
            "residue_number": features.residue_number,
            "domain": cps_result.domain,
        },
        "cps": {
            "score": cps_result.cps,
            "ci_lower": cps_result.ci_lower,
            "ci_upper": cps_result.ci_upper,
            "classification": cps_result.classification,
            "confidence_flag": cps_result.confidence_flag,
            "data_completeness": cps_result.data_completeness,
        },
        "features": {
            "f1_gnomad_af": {"score": features.f1, "raw_af": features.gnomad_af},
            "f2_ddg": {"score": features.f2, "raw_kcal_mol": features.ddg_raw},
            "f3_rmsd": {"score": features.f3, "raw_angstrom": features.rmsd_raw},
            "f4_conservation": {"score": features.f4, "raw_rate4site": features.rate4site_raw},
            "f5_domain_weight": {"score": features.f5},
            "f6_alphamissense": {"score": features.f6},
            "f7_cadd": {"score": features.f7},
            "f8_grantham": {"score": features.f8},
        },
        "feature_contributions": cps_result.feature_contributions,
        "weights_used": cps_result.weights_used,
        "clinvar": {
            "sig": cps_result.clinvar_sig,
            "override": cps_result.clinvar_override,
        },
        "plddt": features.plddt,
        "is_tm_domain": features.is_tm_domain,
        "citations": CITATIONS,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_markdown(cps_result: CPSResult, features: VariantFeatures, path: str) -> None:
    lines = []
    a = lines.append

    a(f"# ChanVar Annotation Report")
    a(f"**Variant:** CACNA1A {cps_result.aa_change}")
    a(f"**Generated:** {TODAY} | **ChanVar v{CHANVAR_VERSION}** | thexiulab.org\n")
    a("---\n")

    # CPS box
    bar = _cps_bar_text(cps_result.cps)
    a(f"## ChanVar Pathogenicity Score (CPS)")
    a(f"```")
    a(f"CPS = {cps_result.cps:.3f}  [{bar}]")
    a(f"95% CI: {cps_result.ci_lower:.3f} – {cps_result.ci_upper:.3f}")
    a(f"Classification: {cps_result.classification}")
    a(f"Data completeness: {cps_result.data_completeness:.0%}")
    if cps_result.confidence_flag:
        a(f"⚠ Flags: {cps_result.confidence_flag}")
    a(f"```\n")

    # Variant context
    a(f"## Variant")
    a(f"- **gnomAD ID:** {cps_result.variant_id}")
    a(f"- **Amino acid change:** {cps_result.aa_change}")
    a(f"- **Residue:** {features.residue_number}")
    a(f"- **Functional domain:** {cps_result.domain}")
    a(f"- **Transmembrane:** {'Yes' if features.is_tm_domain else 'No'}")
    if features.plddt is not None:
        a(f"- **AlphaFold2 pLDDT:** {features.plddt:.1f}/100")
    a("")

    # Domain description
    desc = DOMAIN_DESCRIPTIONS.get(cps_result.domain, "")
    if desc:
        a(f"**Domain context:** {desc}\n")

    # ClinVar
    if cps_result.clinvar_sig:
        a(f"## ClinVar")
        a(f"- **Classification:** {cps_result.clinvar_sig}")
        if cps_result.clinvar_override:
            a(f"- CPS is reported but classification follows ClinVar high-confidence verdict.")
        a("")

    # Feature breakdown
    a(f"## Feature Breakdown\n")
    a(f"| Feature | Score | Raw Value | Contribution |")
    a(f"|---------|-------|-----------|-------------|")
    for fname in ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]:
        label, desc_short = FEATURE_DESCRIPTIONS[fname]
        score = getattr(features, fname)
        score_str = f"{score:.3f}" if score is not None else "missing"
        contribution = cps_result.feature_contributions.get(fname, 0)
        raw = _get_raw(fname, features)
        a(f"| {label} | {score_str} | {raw} | {contribution:.3f} |")
    a("")

    # Limitations
    a(f"## Limitations")
    a(f"- CPS is a computational prediction; it does not constitute a clinical classification.")
    a(f"- dDDG calculations for transmembrane segments assume aqueous solvent and are less reliable than for cytoplasmic domains.")
    a(f"- The ClinVar training set for CACNA1A is small (~150-250 high-confidence P/LP variants); the model may not generalize to understudied regions.")
    a(f"- Absence of a positive CPS signal does not exclude pathogenicity; FAERS-based signal detection is susceptible to underreporting.")
    a(f"- For clinical decision-making, functional electrophysiological validation (patch-clamp, calcium imaging) is required.\n")

    # Citations
    a(f"## References")
    for ref_id, ref_text in CITATIONS.items():
        a(f"- {ref_text}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_html(cps_result: CPSResult, features: VariantFeatures, path: str) -> None:
    """Generate a self-contained HTML report with visual CPS indicator."""
    cps_pct = int(cps_result.cps * 100)
    ci_low_pct = int(cps_result.ci_lower * 100)
    ci_high_pct = int(cps_result.ci_upper * 100)

    classification_colors = {
        "Likely Pathogenic": "#d62728",
        "Possibly Pathogenic": "#ff7f0e",
        "Uncertain Significance": "#bcbd22",
        "Possibly Benign": "#17becf",
        "Likely Benign": "#1f77b4",
        "Pathogenic (ClinVar)": "#d62728",
        "Benign (ClinVar)": "#1f77b4",
    }
    class_color = classification_colors.get(cps_result.classification, "#888888")

    feature_rows = ""
    for fname in ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]:
        label, desc_short = FEATURE_DESCRIPTIONS[fname]
        score = getattr(features, fname)
        score_str = f"{score:.3f}" if score is not None else "—"
        contribution = cps_result.feature_contributions.get(fname, 0)
        raw = _get_raw(fname, features)
        bar_width = int(contribution * 400) if contribution else 0
        feature_rows += f"""
        <tr>
          <td><strong>{label}</strong><br><small style="color:#666">{desc_short}</small></td>
          <td style="text-align:center">{score_str}</td>
          <td>{raw}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px">
              <div style="width:{bar_width}px;height:12px;background:#0d7a7a;border-radius:2px"></div>
              <span style="font-size:0.85em">{contribution:.3f}</span>
            </div>
          </td>
        </tr>"""

    flags_html = ""
    if cps_result.confidence_flag:
        flags_html = f'<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px;margin:8px 0;border-radius:4px">⚠ <strong>Flags:</strong> {cps_result.confidence_flag}</div>'

    domain_desc = DOMAIN_DESCRIPTIONS.get(cps_result.domain, "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChanVar: CACNA1A {cps_result.aa_change}</title>
<style>
  body {{ font-family: "Times New Roman", Times, serif; max-width: 900px; margin: 40px auto; padding: 0 24px; color: #222; background: #fff; }}
  h1 {{ font-size: 1.6em; border-bottom: 2px solid #0d7a7a; padding-bottom: 8px; }}
  h2 {{ font-size: 1.2em; color: #0d7a7a; margin-top: 32px; }}
  .cps-box {{ background: #f4f8f8; border: 1px solid #0d7a7a; border-radius: 6px; padding: 20px; margin: 16px 0; }}
  .cps-score {{ font-size: 2.4em; font-weight: bold; color: {class_color}; }}
  .cps-bar-container {{ background: #e0e0e0; border-radius: 4px; height: 20px; width: 100%; margin: 10px 0; position: relative; }}
  .cps-bar {{ background: {class_color}; height: 100%; width: {cps_pct}%; border-radius: 4px; transition: width 0.3s; }}
  .ci-region {{ position: absolute; top: 0; left: {ci_low_pct}%; width: {ci_high_pct - ci_low_pct}%; height: 100%; background: rgba(0,0,0,0.15); border-radius: 4px; }}
  .classification {{ font-size: 1.1em; font-weight: bold; color: {class_color}; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
  th {{ background: #0d7a7a; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .meta {{ color: #555; font-size: 0.88em; }}
  .domain-box {{ background: #f0f7f7; border-left: 3px solid #0d7a7a; padding: 10px 14px; margin: 10px 0; font-size: 0.92em; }}
  .limitation {{ background: #fff8f0; border-left: 3px solid #ff7f0e; padding: 8px 12px; margin: 6px 0; font-size: 0.88em; }}
  .ref {{ font-size: 0.82em; color: #444; margin: 4px 0; }}
  .footer {{ margin-top: 40px; border-top: 1px solid #ddd; padding-top: 16px; font-size: 0.82em; color: #666; }}
</style>
</head>
<body>
<h1>ChanVar Annotation: CACNA1A {cps_result.aa_change}</h1>
<p class="meta">Generated {TODAY} | ChanVar v{CHANVAR_VERSION} | 
<a href="https://thexiulab.org" style="color:#0d7a7a">The Xiu Lab</a> | 
<a href="https://github.com/axshoe/chanvar" style="color:#0d7a7a">github.com/axshoe/chanvar</a></p>

<div class="cps-box">
  <div class="cps-score">{cps_result.cps:.3f}</div>
  <div class="classification">{cps_result.classification}</div>
  <div class="cps-bar-container">
    <div class="cps-bar"></div>
    <div class="ci-region" title="95% CI: {cps_result.ci_lower:.3f}–{cps_result.ci_upper:.3f}"></div>
  </div>
  <p style="font-size:0.9em;color:#555">95% CI: {cps_result.ci_lower:.3f} – {cps_result.ci_upper:.3f} &nbsp;|&nbsp; 
  Data completeness: {cps_result.data_completeness:.0%} &nbsp;|&nbsp; 
  Domain: {cps_result.domain}</p>
  {flags_html}
</div>

<h2>Variant</h2>
<table>
<tr><td><strong>gnomAD ID</strong></td><td>{cps_result.variant_id}</td></tr>
<tr><td><strong>Amino acid change</strong></td><td>{cps_result.aa_change}</td></tr>
<tr><td><strong>Residue position</strong></td><td>{features.residue_number}</td></tr>
<tr><td><strong>Functional domain</strong></td><td>{cps_result.domain}</td></tr>
<tr><td><strong>Transmembrane</strong></td><td>{'Yes' if features.is_tm_domain else 'No'}</td></tr>
{'<tr><td><strong>AlphaFold2 pLDDT</strong></td><td>' + f"{features.plddt:.1f}/100" + '</td></tr>' if features.plddt is not None else ''}
{'<tr><td><strong>ClinVar</strong></td><td>' + (cps_result.clinvar_sig or '—') + '</td></tr>'}
</table>

<div class="domain-box"><strong>Domain context:</strong> {domain_desc}</div>

<h2>Feature Breakdown</h2>
<table>
<thead><tr><th>Feature</th><th>Score</th><th>Raw Value</th><th>Contribution</th></tr></thead>
<tbody>{feature_rows}</tbody>
</table>

<h2>Interpretation</h2>
<p>
CPS {cps_result.cps:.3f} (95% CI: {cps_result.ci_lower:.3f}–{cps_result.ci_upper:.3f}) corresponds to 
<strong>{cps_result.classification}</strong> under provisional ChanVar thresholds calibrated against 
ClinVar CACNA1A classifications. This is a computational prediction and must be interpreted in 
clinical context alongside family history, electrophysiological data, and specialist review.
</p>

<h2>Limitations</h2>
<div class="limitation">CPS is a computational prediction; it does not constitute a clinical classification. Functional electrophysiological validation (patch-clamp) is required for clinical decision-making.</div>
<div class="limitation">dDDG calculations for transmembrane segments assume aqueous solvent and are less reliable than for cytoplasmic domains (domain weight adjustment applied).</div>
<div class="limitation">The ClinVar CACNA1A training set is small (~150-250 high-confidence P/LP variants). The model may underperform for variants in understudied regions.</div>
<div class="limitation">Rate4site conservation estimates use a simplified single-site model. Full tree-based estimation (ConSurf) would provide more accurate per-position rates.</div>

<h2>References</h2>
{''.join(f'<p class="ref">{text}</p>' for text in CITATIONS.values())}

<div class="footer">
ChanVar v{CHANVAR_VERSION} | Part of the MiSOF series | The Xiu Lab | thexiulab.org<br>
This tool is open-source (github.com/axshoe/chanvar) and for research use only. 
All outputs should be validated against experimental functional data before clinical application.
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _cps_bar_text(cps: float) -> str:
    """Text-based CPS visualization bar for terminal/markdown."""
    filled = int(cps * 40)
    empty = 40 - filled
    marker = _classify_marker(cps)
    return f"{'█' * filled}{'░' * empty} {marker}"


def _classify_marker(cps: float) -> str:
    if cps >= 0.85:
        return "LP"
    elif cps >= 0.70:
        return "PP"
    elif cps >= 0.40:
        return "VUS"
    elif cps >= 0.20:
        return "PB"
    return "LB"


def _get_raw(fname: str, features: VariantFeatures) -> str:
    """Get raw value string for a feature."""
    raw_map = {
        "f1": f"AF={features.gnomad_af:.2e}" if features.gnomad_af else "absent",
        "f2": f"{features.ddg_raw:.2f} kcal/mol" if features.ddg_raw is not None else "—",
        "f3": f"{features.rmsd_raw:.2f} Å" if features.rmsd_raw is not None else "—",
        "f4": f"rate={features.rate4site_raw:.2f}" if features.rate4site_raw is not None else "—",
        "f5": features.domain,
        "f6": f"AlphaMissense={features.f6:.3f}" if features.f6 is not None else "—",
        "f7": f"CADD Phred = {features.f7 * 40:.1f}" if features.f7 is not None else "—",
        "f8": f"Grantham={int((features.f8 or 0.5) * 215)}" if features.f8 is not None else "—",
    }
    return raw_map.get(fname, "—")
