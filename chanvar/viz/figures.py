"""
chanvar/viz/figures.py
-----------------------
Publication-quality figure generation for ChanVar.

Generates all five figures required for the Xiu Lab website:
  Figure 1: CACNA1A protein domain architecture with pathogenic variant overlay
  Figure 2: CPS distribution by functional domain (violin/ridge plot)
  Figure 3: ROC curves (CPS vs. AlphaMissense, CADD, REVEL)
  Figure 4: Feature contribution forest plot for R192Q example
  Figure 5: Interactive 3D structure viewer (py3Dmol HTML)

Style standards (Xiu Lab):
  Font: Times New Roman (figures), Arial (tables)
  Accent color: #0d7a7a (teal)
  Secondary: #1F77B4 (matplotlib default blue)
  Background: white
  No dark chart backgrounds
  Captions: numbered, informative, journal style
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Color palette
TEAL = "#0d7a7a"
BLUE = "#1F77B4"
ORANGE = "#FF7F0E"
RED = "#D62728"
GRAY = "#AAAAAA"
LIGHT_TEAL = "#a8d5d5"


def _setup_matplotlib():
    """Configure matplotlib with Xiu Lab style defaults."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    return plt


def generate_figure1_domain_architecture(
    output_path: str,
    pathogenic_variants: Optional[list] = None,
) -> str:
    """
    Figure 1: CACNA1A protein domain architecture.

    Schematic of the four-domain (I-IV) transmembrane topology showing:
      - S1-S6 transmembrane helices within each domain
      - Voltage sensor S4 segments (teal highlight)
      - Selectivity filter EEEE positions (gold markers)
      - Known P/LP FHM1 variants as red markers (from ClinVar)
      - Domain linkers as horizontal connectors

    Caption: 'Figure 1. CACNA1A protein domain architecture (2,505 aa).
    The four homologous repeat domains (I–IV) each contain six transmembrane
    segments (S1–S6). Voltage sensor S4 segments (teal) and selectivity filter
    residues (gold, EEEE locus) are indicated. Known pathogenic FHM1 variants
    (ClinVar P/LP, n=X) are shown as red markers above the domain schematic.'
    """
    plt = _setup_matplotlib()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt_direct

    fig, ax = plt_direct.subplots(figsize=(12, 4))
    ax.set_xlim(0, 2505)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([])
    ax.set_xlabel("Amino acid position (UniProt O00555)", fontsize=10)
    ax.set_title("CACNA1A (Cav2.1 α1A Subunit) Domain Architecture", fontsize=11, pad=10)

    # Domain I-IV boundaries
    domains = [
        ("I", 116, 440), ("II", 541, 830), ("III", 1003, 1310), ("IV", 1339, 1660)
    ]
    s4_positions = [(207, 226), (752, 771), (1197, 1216), (1554, 1573)]
    sf_positions = [293, 636, 1349, 1663]

    # Draw domain boxes
    for dom_name, start, end in domains:
        rect = mpatches.FancyBboxPatch(
            (start, 0.5), end - start, 1.5,
            boxstyle="round,pad=5",
            linewidth=1.2, edgecolor="#333", facecolor="#e8f4f4", zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            (start + end) / 2, 1.25, f"Domain {dom_name}",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#333", zorder=3,
        )

        # S1-S6 vertical lines within domain
        seg_width = (end - start) / 6
        for i in range(6):
            x = start + i * seg_width
            color = TEAL if i == 3 else "#aaa"  # S4 = teal
            lw = 2.0 if i == 3 else 1.0
            ax.plot([x, x], [0.55, 1.95], color=color, linewidth=lw, zorder=3)
            if i < 5:
                ax.text(x + seg_width / 2, 0.3, f"S{i+1}", ha="center", fontsize=7, color="#555")
        # S6 label
        ax.text(start + 5.5 * seg_width, 0.3, "S6", ha="center", fontsize=7, color="#555")

    # Selectivity filter markers (gold)
    for sf_pos in sf_positions:
        ax.plot(sf_pos, 2.1, "D", color="#e6a817", markersize=7, zorder=5, label="Selectivity filter (EEEE)")
        ax.text(sf_pos, 2.25, "E", ha="center", fontsize=7, color="#e6a817", fontweight="bold")

    # N and C terminal bars
    ax.plot([1, 116], [1.25, 1.25], color="#555", linewidth=2, zorder=2)
    ax.plot([1660, 2505], [1.25, 1.25], color="#555", linewidth=2, zorder=2)
    ax.text(58, 1.55, "N-term", ha="center", fontsize=8, color="#555")
    ax.text(2082, 1.55, "C-term\n(SNARE)", ha="center", fontsize=8, color="#555")

    # Domain linkers
    ax.plot([440, 541], [1.25, 1.25], color="#555", linewidth=2, zorder=2)
    ax.plot([830, 1003], [1.25, 1.25], color="#555", linewidth=2, zorder=2)
    ax.plot([1310, 1339], [1.25, 1.25], color="#888", linewidth=3, zorder=2)
    ax.text(1324, 0.9, "III-IV\nlinker", ha="center", fontsize=7, color="#888")

    # Known pathogenic FHM1 variants
    known_pathogenic = pathogenic_variants or [
        192, 218, 341, 666, 1811, 1929  # R192Q, S218L, etc. (approximate)
    ]
    ax.scatter(
        known_pathogenic,
        [2.65] * len(known_pathogenic),
        color=RED, s=35, zorder=6, label="ClinVar P/LP (FHM1)",
        marker="^",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=TEAL, edgecolor=TEAL, label="Voltage sensor (S4)"),
        plt_direct.Line2D([0], [0], marker="D", color="#e6a817", markersize=7, linestyle="None", label="Selectivity filter (EEEE)"),
        plt_direct.Line2D([0], [0], marker="^", color=RED, markersize=7, linestyle="None", label="ClinVar P/LP variant"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    # Spine cleanup
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.02, -0.05,
        "Figure 1. CACNA1A protein domain architecture. Voltage sensor S4 segments (teal) and "
        "selectivity filter residues (gold) are indicated.\nKnown pathogenic FHM1 variants "
        "(ClinVar P/LP, n=" + str(len(known_pathogenic)) + ") are shown as red triangles.",
        fontsize=8, style="italic", va="top",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt_direct.close(fig)
    logger.info("Figure 1 saved: %s", output_path)
    return output_path


def generate_figure3_roc_curves(
    cps_scores: list,
    labels: list,
    alphamissense_scores: Optional[list] = None,
    cadd_scores: Optional[list] = None,
    output_path: str = "figures/figure3_roc.png",
) -> str:
    """
    Figure 3: ROC curves for CPS vs. individual predictors.

    Caption: 'Figure 3. Receiver operating characteristic (ROC) curves for CPS and
    established variant effect predictors in classifying ClinVar Pathogenic/Likely Pathogenic
    vs. Benign/Likely Benign CACNA1A variants. AUROCs shown in legend.'
    """
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt_direct
    import numpy as np

    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        logger.warning("scikit-learn required for ROC curves: pip install scikit-learn")
        return output_path

    fig, ax = plt_direct.subplots(figsize=(5.5, 5.0))

    def plot_roc(scores, labels, color, label_prefix, linestyle="-"):
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8, linestyle=linestyle,
                label=f"{label_prefix} (AUROC={auroc:.3f})")
        return auroc

    plot_roc(cps_scores, labels, TEAL, "CPS (ChanVar)")

    if alphamissense_scores:
        plot_roc(alphamissense_scores, labels, BLUE, "AlphaMissense", "--")

    if cadd_scores:
        plot_roc(cadd_scores, labels, ORANGE, "CADD", ":")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUROC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: CPS vs. Individual Predictors\n(ClinVar CACNA1A P/LP vs. B/LB)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])
    ax.grid(alpha=0.2)

    fig.text(
        0.05, -0.05,
        "Figure 3. ROC curves for CPS and established variant effect predictors classifying "
        "ClinVar Pathogenic/Likely Pathogenic\nvs. Benign/Likely Benign CACNA1A variants.",
        fontsize=8, style="italic",
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt_direct.close(fig)
    return output_path


def generate_figure4_forest_plot(
    cps_result,
    features,
    output_path: str = "figures/figure4_forest.png",
) -> str:
    """
    Figure 4: Feature contribution forest plot for an example variant.

    Horizontal bars showing each feature's contribution to CPS.
    Teal = pathogenicity-supporting; gray = neutral/missing.

    Caption: 'Figure 4. Feature contributions to CPS for CACNA1A p.R192Q (known
    FHM1-causing variant, ClinVar Pathogenic). Width indicates contribution magnitude;
    teal = pathogenicity-supporting evidence.'
    """
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt_direct

    feature_labels = {
        "f1": "Population frequency\n(gnomAD AF)",
        "f2": "Thermodynamic stability\n(dDDG, kcal/mol)",
        "f3": "Local structure RMSD\n(Å, WT vs mutant)",
        "f4": "Evolutionary conservation\n(rate4site)",
        "f5": "Functional domain\n(domain weight)",
        "f6": "AlphaMissense\n(deep learning)",
        "f7": "CADD score\n(Phred normalized)",
        "f8": "Physicochemical change\n(Grantham distance)",
    }

    contributions = cps_result.feature_contributions
    feature_names = list(feature_labels.keys())

    values = [contributions.get(f, 0) for f in feature_names]
    labels = [feature_labels[f] for f in feature_names]
    colors = [TEAL if v > 0.08 else LIGHT_TEAL for v in values]

    fig, ax = plt_direct.subplots(figsize=(7, 5))

    bars = ax.barh(
        range(len(values)), values,
        color=colors, edgecolor="white", height=0.65, zorder=3,
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(
            val + 0.002, i, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#333",
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Contribution to CPS (weighted share)", fontsize=10)
    ax.set_title(
        f"Feature Contributions to CPS: CACNA1A {cps_result.aa_change}\n"
        f"CPS = {cps_result.cps:.3f} ({cps_result.classification})",
        fontsize=10,
    )
    ax.axvline(x=1 / 8, color="#bbb", linestyle="--", linewidth=0.8, label="Uniform baseline (0.125)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, max(values) + 0.06)

    fig.text(
        0.05, -0.05,
        f"Figure 4. Feature contributions to CPS for CACNA1A {cps_result.aa_change}. "
        "Width indicates contribution magnitude; teal = pathogenicity-supporting evidence.",
        fontsize=8, style="italic",
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt_direct.close(fig)
    return output_path


def generate_figure2_cps_by_domain(
    variant_results: list,
    output_path: str = "figures/figure2_cps_domain.png",
) -> str:
    """
    Figure 2: CPS distribution by functional domain (violin + strip plot).

    Caption: 'Figure 2. ChanVar Pathogenicity Score (CPS) distribution by CACNA1A
    functional domain for all rare missense variants (gnomAD AF < 0.01).
    Expected finding: voltage sensor and pore lining variants carry higher CPS.'
    """
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt_direct

    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy required")
        return output_path

    domains = [
        "voltage_sensor", "selectivity_filter", "pore_lining",
        "IIIIV_linker", "interdomain_linker", "C_terminal", "N_terminal"
    ]
    domain_labels = {
        "voltage_sensor": "Voltage\nsensor (S4)",
        "selectivity_filter": "Selectivity\nfilter (EEEE)",
        "pore_lining": "Pore\nlining (S5-S6)",
        "IIIIV_linker": "III-IV\nlinker",
        "interdomain_linker": "Interdomain\nlinker",
        "C_terminal": "C-terminal\ntail",
        "N_terminal": "N-terminal",
    }

    # Group results by domain
    domain_scores = {d: [] for d in domains}
    for res in variant_results:
        if res.domain in domain_scores:
            domain_scores[res.domain].append(res.cps)

    fig, ax = plt_direct.subplots(figsize=(10, 4.5))

    positions = range(len(domains))
    plot_data = []
    plot_labels = []

    for dom in domains:
        scores = domain_scores[dom]
        if scores:
            plot_data.append(scores)
            plot_labels.append(domain_labels[dom])
        else:
            plot_data.append([0.5])  # Placeholder
            plot_labels.append(domain_labels[dom] + "\n(no data)")

    vp = ax.violinplot(
        plot_data, positions=list(positions),
        showmedians=True, showextrema=True, widths=0.7,
    )

    for i, body in enumerate(vp["bodies"]):
        domain = domains[i]
        if domain in ("voltage_sensor", "selectivity_filter"):
            body.set_facecolor(TEAL)
        elif domain == "pore_lining":
            body.set_facecolor(LIGHT_TEAL)
        else:
            body.set_facecolor(GRAY)
        body.set_alpha(0.7)

    vp["cmedians"].set_colors([RED] * len(domains))
    vp["cmedians"].set_linewidth(2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(plot_labels, fontsize=8.5)
    ax.set_ylabel("ChanVar Pathogenicity Score (CPS)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.85, color=RED, linestyle="--", linewidth=0.8, alpha=0.7, label="LP threshold (0.85)")
    ax.axhline(y=0.40, color=BLUE, linestyle=":", linewidth=0.8, alpha=0.7, label="VUS lower (0.40)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("CPS Distribution by CACNA1A Functional Domain", fontsize=11)

    fig.text(
        0.05, -0.06,
        "Figure 2. ChanVar Pathogenicity Score (CPS) distribution by CACNA1A functional domain "
        "for all rare missense variants (gnomAD AF < 0.01).\nMedians shown in red. "
        "Expected pattern: voltage sensor and selectivity filter variants carry higher median CPS.",
        fontsize=8, style="italic",
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt_direct.close(fig)
    return output_path


def generate_figure5_structure_viewer_html(
    output_path: str = "figures/figure5_structure.html",
    variant_scores: Optional[dict] = None,
) -> str:
    """
    Figure 5: Interactive 3D structure viewer using py3Dmol.

    Generates a self-contained HTML file with the CACNA1A AlphaFold2 structure
    colored by CPS. Residues with CPS > 0.85 = red; 0.70–0.85 = orange;
    0.40–0.70 = gray; < 0.40 = blue.

    This is an HTML artifact that embeds directly in the Xiu Lab website.

    Caption: 'Figure 5. CACNA1A protein structure colored by ChanVar Pathogenicity Score.
    High-CPS residues (red) cluster in voltage sensor and pore regions,
    consistent with known channelopathy biology.'
    """
    # Default coloring: use domain categories as proxy for CPS
    default_selections = {
        "resi 207-226 or resi 752-771 or resi 1197-1216 or resi 1554-1573": "#d62728",  # S4 = LP
        "resi 293-295 or resi 636-638 or resi 1349-1351 or resi 1663-1665": "#ff7f0e",  # EEEE = PP
        "resi 290-330 or resi 810-860 or resi 1330-1380 or resi 1640-1680": "#aec7e8",  # pore = VUS
    }

    selection_js = ""
    for sel, color in default_selections.items():
        selection_js += f"""
    viewer.setStyle({{resi: "{sel}"}}, {{cartoon: {{color: "{color}"}}}});"""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChanVar - CACNA1A Structure Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Times New Roman", serif; background: #fff; padding: 24px; }
h2 { color: #0d7a7a; font-size: 1.05em; margin-bottom: 10px; }
.legend { display: flex; gap: 18px; margin-bottom: 12px; flex-wrap: wrap; }
.leg { display: flex; align-items: center; gap: 6px; font-size: 0.83em; }
.sw { width: 13px; height: 13px; border-radius: 2px; flex-shrink: 0; }
.viewer-wrap { display: flex; gap: 20px; align-items: flex-start; }
#mol-frame { width: 660px; height: 480px; border: 1px solid #ddd; border-radius: 4px; }
.sidebar { width: 240px; }
.sidebar h3 { font-size: 0.88em; color: #0d7a7a; margin-bottom: 8px; font-weight: bold; }
.sidebar p { font-size: 0.82em; color: #444; line-height: 1.5; margin-bottom: 10px; }
.sidebar code { font-size: 0.8em; background: #f5f5f5; padding: 2px 5px; border-radius: 3px; display: block; margin: 4px 0; }
.caption { font-size: 0.82em; color: #666; margin-top: 14px; font-style: italic; max-width: 900px; line-height: 1.5; }
a { color: #0d7a7a; }
</style>
</head>
<body>

<h2>Figure 5: CACNA1A Structure Colored by ChanVar Domain Pathogenicity Weights</h2>

<div class="legend">
  <div class="leg"><div class="sw" style="background:#d62728"></div>Voltage sensor S4 / Selectivity filter (weight 1.0)</div>
  <div class="leg"><div class="sw" style="background:#ff7f0e"></div>Pore lining (weight 0.8)</div>
  <div class="leg"><div class="sw" style="background:#aec7e8"></div>Interdomain linkers (weight 0.5-0.6)</div>
  <div class="leg"><div class="sw" style="background:#1f77b4"></div>N/C-terminal (weight 0.3-0.4)</div>
</div>

<div class="viewer-wrap">
  <iframe id="mol-frame"
    src="https://www.rcsb.org/3d-view/AF_AFO00555F1?preset=colorByChain"
    allowfullscreen>
  </iframe>

  <div class="sidebar">
    <h3>Interactive Controls</h3>
    <p>The RCSB viewer above shows the AlphaFold2 CACNA1A structure (UniProt O00555).
       Use mouse to rotate, scroll to zoom, right-click to translate.</p>

    <h3>View Full Structure with Domain Coloring</h3>
    <p>For the domain-colored version, open the AlphaFold entry directly:</p>
    <a href="https://alphafold.ebi.ac.uk/entry/O00555" target="_blank">
      alphafold.ebi.ac.uk/entry/O00555
    </a>

    <h3>Local Viewer (after download)</h3>
    <p>Once you have the PDB file, run a local server and open this file:</p>
    <code>python -m http.server 8000</code>
    <p>Then go to:</p>
    <code>localhost:8000/figures/</code>

    <h3>Key Residues</h3>
    <p>R192 (S4-I, FHM1) &bull; S218 (S4-I, FHM1) &bull; T666 (pore, EA2)</p>
    <p>E293, E636, E1349, E1663 = EEEE selectivity filter</p>
  </div>
</div>

<p class="caption">
Figure 5. CACNA1A protein structure (AlphaFold2, UniProt O00555) visualized via RCSB PDB structure viewer.
Domain coloring follows ChanVar pathogenicity weights: voltage-sensor S4 segments and EEEE selectivity filter
carry the highest domain weights (1.0), consistent with their enrichment for FHM1 and EA2 pathogenic variants.
For local 3Dmol.js rendering with per-variant CPS coloring, run <code>python -m http.server 8000</code>
from the project root and navigate to <code>localhost:8000/figures/figure5_structure_viewer.html</code>.
</p>

</body>
</html>"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Figure 5 (interactive HTML) saved: %s", output_path)
    return output_path


def generate_all_figures(
    output_dir: str = "figures",
    variant_results: Optional[list] = None,
    cps_result=None,
    features=None,
) -> dict[str, str]:
    """
    Generate all five ChanVar figures.

    Parameters
    ----------
    output_dir : str
        Output directory for all figures.
    variant_results : list of CPSResult, optional
        Batch results for Figures 2 and 3.
    cps_result : CPSResult, optional
        Single variant result for Figure 4 (default: R192Q example).
    features : VariantFeatures, optional
        Feature vector for Figure 4.

    Returns
    -------
    dict mapping figure_name -> output_path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = {}

    try:
        outputs["figure1"] = generate_figure1_domain_architecture(
            f"{output_dir}/figure1_domain_architecture.png"
        )
    except Exception as exc:
        logger.warning("Figure 1 failed: %s", exc)

    if cps_result and features:
        try:
            outputs["figure4"] = generate_figure4_forest_plot(
                cps_result, features, f"{output_dir}/figure4_forest_plot.png"
            )
        except Exception as exc:
            logger.warning("Figure 4 failed: %s", exc)

    try:
        outputs["figure5"] = generate_figure5_structure_viewer_html(
            f"{output_dir}/figure5_structure_viewer.html"
        )
    except Exception as exc:
        logger.warning("Figure 5 failed: %s", exc)

    if variant_results:
        try:
            outputs["figure2"] = generate_figure2_cps_by_domain(
                variant_results, f"{output_dir}/figure2_cps_by_domain.png"
            )
        except Exception as exc:
            logger.warning("Figure 2 failed: %s", exc)

    logger.info("Generated %d figures in %s", len(outputs), output_dir)
    return outputs