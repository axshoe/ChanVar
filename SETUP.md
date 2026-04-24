# ChanVar — Setup and Usage Guide

**In Silico Functional Annotation of Rare CACNA1A Variants in Familial Hemiplegic Migraine**  
The Xiu Lab · thexiulab.org · github.com/axshoe/ChanVar

This guide walks you through everything from cloning the repo to running your first annotation,
generating all five figures, and pushing to GitHub. It assumes you are starting from scratch
and have never set up a Python research project before.

---

## Table of Contents

1. [What you need before starting](#1-what-you-need-before-starting)
2. [Clone the repository](#2-clone-the-repository)
3. [Create a virtual environment](#3-create-a-virtual-environment)
4. [Install Python dependencies](#4-install-python-dependencies)
5. [Verify the installation](#5-verify-the-installation)
6. [Run the test suite](#6-run-the-test-suite)
7. [Your first annotation — no external tools required](#7-your-first-annotation--no-external-tools-required)
8. [Understanding the output files](#8-understanding-the-output-files)
9. [Generate all five figures](#9-generate-all-five-figures)
10. [Run a batch of variants](#10-run-a-batch-of-variants)
11. [Install FoldX (optional but recommended)](#11-install-foldx-optional-but-recommended)
12. [Install EvoEF2 (open-source alternative to FoldX)](#12-install-evoef2-open-source-alternative-to-foldx)
13. [Install MUSCLE for conservation alignment](#13-install-muscle-for-conservation-alignment)
14. [Full analysis run — all features enabled](#14-full-analysis-run--all-features-enabled)
15. [Train weights on ClinVar data](#15-train-weights-on-clinvar-data)
16. [Push to GitHub](#16-push-to-github)
17. [Troubleshooting](#17-troubleshooting)
18. [What each file in the project does](#18-what-each-file-in-the-project-does)

---

## 1. What you need before starting

### Check Python

ChanVar requires Python 3.9 or later. Open PowerShell and run:

```powershell
python --version
```

You need to see `Python 3.9.x`, `3.10.x`, `3.11.x`, or `3.12.x`. If you see `Python 2.x.x`
or an error, go to python.org, download Python 3.11, and during installation check the box
that says **"Add Python to PATH"**. Then close and reopen PowerShell and check again.

### Check Git

```powershell
git --version
```

If you get an error, go to git-scm.com and install Git. During installation, accept all
defaults. Close and reopen PowerShell after installing.

### Check pip

```powershell
pip --version
```

pip is Python's package installer. It comes with Python 3.9+. If it is missing, run:

```powershell
python -m ensurepip --upgrade
```

---

## 2. Clone the repository

Navigate to wherever you want the project to live. For example, your Documents folder:

```powershell
cd C:\Users\YourName\Documents
```

Then clone:

```powershell
git clone https://github.com/axshoe/ChanVar.git
cd ChanVar
```

You should now be inside a folder called `ChanVar`. Confirm with:

```powershell
dir
```

You should see: `chanvar`, `data`, `figures`, `tests`, `pyproject.toml`, `README.md`, `SETUP.md`.

---

## 3. Create a virtual environment

A virtual environment is a clean, isolated Python installation specific to this project.
It prevents ChanVar's dependencies from clashing with anything else you have installed.
Always do this for research Python projects.

```powershell
python -m venv .venv
```

This creates a hidden folder `.venv` inside ChanVar. Now activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Every time you open a new PowerShell window
to work on ChanVar, run this activation line again before doing anything else.

**If you get an execution policy error**, run this once and then try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 4. Install Python dependencies

With `(.venv)` showing in your prompt:

```powershell
pip install -e ".[dev]"
```

What this does:
- `pip install` — installs packages
- `-e` — editable mode: changes to source files take effect immediately without reinstalling
- `".[dev]"` — installs this project plus the dev extras (pytest, pytest-cov)

This downloads and installs: numpy, scipy, pandas, biopython, requests, scikit-learn,
matplotlib, xgboost, shap, pytest, and their sub-dependencies. Takes 1-3 minutes.

When finished you should see: `Successfully installed chanvar-1.0.0 ...`

---

## 5. Verify the installation

**Check the chanvar command works:**

```powershell
chanvar --help
```

Should print usage information starting with `usage: chanvar [-h] {annotate,batch,train,validate}`.
If you see "command not found," the venv is not active — run `.venv\Scripts\Activate.ps1`.

**Check the core module imports:**

```powershell
python -c "import chanvar; print('OK')"
```

Should print `OK`.

**Check the bundled data files exist:**

```powershell
python -c "
from pathlib import Path
print('Grantham matrix:', 'OK' if Path('chanvar/data/grantham_matrix.json').exists() else 'MISSING')
print('Conservation TSV:', 'OK' if Path('chanvar/data/cacna1a_conservation.tsv').exists() else 'MISSING')
"
```

Both should print `OK`. These files ship with the repo and must never be deleted.

---

## 6. Run the test suite

ChanVar has 58 unit tests that verify the math is correct. Run them all:

```powershell
python -m pytest tests/ -v
```

The `-v` flag prints each test name and its pass/fail status. You should see:

```
58 passed in X.Xs
```

**Do not proceed until all 58 pass.** If any fail, the math is wrong and the scores are
meaningless. Go to Section 17 (Troubleshooting) or reinstall from scratch.

### What the tests verify:

- Grantham distance R→Q = exactly 43 (matching the 1974 paper's Table 1)
- Allele frequency 0.0 → frequency feature 1.0
- Allele frequency ≥ 0.01 → frequency feature 0.0
- Frequency feature is strictly monotone (commoner = lower score)
- Perfect conservation → Shannon entropy = 0
- Uniform distribution across all 20 amino acids → Shannon entropy = log2(20)
- R192Q scores above 0.65 with simulated feature inputs (pre-registered threshold)
- A synthetic common variant scores below 0.50
- Sigmoid inflection at the correct values for f2 and f3
- All five classification threshold bins return the right label
- 47 more checks of this kind

---

## 7. Your first annotation — no external tools required

You do not need FoldX, EvoEF2, or MUSCLE for a basic run. Missing features default to
neutral (0.5) and data_completeness reflects what is absent. You still get a valid CPS
with a wider confidence interval.

### Offline run — no internet needed:

```powershell
chanvar annotate --variant R192Q --af 0.0 --cadd 32.5 --output results\
```

- `--variant R192Q` — the amino acid change. Format: [ref amino acid][position][alt amino acid].
  R = arginine, 192 = position in the protein, Q = glutamine.
- `--af 0.0` — gnomAD allele frequency. 0.0 means this variant is not observed in 730,000
  sequenced humans, which is strong evidence it could be pathogenic.
- `--cadd 32.5` — CADD-PHRED score. PHRED 32.5 means this variant is in the top 0.05%
  most damaging variants genome-wide. You can look this up at cadd.gs.washington.edu.
- `--output results\` — folder for output files, created automatically if it does not exist.

This creates three files: `results\R192Q.json`, `results\R192Q.md`, `results\R192Q.html`.

### With live gnomAD lookup (requires internet):

```powershell
chanvar annotate --variant R192Q --output results\
```

ChanVar queries gnomAD's API automatically. If the API is slow, pass `--af 0.0` to skip it.

### With all available feature values:

```powershell
chanvar annotate ^
  --variant R192Q ^
  --af 0.0 ^
  --ddg 2.1 ^
  --rmsd 1.4 ^
  --cadd 32.5 ^
  --clinvar-sig Pathogenic ^
  --output results\
```

The `^` at the end of each line is PowerShell's line continuation character. You can also
write this as one long line without the `^`.

- `--ddg 2.1` — FoldX ΔΔG in kcal/mol. Positive = destabilizing. 2.1 is a moderately
  destabilizing mutation.
- `--rmsd 1.4` — local C-alpha RMSD in Angstroms. Above 1.0 Å is considered structurally
  significant.
- `--clinvar-sig Pathogenic` — if ClinVar already has a high-confidence classification,
  pass it here and it will override the computational score.

---

## 8. Understanding the output files

### The JSON file

Open `results\R192Q.json` in any text editor. Key fields:

```json
{
  "cps": 0.718,
  "ci_lower": 0.631,
  "ci_upper": 0.760,
  "classification": "Possibly Pathogenic",
  "domain": "voltage_sensor",
  "data_completeness": 0.75,
  "confidence_flag": null,
  "feature_contributions": {
    "f1": 0.310,
    "f2": 0.000,
    "f3": 0.000,
    ...
  }
}
```

- **cps** — the score, 0 to 1. Higher means more likely pathogenic.
- **ci_lower / ci_upper** — 95% confidence interval from 1,000 bootstrap iterations.
  A narrow CI (e.g., 0.71 to 0.74) means the score is robust. A wide CI (e.g., 0.40 to 0.80)
  means substantial uncertainty — usually because several features are missing.
- **classification** — one of: Likely Pathogenic (≥0.85), Possibly Pathogenic (0.70-0.85),
  VUS (0.40-0.70), Possibly Benign (0.20-0.40), Likely Benign (<0.20)
- **domain** — which part of the Cav2.1 channel the variant is in. voltage_sensor means
  it is in the S4 segment — the highest-weight domain.
- **data_completeness** — fraction of the 8 features with real data. 1.0 = all 8 present.
  0.75 = 6 of 8 present (typical when FoldX is not installed).
- **confidence_flag** — null if everything is fine. LOW_DATA if fewer than 7 features
  are available. CLINVAR_OVERRIDE if a high-quality ClinVar classification replaced
  the computational score.
- **feature_contributions** — how many CPS points each feature contributed. These sum to
  the CPS value itself (not to 1.0). If f1_contribution is 0.31, allele frequency
  contributed 0.31 out of the total score of (say) 0.72. Missing features show 0.00.

### The HTML file

Double-click `results\R192Q.html` to open it in your browser. It shows:
- A color-coded CPS bar (red = high/pathogenic end, blue = low/benign end)
- A bar chart of feature contributions — which features drove the score
- Domain description — what the voltage_sensor domain is and why it matters
- Limitations section specific to this variant's data completeness

This is the most readable output and the right one to share.

### The Markdown file

Plain text with tables and headers. Useful for pasting into lab notebooks, GitHub issues,
or anywhere that renders Markdown.

---

## 9. Generate all five figures

matplotlib was installed in Step 4. Run:

```powershell
python -c "
import matplotlib
matplotlib.use('Agg')
from chanvar.viz.figures import generate_all_figures
paths = generate_all_figures(output_dir='figures')
for p in paths:
    print('Generated:', p)
"
```

Output files in `figures/`:

| File | What it shows |
|------|---------------|
| `figure1_domain_architecture.png` | Schematic of the Cav2.1 four-domain protein with S4 voltage sensors (teal), EEEE selectivity filter (gold), and known FHM1 pathogenic variant positions (red triangles) |
| `figure2_cps_by_domain.png` | Violin plots of CPS distribution per functional domain. Voltage-sensor variants cluster high; C-terminal variants cluster low. |
| `figure3_roc_curves.png` | ROC curves: CPS vs. AlphaMissense, CADD, REVEL. Higher AUC = better separation of pathogenic from benign variants. Currently uses synthetic calibration data — see note below. |
| `figure4_forest_plot.png` | Horizontal bar chart showing R192Q feature contributions. |
| `figure5_structure_viewer.html` | Interactive 3D Cav2.1 structure colored by domain. Open in any browser. Requires internet on first load for the 3Dmol.js viewer. |

**Note on Figures 2 and 3:** These use synthetic data calibrated to ClinVar variant
distributions until you run the full annotation pipeline on real variants (Step 14).
After that run, regenerate them following the instructions in Step 14.

Open the structure viewer:

```powershell
Start-Process "figures\figure5_structure_viewer.html"
```

---

## 10. Run a batch of variants

Create a file called `my_variants.csv` in the ChanVar directory using any text editor:

```
aa_change,af,cadd
R192Q,0.0,32.5
S218L,0.0,34.1
T666M,0.0,29.8
A454T,0.000012,22.4
```

The header row is required. `aa_change` is required. `af` and `cadd` are optional —
leave the cell blank if you do not have the value.

Run:

```powershell
chanvar batch --input my_variants.csv --output batch_results\
```

Produces one JSON + MD + HTML per variant, plus `batch_results\batch_summary.json`
with all scores combined for easy comparison.

---

## 11. Install FoldX (optional but recommended)

FoldX computes the ΔΔG thermodynamic stability feature (f2). Without it, f2 is neutral
(0.5) and data_completeness drops to 0.75 at best. For serious variant analysis, install it.

**Step 1:** Go to foldxsuite.crg.eu and create a free academic account. Download the
FoldX binary for Windows. You receive a ZIP file.

**Step 2:** Extract the ZIP. The folder must contain both:
- `foldx.exe` (sometimes named `FoldX.exe`)
- `rotabase.txt` — required, must be in the same folder as the executable

Place both at e.g. `C:\tools\foldx\`.

**Step 3:** Tell ChanVar where FoldX lives. In PowerShell:

```powershell
$env:FOLDX_PATH = "C:\tools\foldx\foldx.exe"
```

To make this permanent across sessions, open your PowerShell profile:

```powershell
notepad $PROFILE
```

If the file does not exist, PowerShell will ask to create it — say yes. Add this line
and save:

```
$env:FOLDX_PATH = "C:\tools\foldx\foldx.exe"
```

**Step 4:** Verify FoldX is accessible:

```powershell
& $env:FOLDX_PATH --help
```

Should print FoldX's help output.

**Step 5:** Test ChanVar can call it:

```powershell
python -c "from chanvar.structure.stability import run_foldx_ddg; print('FoldX: ready')"
```

**What FoldX does when ChanVar calls it:**

1. First run on any structure: ChanVar runs `foldx --command=RepairPDB` on the AlphaFold2
   PDB file. This takes 2-5 minutes and fixes minor structural issues. The repaired structure
   is cached — this step only happens once per structure.
2. For each variant: ChanVar runs `foldx --command=BuildModel` with a mutation list file.
   This takes 1-3 minutes per variant.
3. FoldX outputs a `Dif_BuildModel_*.fxout` file containing the ΔΔG value, which ChanVar
   reads and converts to the f2 feature.

---

## 12. Install EvoEF2 (open-source alternative to FoldX)

EvoEF2 is an open-source stability predictor with similar accuracy to FoldX. On Windows
it requires WSL2 (Windows Subsystem for Linux). If you do not want to set up WSL2,
use FoldX instead and skip this section.

**With WSL2 installed:**

```bash
git clone https://github.com/tommyhuangthu/EvoEF2.git
cd EvoEF2
g++ -O3 -o EvoEF2 src/*.cpp
```

In PowerShell:

```powershell
$env:EVOEF2_PATH = "wsl /path/to/EvoEF2/EvoEF2"
```

ChanVar prefers FoldX when both are available. To force EvoEF2:

```powershell
$env:PREFER_EVOEF2 = "1"
```

---

## 13. Install MUSCLE for conservation alignment

MUSCLE aligns CACNA1A sequences across 87 vertebrate species to compute the evolutionary
conservation score (f4). The pre-computed conservation file
(`chanvar/data/cacna1a_conservation.tsv`) covers ~100 critical positions and ships with
the repo. You only need MUSCLE if you want to recompute or extend that file.

**Download:** Go to drive5.com/muscle, download the muscle5 binary for Windows.

Place `muscle.exe` at e.g. `C:\tools\muscle\` and set the path:

```powershell
$env:MUSCLE_PATH = "C:\tools\muscle\muscle.exe"
```

Verify:

```powershell
muscle -version
```

**To recompute the full conservation file:**

```powershell
chanvar compute-conservation --output chanvar\data\cacna1a_conservation.tsv
```

This fetches 87 vertebrate CACNA1A orthologs from Ensembl, runs MUSCLE alignment, and
computes JSD per position. Takes 10-30 minutes depending on Ensembl API response times.

---

## 14. Full analysis run — all features enabled

With FoldX installed, this is the complete pipeline for a single variant:

```powershell
chanvar annotate ^
  --variant R192Q ^
  --output results\full\ ^
  --format json,markdown,html ^
  --bootstrap-n 1000
```

On the first run with the AlphaFold2 structure, ChanVar:
1. Downloads `AF-O00555-F1-model_v4.pdb` from the AlphaFold EBI server (cached after
   first download at `C:\Users\YourName\.chanvar\cache\structures\`)
2. Runs FoldX RepairPDB on it — takes 2-5 minutes, cached for subsequent runs
3. Runs FoldX BuildModel for R192Q — takes 1-3 minutes
4. Queries gnomAD for allele frequency
5. Computes local RMSD using BioPython
6. Looks up conservation from the TSV file
7. Queries ClinVar
8. Assembles ACMG codes
9. Computes CPS, 1000-iteration bootstrap, outputs three report files

You should see `data_completeness: 1.0` in the JSON output when all features are present.

### Benchmark run — three known pathogenic variants:

Create `benchmark.csv`:

```
aa_change,af,cadd
R192Q,0.0,32.5
S218L,0.0,34.7
T666M,0.0,29.8
```

Run:

```powershell
chanvar batch --input benchmark.csv --output results\benchmark\
```

Expected results:
- R192Q: CPS approximately 0.79-0.83, classified Possibly Pathogenic or Likely Pathogenic
- S218L: CPS approximately 0.81-0.85, classified Likely Pathogenic
- T666M: CPS approximately 0.71-0.77, classified Possibly Pathogenic

If any of these are far outside these ranges with full data completeness, something
is wrong with the feature inputs. Check the feature_contributions in the JSON output
to see which feature is anomalous.

### Regenerate Figure 4 with real results:

```powershell
python -c "
import matplotlib, json
matplotlib.use('Agg')
from chanvar.viz.figures import generate_figure4_forest_plot
with open('results/benchmark/R192Q.json') as f:
    result = json.load(f)
generate_figure4_forest_plot(result, output_dir='figures')
print('Figure 4 regenerated with real R192Q data')
"
```

### Regenerate Figures 2 and 3 after a full batch run:

Once you have annotated a substantial set of variants (ideally the full ClinVar CACNA1A
set), you can regenerate Figures 2 and 3 from real data by passing the batch summary JSON
to the figure generators. The exact command will depend on your batch summary file path:

```powershell
python -c "
import matplotlib, json
matplotlib.use('Agg')
from chanvar.viz.figures import generate_figure2_cps_by_domain, generate_figure3_roc_curves
with open('results/clinvar_batch/batch_summary.json') as f:
    summary = json.load(f)
generate_figure2_cps_by_domain(summary, output_dir='figures')
generate_figure3_roc_curves(summary, output_dir='figures')
print('Figures 2 and 3 regenerated with real data')
"
```

---

## 15. Train weights on ClinVar data

The default feature weights are literature-informed initializations. Fitted weights from
logistic regression on ClinVar data are the intended primary derivation method and should
replace the defaults for any publication or downstream analysis.

**Step 1:** Download the ClinVar VCF from NCBI. In your browser, go to:

```
https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
```

Download the file (it is approximately 500 MB) and place it in your ChanVar directory.

**Step 2:** Filter to CACNA1A only (optional but faster):

```powershell
python -c "
import gzip, re
with gzip.open('clinvar.vcf.gz', 'rt') as fin, open('clinvar_cacna1a.vcf', 'w') as fout:
    for line in fin:
        if line.startswith('#') or 'CACNA1A' in line:
            fout.write(line)
print('Done')
"
```

**Step 3:** Train the weights:

```powershell
chanvar train --clinvar clinvar_cacna1a.vcf --output fitted_weights.json
```

ChanVar will:
1. Parse the VCF and extract CACNA1A variants with ≥2 review stars
2. Split into P/LP positive set and B/LB negative set
3. Annotate each through the feature pipeline (this takes a while — ~200-300 variants)
4. Fit L2-regularized logistic regression with 5-fold cross-validation
5. Print the cross-validated AUROC
6. Save fitted weights to `fitted_weights.json`

**Step 4:** Use fitted weights:

```powershell
chanvar annotate --variant R192Q --weights fitted_weights.json --output results\
```

The cross-validated AUROC and the fitted weights must be reported alongside CPS output
in any publication.

---

## 16. Push to GitHub

### First-time setup:

```powershell
git config --global user.name "Angie Xiu"
git config --global user.email "angie.xiu27@gmail.com"
```

### Check what has changed:

```powershell
git status
```

This shows modified and untracked files since your last commit.

### Set up .gitignore (do this before your first commit):

Create a file called `.gitignore` in the ChanVar directory:

```powershell
notepad .gitignore
```

Paste this content and save:

```
# Virtual environment — never commit this
.venv/

# Python build artifacts
*.egg-info/
__pycache__/
*.pyc
*.pyo
dist/
build/

# Result files — too large for Git; upload to Zenodo instead
results/
*.fxout

# Downloaded structures — regenerated on demand
chanvar/data/*.pdb
*.pdb

# Temporary figures (keep the HTML structure viewer source but not PNGs)
figures/*.png

# AlphaFold cache
.chanvar/

# Keep these tracked — they ship with the repo
!chanvar/data/grantham_matrix.json
!chanvar/data/cacna1a_conservation.tsv
```

### Stage your changes:

```powershell
git add chanvar/
git add tests/
git add pyproject.toml
git add README.md
git add SETUP.md
git add .gitignore
```

Do not `git add .` (everything) until you have confirmed `.gitignore` is correct —
you do not want to accidentally push 500 MB of downloaded data.

### Commit:

```powershell
git commit -m "Initial ChanVar v1.0 release"
```

Good commit messages say what changed. Examples:
- `"Fix S4_I boundary to include R192 (181-226 not 207-226)"`
- `"Add pLDDT gating to f3 RMSD feature"`
- `"Regenerate benchmark figures with FoldX full run"`
- `"Train logistic regression weights on ClinVar Jan 2026 set; AUROC 0.X"`

### Connect to GitHub and push:

If the remote is not set yet:

```powershell
git remote add origin https://github.com/axshoe/ChanVar.git
git branch -M main
git push -u origin main
```

If the remote is already set:

```powershell
git push origin main
```

GitHub will ask for your username and a password. **Use a Personal Access Token, not
your GitHub account password.** To create one:
1. Go to github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Click "Generate new token"
3. Give it a name (e.g. "ChanVar push"), set expiration to 90 days, give it Contents
   read and write permission for the ChanVar repository
4. Click Generate token, copy it immediately — you cannot see it again
5. Paste it as the password when Git prompts you

To avoid being asked for credentials every push, store them:

```powershell
git config --global credential.helper store
```

The next time you push and enter your token, Git will remember it.

---

## 17. Troubleshooting

### `chanvar: command not found`

The virtual environment is not active. Run:
```powershell
.venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError: No module named 'chanvar'`

Package is not installed. With venv active:
```powershell
pip install -e ".[dev]"
```

### Tests fail on Grantham distance values

The `grantham_matrix.json` file is missing or corrupted. Check:
```powershell
python -c "
import json
d = json.load(open('chanvar/data/grantham_matrix.json'))
print('R->Q:', d['matrix']['R']['Q'])
print('R->W:', d['matrix']['R']['W'])
"
```
Should print `43` and `101`. If it errors or prints other values, restore the file
from the GitHub repository: go to the file on github.com/axshoe/ChanVar and
click "Download raw file."

### gnomAD API returns an error or times out

gnomAD's API is occasionally slow or rate-limited. Provide the allele frequency manually:
```powershell
chanvar annotate --variant R192Q --af 0.0 --output results\
```
For any variant, look up the AF manually at gnomad.broadinstitute.org by searching
the protein change (e.g., "R192Q" in the CACNA1A gene search), then pass it with `--af`.

### AlphaFold2 structure download fails

Download manually:
1. Go to alphafold.ebi.ac.uk/entry/O00555
2. Click Download → PDB format
3. Save as `AF-O00555-F1-model_v4.pdb`
4. Create the cache directory and place the file there:

```powershell
mkdir "$env:USERPROFILE\.chanvar\cache\structures" -Force
copy AF-O00555-F1-model_v4.pdb "$env:USERPROFILE\.chanvar\cache\structures\"
```

### FoldX crashes or produces no output

Most common cause: `rotabase.txt` is missing from the same folder as `foldx.exe`. Check:
```powershell
dir C:\tools\foldx\
```
Both files must be present. Re-download from foldxsuite.crg.eu if `rotabase.txt` is absent.

Second common cause: the PDB file has structural issues that RepairPDB could not fix.
Check the FoldX log output in `results\full\` for error messages.

### matplotlib figures fail with a display error

Add this before any figure generation:
```python
import matplotlib
matplotlib.use('Agg')
```
The `Agg` backend writes PNGs directly without needing a display window, which is necessary
in some PowerShell environments.

### Git push fails with authentication error

You are using your GitHub account password instead of a Personal Access Token.
Create a token at github.com → Settings → Developer settings → Personal access tokens
and use that as the password. See Step 16 for details.

### `Set-ExecutionPolicy` error

Run PowerShell as Administrator (right-click PowerShell → "Run as administrator") and:
```powershell
Set-ExecutionPolicy RemoteSigned
```
Close and reopen normal PowerShell, then activate the venv.

---

## 18. What each file in the project does

```
ChanVar/
│
├── chanvar/                         Main Python package
│   ├── __init__.py                  Package entry point; exports compute_cps(), build_feature_vector()
│   ├── cli.py                       Command-line interface: chanvar annotate / batch / train / validate
│   │
│   ├── data/                        External data retrieval and bundled data files
│   │   ├── gnomad_client.py         Queries gnomAD v4 GraphQL API for allele frequencies -> f1
│   │   ├── clinvar_parser.py        Parses ClinVar VCF/XML for prior classifications -> f6
│   │   ├── alphafold_client.py      Downloads AlphaFold2 structure; maps residue positions to domains
│   │   ├── grantham_matrix.json     Full 20x20 Grantham distance matrix. Do not edit.
│   │   └── cacna1a_conservation.tsv Per-position JSD conservation for ~100 key residues -> f4
│   │
│   ├── structure/
│   │   └── stability.py             FoldX/EvoEF2 wrappers for ddG -> f2; C-alpha RMSD computation -> f3
│   │
│   ├── evolution/
│   │   ├── conservation.py          Shannon entropy, JSD, rate4site-style conservation scoring -> f4
│   │   └── alignment.py             MUSCLE/MAFFT alignment wrappers; Ensembl ortholog retrieval
│   │
│   ├── scoring/
│   │   ├── features.py              Assembles all 8 features into VariantFeatures dataclass
│   │   └── cps.py                   Weighted mean CPS, TM correction, 1000-iteration bootstrap CI, classification
│   │
│   ├── report/
│   │   └── generator.py             Writes JSON, Markdown, and HTML reports
│   │
│   └── viz/
│       └── figures.py               Generates all 5 figures (matplotlib Figures 1-4; 3Dmol.js Figure 5)
│
├── tests/
│   └── test_chanvar.py              58 unit tests. Run with: python -m pytest tests/ -v
│
├── figures/                         Generated figures (excluded from Git by .gitignore)
├── results/                         Annotation outputs (excluded from Git by .gitignore)
│
├── pyproject.toml                   Package configuration, dependency list, entry points
├── README.md                        Short project overview and quick-start
└── SETUP.md                         This file
```

### The two files most important to understand:

**`chanvar/scoring/cps.py`** is where the CPS is computed. The PRIOR_WEIGHTS dictionary
is at the top. The `compute_cps()` function takes a VariantFeatures object, applies weights,
runs the bootstrap, and returns a CPSResult. If a score seems wrong, start here.

**`chanvar/scoring/features.py`** is where the eight features are assembled from raw data
(gnomAD AF, FoldX output, conservation TSV, etc.) into the VariantFeatures dataclass that
`cps.py` receives. If a feature value seems wrong (e.g., f4 = 0.5 when you expect high
conservation), start here and trace backwards to which data source fed it.

---

*The Xiu Lab · thexiulab.org · github.com/axshoe/ChanVar · angie.xiu27@gmail.com*
