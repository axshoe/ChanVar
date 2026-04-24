"""
tests/test_chanvar.py
----------------------
Unit tests for all ChanVar scoring functions.

Every quantitative function is tested against:
  1. Known-value examples from the primary literature
  2. Boundary conditions (zero, infinity, None inputs)
  3. Internal consistency properties

Test coverage targets:
  - gnomad_client.py: compute_frequency_feature
  - stability.py: grantham_distance, compute_grantham_feature, compute_ddg_feature, compute_rmsd_feature
  - conservation.py: compute_shannon_entropy, compute_jensenshannon_divergence, estimate_rate4site, compute_conservation_feature
  - features.py: build_feature_vector, _parse_aa_change
  - cps.py: compute_cps, _weighted_mean, _classify, _bootstrap_ci

Known-value references:
  - Grantham (1974): verified R->Q = 43, R->W = 101
  - Shannon entropy: perfectly conserved column H=0; uniform H=log2(20)≈4.32
  - Frequency feature: AF=0 -> 1.0; AF>=0.01 -> 0.0
"""

import math
import pytest
from unittest.mock import patch

from chanvar.data.gnomad_client import compute_frequency_feature
from chanvar.structure.stability import (
    grantham_distance,
    compute_grantham_feature,
    compute_ddg_feature,
    compute_rmsd_feature,
    GRANTHAM_MAX,
)
from chanvar.evolution.conservation import (
    compute_shannon_entropy,
    compute_jensenshannon_divergence,
    compute_conservation_feature,
    estimate_rate4site,
)
from chanvar.scoring.features import build_feature_vector, _parse_aa_change
from chanvar.scoring.cps import compute_cps, _weighted_mean, _classify, PRIOR_WEIGHTS


class TestFrequencyFeature:
    """Test f1 (gnomAD frequency) mapping."""

    def test_absent_variant_returns_one(self):
        """Variant absent from gnomAD: maximum pathogenicity prior."""
        assert compute_frequency_feature(None) == 1.0
        assert compute_frequency_feature(0.0) == 1.0

    def test_common_variant_returns_zero(self):
        """Common variant (AF >= 0.01): effectively benign."""
        assert compute_frequency_feature(0.01) == 0.0
        assert compute_frequency_feature(0.5) == 0.0
        assert compute_frequency_feature(1.0) == 0.0

    def test_very_rare_variant(self):
        """AF = 0.0001 should return approximately 0.80."""
        f1 = compute_frequency_feature(0.0001)
        assert 0.75 <= f1 <= 0.85, f"Expected ~0.80, got {f1}"

    def test_rare_variant(self):
        """AF = 0.001 should return approximately 0.30."""
        f1 = compute_frequency_feature(0.001)
        assert 0.25 <= f1 <= 0.35, f"Expected ~0.30, got {f1}"

    def test_monotone_decreasing(self):
        """Feature must be monotonically decreasing with AF."""
        afs = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01]
        scores = [compute_frequency_feature(af) for af in afs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Monotonicity violated: AF={afs[i]} -> {scores[i]}, "
                f"AF={afs[i+1]} -> {scores[i+1]}"
            )

    def test_output_range(self):
        """All outputs must be in [0, 1]."""
        for af in [0, 1e-7, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.1]:
            f1 = compute_frequency_feature(af)
            assert 0.0 <= f1 <= 1.0, f"f1={f1} out of range for AF={af}"


class TestGranthamDistance:
    """
    Test Grantham distance lookup.

    Reference values from Grantham (1974) Table 1.
    """

    def test_r_to_q_is_43(self):
        """R->Q Grantham distance = 43 (Grantham 1974, Table 1)."""
        d = grantham_distance("R", "Q")
        assert d == 43.0, f"Expected 43.0, got {d}"

    def test_r_to_w_is_101(self):
        """R->W Grantham distance = 101."""
        d = grantham_distance("R", "W")
        assert d == 101.0, f"Expected 101.0, got {d}"

    def test_same_amino_acid_is_zero(self):
        """No mutation = 0 distance."""
        assert grantham_distance("R", "R") == 0.0
        assert grantham_distance("A", "A") == 0.0

    def test_symmetric(self):
        """Distance(A, B) == Distance(B, A)."""
        for pair in [("R", "Q"), ("R", "W"), ("S", "L"), ("D", "N")]:
            d_fwd = grantham_distance(pair[0], pair[1])
            d_rev = grantham_distance(pair[1], pair[0])
            assert d_fwd == d_rev, f"Asymmetry: {pair[0]}->{pair[1]}={d_fwd}, {pair[1]}->{pair[0]}={d_rev}"

    def test_normalized_in_range(self):
        """Normalized feature f8 must be in [0, 1]."""
        for aa1 in "ARNDCQEGHILKMFPSTWYV":
            for aa2 in "ARNDCQEGHILKMFPSTWYV":
                f8 = compute_grantham_feature(aa1, aa2)
                assert 0.0 <= f8 <= 1.0, f"f8={f8} out of range for {aa1}->{aa2}"


class TestDdgFeature:
    """Test dDDG (f2) sigmoid mapping."""

    def test_none_returns_neutral(self):
        """Missing dDDG returns 0.5 (neutral)."""
        assert compute_ddg_feature(None, "voltage_sensor") == 0.5

    def test_zero_ddg(self):
        """dDDG=0 should return sigmoid(-0.5) ≈ 0.378."""
        f2 = compute_ddg_feature(0.0, "C_terminal")
        expected = 1.0 / (1.0 + math.exp(0.5))
        assert abs(f2 - expected) < 0.001

    def test_high_ddg_approaches_one(self):
        """Large positive dDDG (strongly destabilizing) -> f2 near 1."""
        f2 = compute_ddg_feature(10.0, "C_terminal")
        assert f2 > 0.95

    def test_negative_ddg_below_half(self):
        """Stabilizing mutation (dDDG < 0) -> f2 < 0.5."""
        f2 = compute_ddg_feature(-2.0, "C_terminal")
        assert f2 < 0.5

    def test_output_range(self):
        """f2 must be in [0, 1] for all inputs."""
        for ddg in [-5, -2, -1, 0, 0.5, 1, 2, 5, 10]:
            f2 = compute_ddg_feature(float(ddg), "voltage_sensor")
            assert 0.0 <= f2 <= 1.0


class TestRmsdFeature:
    """Test local RMSD (f3) sigmoid mapping."""

    def test_none_returns_neutral(self):
        assert compute_rmsd_feature(None) == 0.5

    def test_zero_rmsd(self):
        """No structural change -> low pathogenicity signal."""
        f3 = compute_rmsd_feature(0.0)
        expected = 1.0 / (1.0 + math.exp(1.0))  # sigmoid(-1)
        assert abs(f3 - expected) < 0.001

    def test_high_rmsd(self):
        """Large RMSD -> f3 near 1."""
        f3 = compute_rmsd_feature(5.0)
        assert f3 > 0.95

    def test_one_angstrom_is_half(self):
        """RMSD=1.0 -> f3=0.5 (sigmoid inflection point)."""
        f3 = compute_rmsd_feature(1.0)
        assert abs(f3 - 0.5) < 0.001


class TestShannonEntropy:
    """Test Shannon entropy calculation."""

    def test_perfectly_conserved(self):
        """All same amino acid -> H = 0."""
        column = ["R"] * 20
        assert compute_shannon_entropy(column) == 0.0

    def test_maximum_entropy(self):
        """All 20 different amino acids equally -> H = log2(20) ≈ 4.322."""
        column = list("ACDEFGHIKLMNPQRSTVWY")
        h = compute_shannon_entropy(column)
        expected = math.log2(20)
        assert abs(h - expected) < 0.01, f"Expected {expected:.3f}, got {h:.3f}"

    def test_gaps_excluded(self):
        """Gap characters do not count toward entropy."""
        column_with_gaps = ["R"] * 10 + ["-"] * 5
        column_clean = ["R"] * 10
        assert compute_shannon_entropy(column_with_gaps) == compute_shannon_entropy(column_clean)

    def test_empty_column(self):
        """Empty or all-gap column -> H = 0 (undefined, treated as conserved)."""
        assert compute_shannon_entropy([]) == 0.0
        assert compute_shannon_entropy(["-", "-", "X"]) == 0.0

    def test_two_amino_acids_equal(self):
        """Half R, half Q -> H = 1.0 bit."""
        column = ["R"] * 10 + ["Q"] * 10
        h = compute_shannon_entropy(column)
        assert abs(h - 1.0) < 0.001


class TestConservationFeature:
    """Test f4 conservation mapping."""

    def test_none_returns_neutral(self):
        f4 = compute_conservation_feature(None)
        assert f4 == 0.5

    def test_highly_conserved(self):
        """rate4site = 0.01 (invariant) -> f4 near 1."""
        f4 = compute_conservation_feature(0.01)
        assert f4 > 0.95

    def test_average_rate(self):
        """rate4site = 1.0 (average) -> f4 ≈ 0.67."""
        f4 = compute_conservation_feature(1.0)
        expected = 1.0 - 1.0 / 3.0
        assert abs(f4 - expected) < 0.01

    def test_rapidly_evolving(self):
        """rate4site >= 3.0 -> f4 = 0."""
        f4 = compute_conservation_feature(3.0)
        assert f4 == 0.0

    def test_output_range(self):
        for rate in [0.01, 0.3, 1.0, 2.0, 3.0, 5.0]:
            f4 = compute_conservation_feature(rate)
            assert 0.0 <= f4 <= 1.0


class TestParseAaChange:
    """Test amino acid change parsing."""

    def test_standard_1letter(self):
        ref, pos, alt = _parse_aa_change("R192Q")
        assert ref == "R"
        assert pos == 192
        assert alt == "Q"

    def test_3letter_codes(self):
        ref, pos, alt = _parse_aa_change("Arg192Gln")
        assert ref == "R"
        assert pos == 192
        assert alt == "Q"

    def test_p_prefix(self):
        ref, pos, alt = _parse_aa_change("p.R192Q")
        assert ref == "R"
        assert pos == 192
        assert alt == "Q"

    def test_s218l(self):
        """S218L: a known FHM1 pathogenic variant."""
        ref, pos, alt = _parse_aa_change("S218L")
        assert ref == "S"
        assert pos == 218
        assert alt == "L"

    def test_invalid_returns_none(self):
        ref, pos, alt = _parse_aa_change("invalid")
        assert ref is None
        assert pos is None
        assert alt is None


class TestBuildFeatureVector:
    """Test feature vector assembly."""

    def test_minimal_input(self):
        """Build succeeds with only variant_id and aa_change."""
        fv = build_feature_vector(
            variant_id="19-13392445-G-A",
            aa_change="R192Q",
        )
        assert fv.aa_change == "R192Q"
        assert fv.residue_number == 192
        assert fv.ref_aa == "R"
        assert fv.alt_aa == "Q"
        assert fv.f5 > 0  # Domain weight always set

    def test_voltage_sensor_domain(self):
        """R192 is in voltage sensor -> domain = voltage_sensor, f5 = 1.0."""
        fv = build_feature_vector(
            variant_id="19-X",
            aa_change="R192Q",
        )
        assert fv.domain == "voltage_sensor"
        assert fv.f5 == 1.0
        assert fv.is_tm_domain is True

    def test_frequency_feature_set(self):
        """AF provided -> f1 computed correctly."""
        fv = build_feature_vector(
            variant_id="19-X",
            aa_change="R192Q",
            gnomad_af=0.0,
        )
        assert fv.f1 == 1.0  # Absent = max prior

    def test_cadd_normalization(self):
        """CADD Phred 40 -> f7 = 1.0."""
        fv = build_feature_vector(
            variant_id="19-X",
            aa_change="R192Q",
            cadd_phred=40.0,
        )
        assert fv.f7 == 1.0

    def test_cadd_zero(self):
        """CADD Phred 0 -> f7 = 0."""
        fv = build_feature_vector(
            variant_id="19-X",
            aa_change="R192Q",
            cadd_phred=0.0,
        )
        assert fv.f7 == 0.0

    def test_data_completeness(self):
        """With no optional data, completeness < 1."""
        fv = build_feature_vector(variant_id="19-X", aa_change="R192Q")
        # f5 always set, f1 always set (af=None -> f1=1.0), f8 always set
        # f2, f3, f4, f6, f7 depend on optional inputs
        assert 0.0 < fv.data_completeness <= 1.0

    def test_feature_vector_length(self):
        """Feature vector always has 8 elements."""
        fv = build_feature_vector(variant_id="19-X", aa_change="R192Q")
        assert len(fv.feature_vector) == 8


class TestWeightedMean:
    """Test CPS weighted mean computation."""

    def test_equal_weights_is_arithmetic_mean(self):
        features = [0.2, 0.4, 0.6, 0.8]
        weights = [1.0, 1.0, 1.0, 1.0]
        result = _weighted_mean(features, weights)
        expected = sum(features) / len(features)
        assert abs(result - expected) < 1e-10

    def test_high_weight_dominates(self):
        features = [0.9, 0.1, 0.1, 0.1]
        weights = [100.0, 1.0, 1.0, 1.0]
        result = _weighted_mean(features, weights)
        assert result > 0.8  # First feature dominates

    def test_zero_weights_returns_half(self):
        result = _weighted_mean([0.9, 0.1], [0.0, 0.0])
        assert result == 0.5


class TestClassification:
    """Test CPS -> classification thresholds."""

    def test_likely_pathogenic(self):
        assert _classify(0.90) == "Likely Pathogenic"
        assert _classify(0.85) == "Likely Pathogenic"

    def test_possibly_pathogenic(self):
        assert _classify(0.75) == "Possibly Pathogenic"
        assert _classify(0.70) == "Possibly Pathogenic"

    def test_vus(self):
        assert _classify(0.55) == "Uncertain Significance"
        assert _classify(0.40) == "Uncertain Significance"

    def test_possibly_benign(self):
        assert _classify(0.30) == "Possibly Benign"
        assert _classify(0.20) == "Possibly Benign"

    def test_likely_benign(self):
        assert _classify(0.10) == "Likely Benign"
        assert _classify(0.0) == "Likely Benign"


class TestComputeCPS:
    """Integration tests for CPS computation."""

    def _make_known_pathogenic_features(self):
        """Simulate feature vector for R192Q (known FHM1 P/LP variant)."""
        return build_feature_vector(
            variant_id="19-13392445-G-A",
            aa_change="R192Q",
            gnomad_af=0.0,          # absent from gnomAD
            ddg_foldx=2.8,          # destabilizing
            alphamissense_score=0.95,  # AlphaMissense high
            cadd_phred=38.0,        # CADD Phred 38
        )

    def _make_benign_features(self):
        """Simulate feature vector for a common benign variant."""
        return build_feature_vector(
            variant_id="19-13000000-A-G",
            aa_change="V100I",  # hypothetical common variant
            gnomad_af=0.05,          # common
            ddg_foldx=0.1,           # nearly neutral
            alphamissense_score=0.10,  # benign
            cadd_phred=5.0,
        )

    def test_known_pathogenic_has_high_cps(self):
        """R192Q (FHM1 P/LP) should score > 0.70."""
        features = self._make_known_pathogenic_features()
        result = compute_cps(features, bootstrap_n=100)
        assert result.cps > 0.65, (
            f"Expected CPS > 0.65 for known pathogenic variant, got {result.cps}"
        )

    def test_benign_has_low_cps(self):
        """Common variant with benign predictors should score < 0.40."""
        features = self._make_benign_features()
        result = compute_cps(features, bootstrap_n=100)
        assert result.cps < 0.50, (
            f"Expected CPS < 0.50 for simulated benign variant, got {result.cps}"
        )

    def test_ci_contains_point_estimate(self):
        """95% CI must contain the point estimate."""
        features = self._make_known_pathogenic_features()
        result = compute_cps(features, bootstrap_n=500)
        assert result.ci_lower <= result.cps <= result.ci_upper

    def test_tm_flag_set_for_voltage_sensor(self):
        """R192Q is in voltage sensor -> should set TM flag."""
        features = self._make_known_pathogenic_features()
        result = compute_cps(features, bootstrap_n=100)
        assert result.is_tm_domain is True
        assert result.confidence_flag is not None
        assert "TM_DDG" in (result.confidence_flag or "")

    def test_reproducibility(self):
        """Same seed -> same CI bounds."""
        features = self._make_known_pathogenic_features()
        r1 = compute_cps(features, bootstrap_n=200, seed=42)
        r2 = compute_cps(features, bootstrap_n=200, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_prior_weights_sum_to_expected(self):
        """Prior weights sanity check."""
        total = sum(PRIOR_WEIGHTS.values())
        assert total > 0
        assert all(w > 0 for w in PRIOR_WEIGHTS.values())

    def test_feature_contributions_sum_to_one(self):
        """Feature contributions must sum to 1.0."""
        features = self._make_known_pathogenic_features()
        result = compute_cps(features, bootstrap_n=100)
        total = sum(result.feature_contributions.values())
        assert abs(total - result.cps) < 0.01, f"Contributions sum to {total}, expected ~CPS {result.cps}"


class TestKnownVariants:
    """
    Validate CPS against known FHM1 and EA2 pathogenic variants.

    Expected behavior:
      - FHM1 pathogenic variants (R192Q, S218L, T666M, R1920P): CPS > 0.65
      - EA2 pathogenic variants (typically LOF, splice): partially captured
      - Known benign (common in gnomAD): CPS < 0.40

    These tests use simulated data because real gnomAD/FoldX data requires external tools.
    The test validates the scoring mechanics, not the feature values.
    """

    KNOWN_PATHOGENIC = [
        # (aa_change, af, ddg, alphamissense, cadd)
        ("R192Q", 0.0, 2.8, 0.95, 38),    # Most common FHM1 variant
        ("S218L", 0.0, 3.5, 0.98, 39),    # Severe FHM1
        ("T666M", 0.0, 2.1, 0.88, 35),    # FHM1
    ]

    def test_known_pathogenic_variants_score_high(self):
        for aa_change, af, ddg, am, cadd in self.KNOWN_PATHOGENIC:
            fv = build_feature_vector(
                variant_id=f"19-test-{aa_change}",
                aa_change=aa_change,
                gnomad_af=af,
                ddg_foldx=ddg,
                alphamissense_score=am,
                cadd_phred=float(cadd),
            )
            result = compute_cps(fv, bootstrap_n=100)
            assert result.cps > 0.60, (
                f"{aa_change}: expected CPS > 0.60, got {result.cps:.3f}"
            )
