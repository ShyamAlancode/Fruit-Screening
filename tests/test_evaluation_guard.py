"""
test_evaluation_guard.py
========================
CI "build-breaker" that fails if any results file / benchmark output
accidentally reports in-sample accuracy or leaky-CV numbers as headline results.

This is the permanent guard against the 82.65% (leaky) vs 56.14% (rigorous)
confusion ever entering a paper or resume claim again.

Priority 1 from the hardening plan (Sep 25 2026).
"""
import json
import re
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Numbers that are ONLY valid as leaky/exploratory results ────────────────
# If these appear in the "headline" sections of any report without the
# mandatory leakage disclaimer, it is a CI failure.
LEAKY_NUMBERS = {
    "82.65",   # random-ROI CV, not LOAO
    "82.6",
    "97.5",    # the old phase-roadmap overclaim
    "95.6",    # intra-varietal Brix binary (never cross-validated externally)
    "83.21",   # Brix binary screening % — not peer-reviewed cross-apple
}

# Required disclaimers that MUST accompany any leaky number
LEAKAGE_DISCLAIMER_PATTERNS = [
    r"leakage[-\s]?affected",
    r"exploratory",
    r"do not (use|cite|report)",
    r"not.*loao",
    r"in[-\s]?sample",
]

# ── Numbers that MUST appear in benchmark reports (primary results) ──────────
PRIMARY_RESULTS = {
    "56.14",   # LOAO per-ROI Macro-F1
    "75.0",    # LOAO scan-level accuracy
    "35",      # held-out test lower bound
}

MANDATORY_METRICS = [
    "sensitivity",
    "specificity",
    "auroc",
    "balanced acc",  # matches 'Balanced accuracy: 65.0%' or table header 'balanced acc'
]

REPORT_FILES = list((PROJECT_ROOT / "reports").glob("*.md")) + \
               list((PROJECT_ROOT / "reports").glob("*.txt")) + \
               list((PROJECT_ROOT / "reports").glob("*.json")) + \
               [PROJECT_ROOT / "FINAL_BASELINE.md",
                PROJECT_ROOT / "README.md"]


class TestEvaluationGuard:
    """Prevents leaky CV numbers from being presented as headline results."""

    def test_leaky_82pct_has_disclaimer_in_baseline(self):
        """
        82.65% must always be accompanied by a leakage/exploratory disclaimer
        in FINAL_BASELINE.md — the authoritative results document.
        """
        baseline = PROJECT_ROOT / "FINAL_BASELINE.md"
        if not baseline.exists():
            pytest.skip("FINAL_BASELINE.md not found")

        text = baseline.read_text(encoding="utf-8").lower()
        assert "82.65" in text or "82.6" in text, "Baseline must document the leaky 82.65% result"

        # Find the line containing 82.65 and assert it has a disclaimer nearby
        lines = baseline.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "82.65" in line or "82.6" in line:
                # Check ±3 lines for disclaimer
                context = "\n".join(lines[max(0, i-3):i+4]).lower()
                has_disclaimer = any(
                    re.search(p, context) for p in LEAKAGE_DISCLAIMER_PATTERNS
                )
                assert has_disclaimer, (
                    f"Line {i+1}: '82.65%' found without a leakage disclaimer in context:\n{context}"
                )

    def test_readme_does_not_lead_with_leaky_numbers(self):
        """
        README.md headline section (first 60 lines) must NOT present 82.65%
        as a positive claim without the disclaimer being on the same or adjacent line.
        """
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md not found")

        lines = readme.read_text(encoding="utf-8").splitlines()[:60]
        for i, line in enumerate(lines):
            if "82.65" in line or "82.6" in line:
                context = "\n".join(lines[max(0, i-2):i+3]).lower()
                has_disclaimer = any(
                    re.search(p, context) for p in LEAKAGE_DISCLAIMER_PATTERNS
                )
                assert has_disclaimer, (
                    f"README headline (line {i+1}) shows 82.65% without disclaimer:\n{context}"
                )

    def test_baseline_contains_primary_loao_results(self):
        """
        FINAL_BASELINE.md must contain the three primary LOAO results.
        If someone removes them, CI fails.
        """
        baseline = PROJECT_ROOT / "FINAL_BASELINE.md"
        if not baseline.exists():
            pytest.skip("FINAL_BASELINE.md not found")
        text = baseline.read_text(encoding="utf-8")

        for number in PRIMARY_RESULTS:
            assert number in text, (
                f"Primary LOAO result '{number}' not found in FINAL_BASELINE.md. "
                "Do not remove primary results from the frozen baseline."
            )

    def test_baseline_contains_held_out_failure(self):
        """
        The pre-registered held-out test failure (35-42% Macro-F1) must
        be present in FINAL_BASELINE.md. It must never be silently removed.
        """
        baseline = PROJECT_ROOT / "FINAL_BASELINE.md"
        if not baseline.exists():
            pytest.skip("FINAL_BASELINE.md not found")
        text = baseline.read_text(encoding="utf-8").lower()

        has_held_out = "held-out" in text or "held out" in text or "p_ma1" in text
        assert has_held_out, (
            "FINAL_BASELINE.md must document the held-out test failure (P_MA1, 35-42%). "
            "This is a primary result — do not remove it."
        )

    def test_baseline_shows_mandatory_metrics_for_pesticide(self):
        """
        For the pesticide result, sensitivity, specificity, and AUROC
        must appear in FINAL_BASELINE.md — not just accuracy.
        """
        baseline = PROJECT_ROOT / "FINAL_BASELINE.md"
        if not baseline.exists():
            pytest.skip("FINAL_BASELINE.md not found")
        text = baseline.read_text(encoding="utf-8").lower()

        for metric in MANDATORY_METRICS:
            assert metric in text, (
                f"Mandatory metric '{metric}' not found in FINAL_BASELINE.md. "
                "All pesticide results must include sensitivity, specificity, "
                "balanced accuracy, and AUROC — not just overall accuracy."
            )

    def test_no_report_claims_perfect_detection(self):
        """
        No report file may claim 100% accuracy without documenting
        the corresponding low specificity. '100%' sensitivity must
        always appear alongside specificity or be qualified.
        """
        for report_path in REPORT_FILES:
            if not report_path.exists():
                continue
            text = report_path.read_text(encoding="utf-8", errors="replace")
            # Check for claims of 100% accuracy in context of detection
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if re.search(r"100\s*%.*(?:accuracy|detection|correct)", line, re.I):
                    context = "\n".join(lines[max(0, i-3):i+4]).lower()
                    # Must mention specificity or a qualifier
                    has_qualifier = any(kw in context for kw in [
                        "specificity", "false positive", "sensitivity",
                        "screening", "not validated", "30%", "simulation"
                    ])
                    assert has_qualifier, (
                        f"{report_path.name} line {i+1}: '100%' accuracy claim "
                        "without specificity/qualifier:\n" + "\n".join(lines[max(0,i-2):i+3])
                    )

    def test_forbidden_terms_not_in_reports(self):
        """
        FORBIDDEN_TERMS from constants.py must not appear as CLAIMS in any report.
        Exception: files that DOCUMENT the forbidden terms list (meta-documentation)
        are exempted — they quote the terms explicitly to explain what's banned.
        """
        from constants import FORBIDDEN_TERMS
        # Files that document the forbidden terms list — skip them
        META_DOC_EXEMPT = {
            "teacher_demo_report.md",  # explicitly documents the banned terms list
            "AUDIT_CORRECTIONS.md",    # audit documentation
            "README.md",               # quotes banned terms in the 'FORBIDDEN_TERMS' explanation section
        }
        for report_path in REPORT_FILES:
            if not report_path.exists():
                continue
            if report_path.name in META_DOC_EXEMPT:
                continue  # these files explain what's banned — not making claims
            text = report_path.read_text(encoding="utf-8", errors="replace").lower()
            for term in FORBIDDEN_TERMS:
                # Skip empty/short terms
                if not term or len(term) < 5:
                    continue
                if term.lower() in text:
                    pytest.fail(
                        f"FORBIDDEN_TERM '{term}' found in {report_path.name}. "
                        "These terms are banned from all CFRD outputs."
                    )


class TestBixProvenance:
    """
    Guard: Brix provenance must be in reconciliation state — never presenting
    ONLY the optimistic R²=0.43 without the authoritative R²=−0.03.
    """

    def test_brix_result_has_both_r2_values(self):
        """BrixResult must expose both generation results, not just the optimistic one."""
        from predictor import BrixResult
        br = BrixResult(estimated_brix=11.5)
        assert hasattr(br, "r2_verified"), "BrixResult must have r2_verified (Gen-1 authoritative)"
        assert hasattr(br, "r2_gen2_unreconciled"), "BrixResult must have r2_gen2_unreconciled"
        assert br.r2_verified < 0, (
            f"Authoritative Brix R2 must be negative ({br.r2_verified}). "
            "If this changed, document WHY and which dataset was used."
        )
        assert br.r2_gen2_unreconciled > 0.4, (
            "Gen-2 unreconciled R2 should be ~0.43 — if this changed, update provenance."
        )

    def test_brix_scientific_warning_mentions_both(self):
        """scientific_warning must mention BOTH R² values to prevent cherry-picking."""
        from predictor import BrixResult
        br = BrixResult(estimated_brix=11.5)
        w = br.scientific_warning
        assert "-0.03" in w or "0.03" in w, "scientific_warning must mention the negative R2"
        assert "0.43" in w or "reconciliation" in w.lower(), \
            "scientific_warning must mention the unreconciled Gen-2 R2 or reconciliation status"


class TestModelOutputFormatGuard:
    """
    Guard: model outputs must never silently return None in ways that
    could be misinterpreted as a real prediction.
    """

    def test_predictor_returns_ood_status_on_edge_input(self):
        """
        If a spectrum is all zeros (edge case / sensor failure),
        the predictor must still return a result — not crash — but
        OOD status must be flagged.
        """
        from predictor import ApplePredictor
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = ApplePredictor(default_pesticide_model="synthetic")

        zero_spec = [0.0] * 18
        result = predictor.predict_spectrum(zero_spec)
        # Must not crash
        assert result is not None
        # OOD should be flagged — all-zeros is clearly out of distribution
        # (we don't assert exact OOD label since threshold varies, but
        # we assert ood_status is not an empty string)
        assert result.pesticide.ood_status != "", \
            "OOD status must not be empty — always report detection confidence"

    def test_predictor_forbidden_terms_not_in_output(self):
        """
        Predictor output (as JSON string) must never contain forbidden terms.
        This guards against future code paths that might accidentally generate
        claims like 'pesticide detected' or 'safe to eat'.
        """
        from predictor import ApplePredictor
        from constants import FORBIDDEN_TERMS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = ApplePredictor(default_pesticide_model="synthetic")

        spec = [0.09, 0.10, 0.11, 0.14, 0.22, 0.38,
                0.44, 0.42, 0.40, 0.35, 0.22, 0.48,
                0.72, 0.76, 0.79, 0.81, 0.82, 0.83]
        result = predictor.predict_spectrum(spec)
        output_text = result.to_json().lower()

        for term in FORBIDDEN_TERMS:
            if term and len(term) > 4:
                assert term.lower() not in output_text, (
                    f"FORBIDDEN_TERM '{term}' found in predictor JSON output. "
                    "Predictor outputs must never make absolute safety claims."
                )
