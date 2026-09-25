"""
test_model_compatibility.py
===========================
Verifies that all shipped model artifacts:
  1. Load without error under the current scikit-learn version.
  2. Match the expected scikit-learn training version (1.5.0).
  3. Accept an 18-channel input and return a valid output.
  4. Produce an InconsistentVersionWarning (not a crash) when loaded under
     a different sklearn version — this test documents the boundary.

Audit Reference: Second Technical Review §9.2 — "11 tests errored due to
sklearn 1.5.0 → 1.8.0 version mismatch". This test suite makes the
version requirement explicit and testable.

Model Format Note:
  All CFRD models are stored as dicts:
    {'pipeline': sklearn.Pipeline, 'classes': List[str], 'metrics': dict}
  Use model['pipeline'].predict() not model.predict().
"""

import warnings
import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # Required: models depend on local ood module

MODELS_DIR = PROJECT_ROOT / "models"

# Models that must be loadable and functional
REQUIRED_MODELS = {
    "treatment_3class_synthetic": MODELS_DIR / "treatment_condition" / "treatment_3class_pca_lda_synthetic_20260923_082626.joblib",
    "treatment_3class_real":      MODELS_DIR / "treatment_condition" / "treatment_3class_pca_lda_real_20260915_123548.joblib",
    "ripeness_classifier":        MODELS_DIR / "quality" / "ripeness_classifier_20260923_082655.joblib",
    "variety_classifier":         MODELS_DIR / "quality" / "variety_classifier_20260923_082655.joblib",
}

BRIX_MODELS = {
    "brix_as7265x_18ch_best":    MODELS_DIR / "quality" / "brix_as7265x_18channel_best.joblib",
    "brix_elasticnet_twin":      MODELS_DIR / "brix" / "as7265x_18ch_twin_best_elasticnet.joblib",
    "brix_ridge_18ch":           MODELS_DIR / "brix" / "ridge_18ch_clean.joblib",
}

MOCK_SPECTRUM_18CH = np.array([
    0.09, 0.10, 0.11, 0.14, 0.22, 0.38,
    0.44, 0.42, 0.40, 0.35, 0.22, 0.48,
    0.72, 0.76, 0.79, 0.81, 0.82, 0.83,
], dtype=np.float32).reshape(1, -1)

# Ripeness and variety models use 36 colour/HSV features
# (NOT 18-channel spectra — they are appearance-based models, not AS7265x spectral models)
# This is documented in AUDIT_CORRECTIONS.md and the model manifest.
MOCK_COLOUR_36 = np.random.RandomState(42).rand(1, 36).astype(np.float32)

EXPECTED_SKLEARN_VERSION = "1.5.0"


class TestSklearnVersionCompatibility:
    """Documents and tests sklearn version requirements."""

    def test_current_sklearn_version_is_pinned(self):
        """Verify sklearn is the pinned version matching model serialization."""
        import sklearn
        current = sklearn.__version__
        assert current == EXPECTED_SKLEARN_VERSION, (
            f"scikit-learn version mismatch: models were serialized under {EXPECTED_SKLEARN_VERSION}, "
            f"but current version is {current}. "
            f"Run: pip install scikit-learn=={EXPECTED_SKLEARN_VERSION} "
            f"or use: pip install -r requirements_lock.txt"
        )

    def test_sklearn_version_pinned_in_requirements(self):
        """requirements.txt must pin scikit-learn to the exact version."""
        req_path = PROJECT_ROOT / "requirements.txt"
        content = req_path.read_text()
        assert f"scikit-learn=={EXPECTED_SKLEARN_VERSION}" in content, (
            f"requirements.txt does not pin scikit-learn=={EXPECTED_SKLEARN_VERSION}. "
            "Floating version allows breaking changes on model loading."
        )

    def test_lockfile_pins_sklearn(self):
        """requirements_lock.txt must pin scikit-learn."""
        lockfile = PROJECT_ROOT / "requirements_lock.txt"
        if lockfile.exists():
            content = lockfile.read_text()
            assert f"scikit-learn=={EXPECTED_SKLEARN_VERSION}" in content


class TestRequiredModelLoading:
    """All required models must load without errors under current environment."""

    @pytest.mark.parametrize("name,path", list(REQUIRED_MODELS.items()))
    def test_model_file_exists(self, name, path):
        assert path.exists(), f"Required model file missing: {path}"

    def test_model_dict_structure(self):
        """Document the model dict format: all CFRD models are {'pipeline', 'classes', 'metrics'}"""
        import joblib
        path = REQUIRED_MODELS["ripeness_classifier"]
        if not path.exists():
            pytest.skip("Ripeness model not found")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
        assert isinstance(model, dict), "Model should be stored as a dict"
        assert "pipeline" in model, "Model dict must have 'pipeline' key"
        assert hasattr(model["pipeline"], "predict"), "pipeline must have predict method"

    @pytest.mark.parametrize("name,path", list(REQUIRED_MODELS.items()))
    def test_model_loads_without_error(self, name, path):
        """Models must load with at most InconsistentVersionWarning (no crashes)."""
        if not path.exists():
            pytest.skip(f"Model file not present: {path.name}")
        import joblib
        # Allow InconsistentVersionWarning but NOT other errors
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        assert model is not None, f"Model loaded as None: {name}"
        # Log any version warnings for visibility
        version_warnings = [w for w in caught if "InconsistentVersion" in str(w.category)]
        if version_warnings:
            pytest.fail(
                f"Model {name} loaded with InconsistentVersionWarning: "
                f"saved under a different sklearn version. "
                f"Reinstall: pip install -r requirements_lock.txt"
            )

    @pytest.mark.parametrize("name,path", [
        ("treatment_3class_synthetic", REQUIRED_MODELS["treatment_3class_synthetic"]),
        ("treatment_3class_real",      REQUIRED_MODELS["treatment_3class_real"]),
    ])
    def test_pesticide_model_predicts(self, name, path):
        """Pesticide models must accept 18-ch input and return valid class index."""
        if not path.exists():
            pytest.skip(f"Model file not present: {path.name}")
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(path)
        # Models stored as {'pipeline': Pipeline, 'classes': [...], 'metrics': {...}}
        pipeline = obj["pipeline"] if isinstance(obj, dict) else obj
        pred = pipeline.predict(MOCK_SPECTRUM_18CH)
        assert len(pred) == 1
        assert pred[0] in (0, 1, 2), f"Invalid class index {pred[0]} from model {name}"

    def test_ripeness_model_predicts(self):
        """
        Ripeness model uses 36 COLOUR/HSV features, NOT 18-channel AS7265x spectra.
        This is the audit-documented distinction between appearance-based and spectral results.
        GroupKFold accuracy: 64.07% (vs 33% chance). NOT the '>90%' image model claim.
        """
        path = REQUIRED_MODELS["ripeness_classifier"]
        if not path.exists():
            pytest.skip("Ripeness model not found")
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(path)
        assert obj["pipeline"][0].n_features_in_ == 36, (
            "Ripeness model expects 36 colour features, not 18 spectral channels. "
            "This is a colour-based model, not an AS7265x spectral model."
        )
        pipeline = obj["pipeline"] if isinstance(obj, dict) else obj
        pred = pipeline.predict(MOCK_COLOUR_36)
        assert len(pred) == 1

    def test_variety_model_predicts(self):
        """
        Variety model uses 36 COLOUR/HSV features, NOT 18-channel AS7265x spectra.
        GroupKFold accuracy: 47.54% (vs 20% chance). Colour-based, not spectral.
        """
        path = REQUIRED_MODELS["variety_classifier"]
        if not path.exists():
            pytest.skip("Variety model not found")
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(path)
        assert obj["pipeline"][0].n_features_in_ == 36, (
            "Variety model expects 36 colour features, not 18 spectral channels."
        )
        pipeline = obj["pipeline"] if isinstance(obj, dict) else obj
        pred = pipeline.predict(MOCK_COLOUR_36)
        assert len(pred) == 1


class TestBrixModelStatus:
    """
    Brix models are under provenance reconciliation.
    These tests document the current state without asserting correctness.
    """

    @pytest.mark.parametrize("name,path", list(BRIX_MODELS.items()))
    def test_brix_model_file_exists(self, name, path):
        if not path.exists():
            pytest.skip(f"Brix model file not present: {path.name}")
        assert path.exists()

    @pytest.mark.parametrize("name,path", list(BRIX_MODELS.items()))
    def test_brix_model_loads(self, name, path):
        """Document whether each brix model can be loaded in current environment."""
        if not path.exists():
            pytest.skip(f"Brix model file not present: {path.name}")
        import joblib
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        version_warnings = [w for w in caught if "InconsistentVersion" in str(w.category)]
        if version_warnings:
            pytest.fail(
                f"Brix model {name} has sklearn version mismatch. "
                "Brix benchmark provenance is already under reconciliation (see REPRODUCIBILITY.md). "
                "Re-train from pinned environment: pip install -r requirements_lock.txt"
            )

    def test_brix_provenance_warning_in_predictor(self):
        """BrixResult must carry provenance reconciliation warning."""
        from predictor import BrixResult
        br = BrixResult(estimated_brix=11.5)
        assert "RECONCILIATION" in br.scientific_warning or "reconciliation" in br.scientific_warning.lower()
        assert br.r2_verified < 0, "Verified Brix R² must be negative (per Sept 15 registry)"


class TestCrossPlatformPaths:
    """
    Tests for the cross-platform model path bug (Review 3, Sep 25 2026).

    Root cause: Path.__str__ returns OS-native separators on Windows (backslashes),
    which break path resolution on Linux/Mac when stored in model_manifest.json.
    Fix: always use Path.as_posix() when writing to manifest.
    See AUDIT_CORRECTIONS.md Issue 8.
    """

    def test_manifest_paths_are_posix(self):
        """
        All filepath/dataset_path values in model_manifest.json must use
        forward slashes. Backslashes break path resolution on Linux/Mac.
        This test catches regressions on Windows where str(Path) uses backslashes.
        """
        import json
        manifest_path = PROJECT_ROOT / "models" / "model_manifest.json"
        assert manifest_path.exists(), "model_manifest.json must exist"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        violations = []
        for key, entry in manifest.get("models", {}).items():
            for field in ("filepath", "dataset_path"):
                value = entry.get(field, "")
                # Check for literal backslash (chr 92)
                if "\\" in value:
                    violations.append(f"  {key}.{field}: {value!r}")
        assert not violations, (
            "model_manifest.json contains Windows backslashes in path fields.\n"
            "These break path resolution on Linux/Mac.\n"
            "Fix: use Path.as_posix() in model_registry.register_model().\n"
            "Violations:\n" + "\n".join(violations)
        )

    def test_get_model_path_returns_path_for_known_keys(self):
        """
        get_model_path must return a valid Path (not None) for all manifest keys.
        Previously returned None on Linux due to backslash paths, causing
        TypeError: 'NoneType' object is not subscriptable downstream.
        """
        import json
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from model_registry import get_model_path
        manifest_path = PROJECT_ROOT / "models" / "model_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        missing = []
        for key in manifest.get("models", {}).keys():
            p = get_model_path(key)
            if p is None or not p.exists():
                missing.append(f"  {key}: resolved to {p}")
        assert not missing, (
            "get_model_path returned None or non-existent path for manifest keys.\n"
            "This causes TypeError: 'NoneType' object is not subscriptable.\n"
            "Keys with resolution failures:\n" + "\n".join(missing)
        )

    def test_register_model_writes_posix_paths(self, tmp_path):
        """register_model must write forward-slash paths even when run on Windows."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import model_registry as mr
        import json

        fake_model = tmp_path / "fake_model.joblib"
        fake_model.write_bytes(b"fake")
        fake_dataset = tmp_path / "fake_data.csv"
        fake_dataset.write_bytes(b"fake")

        # Redirect manifest to temp location to avoid polluting real manifest
        original_manifest = mr.MANIFEST_PATH
        test_manifest = tmp_path / "test_manifest.json"
        test_manifest.write_text('{"models": {}, "experiments": []}')
        mr.MANIFEST_PATH = test_manifest
        try:
            entry = mr.register_model(
                model_key="test_cross_platform",
                filepath=fake_model,
                dataset_path=fake_dataset,
                metrics={"acc": 0.5},
                model_type="test",
            )
        finally:
            mr.MANIFEST_PATH = original_manifest

        backslash = "\\"
        assert backslash not in entry["filepath"], (
            f"register_model wrote backslash path: {entry['filepath']!r}. "
            "Use Path.as_posix() to prevent this."
        )
        assert backslash not in entry["dataset_path"], (
            f"register_model wrote backslash dataset_path: {entry['dataset_path']!r}"
        )
