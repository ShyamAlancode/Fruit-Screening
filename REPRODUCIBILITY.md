# CFRD — Reproducibility Audit

> **Date:** September 24, 2026  
> **Based on:** Second Technical Review, §9 (Reproducibility Audit)  
> **Status:** Authoritative — this is the single source of truth for what can and cannot be reproduced from this repository.

---

## 1. Archive Integrity

| Item | Status | Notes |
|---|---|---|
| ZIP archive integrity | ✅ High | Both supplied ZIP files pass integrity checks |
| Git HEAD commit | `e9f3398` | Clean committed HEAD |
| Working tree status | ⚠️ Modified | Working tree contains modified/untracked files beyond HEAD. See §2. |

> [!IMPORTANT]
> The current working tree has not been tagged as a release. "Final project state" is ambiguous until a release commit with a SHA256 manifest is declared.

---

## 2. Version and Environment

| Item | Status | Details |
|---|---|---|
| Python version | ✅ Pinned | Python 3.11.8 |
| scikit-learn | ✅ PINNED | `scikit-learn==1.5.0` — model artifacts serialized under this version |
| Lock file | ✅ Present | `requirements_lock.txt` — use this for exact reproduction |
| `requirements.txt` | ✅ Fixed (Sep 24) | Now pins `scikit-learn==1.5.0` exactly |
| sklearn 1.8.0 compatibility | ❌ Fails | Reviewer environment (1.8.0) caused 11 test failures. Fixed by pinning. |

### How to Reproduce

```bash
# Exact environment (guaranteed):
pip install -r requirements_lock.txt

# Minimum environment (may work):
pip install -r requirements.txt
```

> [!WARNING]
> Loading models saved under `scikit-learn==1.5.0` into `>=1.6` produces
> `InconsistentVersionWarning` and may fail on internal API changes.
> Always use `requirements_lock.txt` for model inference.

---

## 3. Unit Test Suite

| Test file | Tests | Status | Notes |
|---|---|---|---|
| `test_as7265x_driver.py` | 30 | ✅ Pass | Hardware driver parameter validation, mock mode |
| `test_calibration.py` | 11 | ✅ Pass | Dark/white reference math |
| `test_channel_schema.py` | 10 | ✅ Pass | 18-channel schema constraints |
| `test_no_group_leakage.py` | 5 | ✅ Pass | GroupKFold zero-leakage assertions |
| `test_predictor.py` | 6 | ✅ Pass | ML inference engine |
| `test_sensor_simulator.py` | 6 | ✅ Pass | Optical noise model |
| `test_case_studies.py` | 5 | ✅ Pass | End-to-end prediction |
| **`test_model_compatibility.py`** | **NEW** | ✅ Pass on sklearn 1.5.0 | Explicitly tests version pinning |
| **Total** | **73+** | **✅ All pass (sklearn 1.5.0)** | |

**Reviewer's finding:** "62 pass, 11 fail" — confirmed as sklearn 1.8.0 environment issue, not a code defect. Fixed by pinning `requirements.txt`.

---

## 4. Pesticide Module Reproducibility

| Component | Reproducible from repo? | Assessment |
|---|---|---|
| LOAO benchmark code | ✅ High | `scripts/rigorous_leakfree_benchmark.py` present |
| LOAO benchmark outputs | ✅ High | CSVs and reports present for inspection |
| Feature construction (17 adj-diff) | ✅ High | Code + constants present |
| Aggregation logic (multi-spot) | ✅ High | Code present |
| AS7265x simulation definition | ✅ High | `src/hsi_to_18ch.py` + `src/sensor_simulator.py` |
| **Raw 28 HSI cubes** | ❌ Low | Multi-GB raw source NOT included in review ZIP |
| **700-row ROI extraction matrix** | ❌ Low | Not packaged — requires raw cubes + extraction script |
| Complete numerical pipeline | ❌ Cannot regenerate from ZIP alone | Raw data external |

> [!NOTE]
> A third party can independently verify: code structure, fold logic, cohort IDs, benchmark outputs,
> feature construction, aggregation logic, and provenance manifests.
> A third party **cannot** regenerate: extraction from the original 28 cubes, the raw-to-ROI
> numerical pipeline, or the 700-spectrum matrix.

---

## 5. Brix Benchmark Reproducibility

| Item | Status | Notes |
|---|---|---|
| Generation 1 (Sept 15, 60 scans, 33 apples) | ✅ Verified | Recorded in experiment registry; negative R² |
| Generation 2 (Sept 24, 417 obs, 109 fruits) | ⚠️ External | Raw dataset at external local paths not packaged |
| Processed 417-row master dataset | ❌ Not in repo | SHA256 not declared in registry |
| Model selection strategy | ⚠️ Optimistic | Best-of-12 selected on same CV data |
| **Status** | **Under reconciliation** | Do not publish either as sole final result |

**Required to fix:**
1. Package the processed 417-row dataset with SHA256 hash
2. Declare one pipeline as authoritative
3. Apply nested CV or an untouched external test set for the model selection step

---

## 6. Physical Hardware Reproducibility

| Component | Status | Notes |
|---|---|---|
| Physical AS7265x scans | ❌ 0 scans | Sensor not yet received |
| Physical dark reference | ❌ Not measured | |
| Physical PTFE white reference | ❌ Not measured | |
| Physical LED switching | ❌ Not implemented | `as7265x_driver.py` I2C branch returns zeros |
| `data/hardware/` CSVs | ❌ Simulation only | Prefixed `SIMULATOR_SMOKE_TEST__`; see quarantine README |
| Physical illumination verification | ❌ Not demonstrated | Current data consistent with simulation, not hardware |

---

## 7. Code-Version Consistency

| Issue | Status |
|---|---|
| Git HEAD | `e9f3398` — clean |
| Working tree | Many modified/untracked files |
| Release tag | **Not yet created** |
| SHA256 model manifest | ✅ Present (`models/model_manifest.json`) |

**Required action:** Tag a clean release commit after stabilizing the working tree:
```bash
git add -A
git commit -m "release: v0.1.0-pre-hardware-computational-baseline"
git tag v0.1.0-pre-hardware
```

---

## 8. Reproducibility Scorecard (from Second Technical Review §9.7)

| Component | Reproducibility | Assessment |
|---|---|---|
| Archive integrity | ✅ High | ZIP integrity verified |
| Unit tests (sklearn 1.5.0) | ✅ High | 73 pass, 0 fail |
| Unit tests (sklearn 1.8.0) | ❌ Fails | Fixed by pinning — use lockfile |
| Pesticide code | ✅ High | Main scripts present |
| Pesticide raw data | ❌ Low | Multi-GB raw source omitted |
| Pesticide benchmark outputs | ✅ High for inspection | CSVs / reports present |
| AS7265x simulation definition | ✅ High | Code + constants present |
| Brix Gen 1 benchmark | ✅ Medium | Script present; raw data external |
| Brix Gen 2 benchmark | ❌ Low | External raw archives; not packaged |
| Physical sensor readout | ❌ Low | Placeholder implementation |
| Physical illumination validation | ❌ Low | Current data is simulation |
| Version pinning | ✅ High | `requirements.txt` + `requirements_lock.txt` now both pin sklearn 1.5.0 |
| Release reproducibility | ⚠️ Medium | Working tree modified; no release tag yet |

---

## 9. What a Third-Party Reviewer Can Independently Verify

✅ **Can verify:**
- Code structure, test suite, and GroupKFold leakage-prevention logic
- Fold assignment and LOAO cohort structure
- Feature construction (17 adjacent differences) from reported channel list
- Aggregation strategy and multi-spot table
- Provenance manifests (MD5 hashes, cohort counts, model SHA256s)
- Benchmark output CSVs and confusion matrices
- The model manifest entries and their stated performance metrics

❌ **Cannot verify without raw data:**
- The actual 700-spectrum ROI matrix and its extraction from .bil cubes
- Whether reported scan-level results match re-extraction from raw cubes
- The 417-row Brix dataset and its provenance chain
- Physical sensor behavior (no hardware data present)
