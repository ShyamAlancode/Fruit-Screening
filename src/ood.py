"""
ood.py

Out-of-distribution detection using Mahalanobis distance.
Used for:
  1. Confidence rejection — thresholded acceptance on Roomi synthetic data
  2. Cross-dataset domain-shift stress test — SpectroFood vs Roomi-trained model
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    CHANNEL_COLS, CONFIDENCE_THRESHOLDS, COVERAGE_FORMAT,
    OOD_EXPERIMENT_LABEL, N_CHANNELS,
)


# ---------------------------------------------------------------------------
# Mahalanobis distance scorer
# ---------------------------------------------------------------------------

class MahalanobisOOD:
    """
    Mahalanobis distance OOD scorer using LedoitWolf shrinkage covariance.
    Fitted on in-distribution training data (Roomi synthetic rows) only.

    Parameters
    ----------
    n_pca_components : int
        Apply PCA before Mahalanobis to avoid singular covariance.
        Set to None to skip PCA (not recommended for 18-dim data).
    """

    def __init__(self, n_pca_components: int = 5):
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n_pca_components) if n_pca_components else None
        self.cov_estimator = LedoitWolf()
        self.mean_    = None
        self.cov_inv_ = None
        self.fitted_  = False
        self._train_distances = None

    def fit(self, X_train: np.ndarray) -> "MahalanobisOOD":
        """Fit on in-distribution training data only."""
        Z = self.scaler.fit_transform(X_train)
        if self.pca:
            Z = self.pca.fit_transform(Z)
        self.cov_estimator.fit(Z)
        self.mean_    = Z.mean(axis=0)
        self.cov_inv_ = self.cov_estimator.precision_   # LedoitWolf provides precision matrix
        self._train_distances = self._mahal(Z)
        self.fitted_ = True
        return self

    def _mahal(self, Z: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each row of Z from training mean."""
        diff = Z - self.mean_
        # d^2 = (x - mu)^T Sigma^-1 (x - mu)
        d_sq = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        return np.sqrt(np.clip(d_sq, 0, None))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distance for each row (lower = more in-distribution)."""
        assert self.fitted_, "Call fit() before score()"
        Z = self.scaler.transform(X)
        if self.pca:
            Z = self.pca.transform(Z)
        return self._mahal(Z)

    def threshold_at_percentile(self, percentile: float = 95.0) -> float:
        """Return distance threshold corresponding to given training percentile."""
        assert self.fitted_, "Call fit() before threshold_at_percentile()"
        return float(np.percentile(self._train_distances, percentile))

    def is_ood(self, X: np.ndarray, percentile: float = 95.0) -> np.ndarray:
        """Return boolean mask: True where sample is OOD (distance > threshold)."""
        threshold = self.threshold_at_percentile(percentile)
        return self.score(X) > threshold


# ---------------------------------------------------------------------------
# Confidence rejection sweep
# ---------------------------------------------------------------------------

def run_rejection_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Sweep confidence thresholds. At each threshold, report:
    - accepted count and coverage %
    - accuracy on accepted scans
    - balanced accuracy on accepted scans
    - treated-class sensitivity and specificity

    Uses OUT-OF-FOLD predictions only (no test-set leakage).

    Parameters
    ----------
    y_true  : (N,) ground truth class labels
    y_pred  : (N,) predicted class labels
    y_proba : (N, C) predicted class probabilities
    thresholds : list of float confidence thresholds to sweep

    Returns
    -------
    list of dicts, one per threshold
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    if thresholds is None:
        thresholds = CONFIDENCE_THRESHOLDS

    max_proba = y_proba.max(axis=1)    # confidence = max softmax probability
    n_total = len(y_true)
    results = []

    if verbose:
        print(f"\n{'='*65}")
        print(f"Confidence Rejection Sweep  [{OOD_EXPERIMENT_LABEL}]")
        print(f"Total samples: {n_total}")
        print(f"{'='*65}")
        print(f"{'Threshold':>10} {'Accepted':>10} {'Coverage%':>10} {'Accuracy%':>10} {'BalAcc%':>10}")
        print(f"{'-'*55}")

    for thresh in thresholds:
        accept_mask = max_proba >= thresh
        n_accepted  = int(accept_mask.sum())
        n_rejected  = n_total - n_accepted
        coverage    = 100.0 * n_accepted / n_total

        if n_accepted == 0:
            acc     = 0.0
            bal_acc = 0.0
        else:
            acc     = 100.0 * accuracy_score(y_true[accept_mask], y_pred[accept_mask])
            bal_acc = 100.0 * balanced_accuracy_score(
                y_true[accept_mask], y_pred[accept_mask]
            )

        row = {
            "threshold": thresh,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "coverage_pct": round(coverage, 2),
            "accepted_accuracy_pct": round(acc, 2),
            "accepted_balanced_accuracy_pct": round(bal_acc, 2),
        }
        results.append(row)

        if verbose:
            print(
                f"{thresh:>10.2f} {n_accepted:>10} {coverage:>9.1f}% "
                f"{acc:>9.1f}% {bal_acc:>9.1f}%"
            )
            # Print mandatory format string
            print(f"  {COVERAGE_FORMAT.format(acc=acc, cov=coverage)}")

    if verbose:
        print(f"{'='*65}")

    return results


# ---------------------------------------------------------------------------
# SpectroFood domain-shift stress test
# ---------------------------------------------------------------------------

def run_domain_shift_test(
    X_in: np.ndarray,
    X_ood_candidate: np.ndarray,
    ood_scorer: MahalanobisOOD,
    pipeline,
    percentile: float = 95.0,
    verbose: bool = True,
) -> dict:
    """
    Cross-dataset domain-shift stress test.

    Tests SpectroFood apple spectra against a Roomi-trained OOD model.
    Reports:
    - What % of SpectroFood samples exceed the in-distribution threshold
    - What % of SpectroFood samples get confidently called 'treated-like'
      (a false alarm risk indicator)

    This is an EXPLORATORY experiment only. See OOD_EXPERIMENT_LABEL for
    the correct experiment name.

    Parameters
    ----------
    X_in            : In-distribution samples (Roomi training data)
    X_ood_candidate : Candidate OOD samples (SpectroFood 18-channel)
    ood_scorer      : Fitted MahalanobisOOD (fitted on Roomi training data only)
    pipeline        : Fitted Roomi treatment-condition classifier (frozen)
    percentile      : Threshold percentile on training distance distribution
    """
    assert ood_scorer.fitted_, "OOD scorer must be fitted on Roomi training data first"

    threshold = ood_scorer.threshold_at_percentile(percentile)
    distances_in   = ood_scorer.score(X_in)
    distances_ood  = ood_scorer.score(X_ood_candidate)

    n_ood = len(X_ood_candidate)
    n_flagged = int((distances_ood > threshold).sum())
    pct_flagged = 100.0 * n_flagged / n_ood

    # Predictions from Roomi-trained classifier on SpectroFood samples
    preds = pipeline.predict(X_ood_candidate)
    try:
        proba = pipeline.predict_proba(X_ood_candidate)
        max_conf = proba.max(axis=1)
        n_confident_treated = int(((preds != 0) & (max_conf >= 0.80)).sum())
    except AttributeError:
        max_conf = np.zeros(n_ood)
        n_confident_treated = int((preds != 0).sum())

    pct_false_alarm = 100.0 * n_confident_treated / n_ood

    result = {
        "experiment": OOD_EXPERIMENT_LABEL,
        "n_in_distribution": len(X_in),
        "n_ood_candidate": n_ood,
        "threshold_percentile": percentile,
        "distance_threshold": round(threshold, 4),
        "in_dist_mean_distance": round(float(distances_in.mean()), 4),
        "in_dist_95th_distance": round(float(np.percentile(distances_in, 95)), 4),
        "ood_mean_distance": round(float(distances_ood.mean()), 4),
        "ood_95th_distance": round(float(np.percentile(distances_ood, 95)), 4),
        "n_ood_flagged_as_uncertain": n_flagged,
        "pct_ood_flagged_as_uncertain": round(pct_flagged, 2),
        "n_ood_false_alarm_treated_confident": n_confident_treated,
        "pct_ood_false_alarm": round(pct_false_alarm, 2),
    }

    if verbose:
        print(f"\n{'='*65}")
        print(f"DOMAIN-SHIFT STRESS TEST: {OOD_EXPERIMENT_LABEL}")
        print(f"{'='*65}")
        print(f"  In-distribution (Roomi):   {len(X_in)} samples")
        print(f"  OOD candidate (SpectroFood): {n_ood} samples")
        print(f"  Distance threshold (p{percentile:.0f}): {threshold:.4f}")
        print(f"")
        print(f"  Roomi mean distance:       {result['in_dist_mean_distance']:.4f}")
        print(f"  SpectroFood mean distance: {result['ood_mean_distance']:.4f}")
        print(f"")
        print(f"  SpectroFood flagged UNCERTAIN: {n_flagged}/{n_ood} = {pct_flagged:.1f}%")
        print(f"  SpectroFood false alarm (confident treated-like): {n_confident_treated}/{n_ood} = {pct_false_alarm:.1f}%")
        print(f"")
        print(f"  NOTE: Distance shift may reflect instrument/source difference,")
        print(f"        not biological treatment status. This is exploratory only.")
        print(f"{'='*65}")

    return result
