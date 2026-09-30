"""
cnn_1d_spectrum.py
==================
1D-CNN for 18-channel multispectral spectrum classification.

Design rationale:
- Small n (26-240 samples) means DL will likely OVERFIT versus classical ML.
- The scientific contribution is the comparison table, not the CNN accuracy.
- We use the same LOAO protocol as the classical benchmark — identical folds,
  identical preprocessing — so the comparison is valid.
- Architecture kept minimal (2 conv layers, 64 filters max) to limit
  parameter count relative to sample size.

Reference: "demonstrated classical ML outperforms DL under small-sample-size
biological data, with quantified evidence" — exactly what makes this
comparison publishable and resume-worthy.

Usage:
    from cnn_1d_spectrum import SpectralCNN1D, train_loao_cnn, compare_to_classical
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

# Try to import torch — graceful failure if not installed
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TORCH_AVAILABLE = False
    logger.warning(
        f"PyTorch unavailable ({type(e).__name__}). SpectralCNN1D requires functional torch DLLs."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class SpectralCNN1D(nn.Module):
        """
        Minimal 1D-CNN for 18-channel multispectral spectrum classification.

        Architecture:
            Input: (batch, 18) -> reshape to (batch, 1, 18)
            Conv1(1->32, k=3) -> BN -> ReLU -> Dropout(0.3)
            Conv2(32->64, k=3) -> BN -> ReLU -> Dropout(0.3)
            GlobalAvgPool -> FC(64->n_classes)

        Parameter count with n_classes=3: ~8,800 params.
        This is intentionally SMALL — DL at n=26-240 samples needs
        aggressive regularization or it will overfit badly.

        Inputs to forward(): (batch_size, 18) float32 tensor
        Outputs: (batch_size, n_classes) logits (not softmax)
        """

        def __init__(self, n_channels: int = 18, n_classes: int = 3,
                     dropout: float = 0.3):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes

            self.conv_block = nn.Sequential(
                # Block 1: local spectral patterns (adjacent channel relationships)
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),

                # Block 2: mid-range spectral patterns
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Global average pooling collapses the spectral dimension
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

            # Classification head
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, 18) -> (batch, 1, 18)
            x = x.unsqueeze(1)
            x = self.conv_block(x)           # (batch, 64, 18)
            x = self.global_avg_pool(x)      # (batch, 64, 1)
            x = x.squeeze(-1)                # (batch, 64)
            return self.classifier(x)        # (batch, n_classes)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 3,
    n_epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 16,
    dropout: float = 0.4,
    weight_decay: float = 1e-3,
    random_state: int = 42,
    device: str = "cpu",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Train one fold of the 1D-CNN and return val metrics.

    Returns:
        (val_accuracy, val_predictions, val_probabilities)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. pip install torch>=2.0.0")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralCNN1D(
        n_channels=X_train.shape[1],
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    # Class-balanced loss (important at n=26 where classes are imbalanced)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        logits_val = model(X_vl)
        probs_val = torch.softmax(logits_val, dim=1).cpu().numpy()
        preds_val = logits_val.argmax(dim=1).cpu().numpy()
        y_true_np = y_vl.cpu().numpy()

    acc = float((preds_val == y_true_np).mean())
    return acc, preds_val, probs_val


def run_loao_cnn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int = 3,
    n_epochs: int = 150,
    lr: float = 1e-3,
    dropout: float = 0.4,
    weight_decay: float = 1e-3,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Leave-One-Apple-Out cross-validation with the 1D-CNN.

    IMPORTANT: Uses the SAME LOAO folds as the classical ML benchmark.
    StandardScaler is fitted on training fold only (no leakage).

    Returns dict with:
        fold_accuracies, all_preds, all_true, macro_f1,
        mean_accuracy, std_accuracy, n_params, comparison_note
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. pip install torch>=2.0.0")

    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score

    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies = []
    all_preds = []
    all_true = []
    all_probs = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaler fitted on training fold ONLY (matching classical benchmark)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        held_apple = unique_groups[fold_idx] if fold_idx < len(unique_groups) else "?"
        if verbose:
            print(f"  Fold {fold_idx+1}/{n_splits} (held-out apple: {held_apple}) ...", end=" ")

        acc, preds, probs = train_fold(
            X_train, y_train, X_test, y_test,
            n_classes=n_classes,
            n_epochs=n_epochs, lr=lr,
            dropout=dropout, weight_decay=weight_decay,
            random_state=random_state + fold_idx,
        )

        fold_accuracies.append(acc)
        all_preds.extend(preds.tolist())
        all_true.extend(y_test.tolist())
        all_probs.extend(probs.tolist())

        if verbose:
            print(f"acc={acc:.3f}")

    all_preds_np = np.array(all_preds)
    all_true_np = np.array(all_true)

    macro_f1 = f1_score(all_true_np, all_preds_np, average="macro", zero_division=0)
    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    n_params = 0
    if TORCH_AVAILABLE:
        dummy = SpectralCNN1D(n_channels=X.shape[1], n_classes=n_classes, dropout=dropout)
        n_params = dummy.count_parameters()

    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "macro_f1": macro_f1,
        "all_preds": all_preds,
        "all_true": all_true,
        "all_probs": all_probs,
        "n_params": n_params,
        "n_epochs": n_epochs,
        "dropout": dropout,
        "comparison_note": (
            "1D-CNN vs Classical ML (same LOAO folds, same scaler discipline): "
            "DL likely underperforms at n<250 — comparison table is the contribution."
        ),
    }


def compare_to_classical(
    cnn_results: Dict,
    classical_results: Optional[Dict] = None,
) -> str:
    """
    Format a comparison table between CNN and classical ML results.
    classical_results: dict with keys macro_f1, mean_accuracy, model_name
    """
    lines = [
        "",
        "=" * 65,
        "LOAO BENCHMARK COMPARISON: Classical ML vs 1D-CNN",
        "=" * 65,
        f"{'Method':<30} {'Macro-F1':>10} {'Mean Acc':>10} {'Params':>10}",
        "-" * 65,
    ]

    if classical_results:
        lines.append(
            f"{classical_results.get('model_name', 'Classical ML'):<30} "
            f"{classical_results.get('macro_f1', 0)*100:>9.2f}% "
            f"{classical_results.get('mean_accuracy', 0)*100:>9.2f}% "
            f"{'<1K':>10}"
        )

    cnn_name = f"1D-CNN ({cnn_results['n_params']:,} params)"
    lines.append(
        f"{cnn_name:<30} "
        f"{cnn_results['macro_f1']*100:>9.2f}% "
        f"{cnn_results['mean_accuracy']*100:>9.2f}% "
        f"{cnn_results['n_params']:>10,}"
    )

    lines += [
        "-" * 65,
        "Note: same LOAO folds, same StandardScaler fit on train only.",
        "Expected: classical ML >= 1D-CNN at n<250 (small-sample regime).",
        "The comparison itself is the scientific contribution.",
        "=" * 65,
        "",
    ]

    return "\n".join(lines)
