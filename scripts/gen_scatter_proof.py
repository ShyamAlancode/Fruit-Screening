"""
gen_scatter_proof.py - Generate proof scatter plots for improved Brix model
"""
import re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR  = PROJECT_ROOT / "reports"
AUDIT_JSON   = PROJECT_ROOT / "records_audit.json"
ESSEX_ROOT   = Path(os.environ.get("CFRD_ESSEX_ROOT", PROJECT_ROOT.parent / "New folder"))
EXTRACTED    = ESSEX_ROOT / "extracted"
CSV_NZ23     = ESSEX_ROOT / "data_NZ2023.csv"
CSV_NZ24     = ESSEX_ROOT / "data_NZ2024.csv"

AS7265X_CENTRES = [410,435,460,485,510,535,560,585,610,645,680,705,730,760,810,860,900,940]
SIGMA = 20.0 / 2.35482

def parse_wl(hdr_path):
    txt = open(hdr_path).read()
    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", txt, re.DOTALL)
    return np.array([float(x.strip()) for x in m.group(1).split(",") if x.strip()])

def spec_to_18ch(spec, wl):
    out = np.zeros(18)
    for k, c in enumerate(AS7265X_CENTRES):
        G = np.exp(-0.5*((wl-c)/SIGMA)**2)
        mask = G > 1e-4
        w = G[mask].sum()
        out[k] = float(np.dot(spec[mask], G[mask])/w) if w > 1e-10 else 0.0
    return out

def get_cultivar(apple_id, nz23, nz24):
    for df in [nz24, nz23]:
        if "ID" in df.columns:
            row = df[df["ID"] == apple_id]
            if len(row):
                return str(row.iloc[0]["Cultivar"])
    return apple_id.split("_")[0] if "_" in apple_id else "Unknown"


def main():
    with open(AUDIT_JSON) as f:
        records = json.load(f)

    nz23 = pd.read_csv(CSV_NZ23)
    nz24 = pd.read_csv(CSV_NZ24)

    X204_all, X18_all = [], []
    y_brix, cultivars = [], []

    for rec in records:
        if not rec["csv_rows"]:
            continue
        base = EXTRACTED / rec["batch"] / rec["folder"] / "results"
        dats = sorted(base.glob("REFLECTANCE_*.dat"))
        hdrs = sorted(base.glob("REFLECTANCE_*.hdr"))
        if not dats or not hdrs:
            continue
        cube = np.array(np.memmap(str(dats[0]), dtype=np.float32, mode="r", shape=(512,512,204)))
        wl   = parse_wl(str(hdrs[0]))

        blobs    = rec["blob_positions"]
        csv_rows = rec["csv_rows"]
        n_use    = min(len(blobs), len(csv_rows))

        for i in range(n_use):
            blob = blobs[i]
            row  = csv_rows[i]
            if row["Brix"] <= 0:
                continue
            y1, y2 = blob["y_start"], blob["y_stop"]
            x1, x2 = blob["x_start"], blob["x_stop"]
            dy, dx = int((y2-y1)*0.30), int((x2-x1)*0.30)
            cy1, cy2 = max(y1+dy, 0), min(y2-dy, 512)
            cx1, cx2 = max(x1+dx, 0), min(x2-dx, 512)
            if cy2 <= cy1 or cx2 <= cx1:
                cy1, cy2, cx1, cx2 = y1, y2, x1, x2
            sub  = cube[cy1:cy2, cx1:cx2, :]
            spec = sub.mean(axis=(0,1)).astype(np.float64)
            ch18 = spec_to_18ch(spec, wl)

            X204_all.append(spec)
            X18_all.append(ch18)
            y_brix.append(row["Brix"])
            cultivars.append(get_cultivar(row["ID"], nz23, nz24))

    X204 = np.array(X204_all, dtype=np.float64)
    X18  = np.array(X18_all,  dtype=np.float64)
    y    = np.array(y_brix,   dtype=np.float64)
    cult = np.array(cultivars)

    print(f"N={len(y)}, cultivars: {sorted(set(cult))}")
    for c in sorted(set(cult)):
        sub = y[cult == c]
        print(f"  {c}: N={len(sub)}, Brix {sub.min():.1f}-{sub.max():.1f}")

    # One-hot encode cultivar
    le = LabelEncoder()
    codes   = le.fit_transform(cult)
    onehot  = np.eye(len(le.classes_), dtype=np.float64)[codes]

    X18c  = np.hstack([X18,  onehot])
    X204c = np.hstack([X204, onehot])

    kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)

    # --- 18-ch PLSR + cultivar ---
    n_comp_18 = min(10, X18c.shape[1]-1, len(y)-1)
    plsr18 = PLSRegression(n_components=n_comp_18, max_iter=2000)
    Xs18   = StandardScaler().fit_transform(X18c)
    yp_plsr18 = cross_val_predict(plsr18, Xs18, y, cv=kf)
    r2_plsr18 = r2_score(y, yp_plsr18)
    rmse_plsr18 = np.sqrt(mean_squared_error(y, yp_plsr18))
    r_plsr18, p_plsr18 = pearsonr(y, yp_plsr18)
    print(f"PLSR 18ch+cultivar: R2={r2_plsr18:.4f} RMSE={rmse_plsr18:.3f} r={r_plsr18:.4f}")

    # --- 204-band Ridge + cultivar ---
    n_comp_204 = min(10, X204c.shape[1]-1, len(y)-1)
    Xs204 = StandardScaler().fit_transform(X204c)
    yp_ridge204 = cross_val_predict(Ridge(alpha=0.5), Xs204, y, cv=kf)
    r2_ridge204 = r2_score(y, yp_ridge204)
    rmse_ridge204 = np.sqrt(mean_squared_error(y, yp_ridge204))
    r_ridge204, p_ridge204 = pearsonr(y, yp_ridge204)
    print(f"Ridge 204+cultivar: R2={r2_ridge204:.4f} RMSE={rmse_ridge204:.3f} r={r_ridge204:.4f}")

    # --- Colors ---
    unique_c = sorted(set(cult))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]
    cmap_c = {c: colors[i % len(colors)] for i, c in enumerate(unique_c)}

    def make_scatter(y_true, y_pred, title, out_path, r2, rmse, pearson_r, pval, cult_labels):
        fig, ax = plt.subplots(figsize=(7.5, 7))
        for c in sorted(set(cult_labels)):
            mask = np.array(cult_labels) == c
            ax.scatter(y_true[mask], y_pred[mask], label=c, color=cmap_c[c],
                       s=100, edgecolors="k", linewidths=0.6, zorder=3, alpha=0.9)

        lo = min(y_true.min(), y_pred.min()) - 0.5
        hi = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, label="Perfect prediction", zorder=2)
        xs = np.linspace(lo, hi, 100)
        z = np.polyfit(y_true, y_pred, 1)
        ax.plot(xs, np.poly1d(z)(xs), color="steelblue", lw=2.0, alpha=0.85, label="Trend line", zorder=2)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Refractometer Brix (Ground Truth, deg Bx)", fontsize=12)
        ax.set_ylabel("Model Predicted Brix (deg Bx)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(title="Cultivar", fontsize=10, loc="upper left")
        ax.text(0.97, 0.04,
                f"R2 = {r2:.4f}\nRMSE = {rmse:.3f} dBx\nPearson r = {pearson_r:.4f}\np = {pval:.2e}\nN = {len(y_true)}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {Path(out_path).name}")

    make_scatter(y, yp_plsr18,
                 "AS7265x 18-Ch Digital Twin + Cultivar --> Brix\n(PLSR, 5-Fold CV, Essex Specim IQ Ground Truth)",
                 REPORTS_DIR / "brix_proof_18ch_plsr_cultivar.png",
                 r2_plsr18, rmse_plsr18, r_plsr18, p_plsr18, cult)

    make_scatter(y, yp_ridge204,
                 "Real 204-Band Specim IQ + Cultivar --> Brix\n(Ridge, 5-Fold CV, Essex Specim IQ Ground Truth)",
                 REPORTS_DIR / "brix_proof_204band_ridge_cultivar.png",
                 r2_ridge204, rmse_ridge204, r_ridge204, p_ridge204, cult)

    # Comparison bar chart
    labels = ["Naive\n204-band", "Naive\n18-ch", "Improved\n204+Cultivar", "Improved\n18ch+Cultivar"]
    r2s    = [-0.0139,            0.0186,          r2_ridge204,              r2_plsr18]
    rmses  = [1.506,              1.482,            rmse_ridge204,            rmse_plsr18]
    colors_bar = ["#E67E22","#E67E22","#27AE60","#27AE60"]
    alpha_bar  = [0.6, 0.6, 1.0, 1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    bars = ax1.bar(labels, r2s, color=colors_bar, alpha=0.85, edgecolor="k", linewidth=0.7)
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("R2 (Cross-Validated)", fontsize=12)
    ax1.set_title("Model Comparison: R2", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.2, max(r2s)+0.15)
    for bar, val in zip(bars, r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, max(val+0.01, 0.01),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(labels, rmses, color=colors_bar, alpha=0.85, edgecolor="k", linewidth=0.7)
    ax2.set_ylabel("RMSE (deg Bx)", fontsize=12)
    ax2.set_title("Model Comparison: RMSE", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(rmses)+0.2)
    for bar, val in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, val+0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Effect of Cultivar Feature on Brix Prediction Accuracy\n(N=30 apples, 3 cultivars: Braeburn, Cox, Gala)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "brix_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: brix_comparison_bar.png")

    print("\nDONE.")
    print(f"  Naive 204-band:         R2 = -0.0139")
    print(f"  Naive 18-ch twin:       R2 =  0.0186")
    print(f"  Improved 204+cultivar:  R2 = {r2_ridge204:.4f}   RMSE = {rmse_ridge204:.3f} dBx")
    print(f"  Improved 18ch+cultivar: R2 = {r2_plsr18:.4f}   RMSE = {rmse_plsr18:.3f} dBx")


if __name__ == "__main__":
    main()
