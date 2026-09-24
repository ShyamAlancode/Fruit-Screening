"""
hardware_calibration_tool.py
=============================
Standalone command-line calibration tool for the AMS AS7265x sensor.

Runs the full dark → white → UV crosstalk → sweep protocol and saves
calibration files to data/calibrations/<session_id>/

Usage:
    python scripts/hardware_calibration_tool.py --mode uart --port /dev/serial0
    python scripts/hardware_calibration_tool.py --mode i2c --bus 1
    python scripts/hardware_calibration_tool.py --mode mock   # test without hardware

Outputs (saved to data/calibrations/<timestamp>/):
    dark_ref.npy         — 18-channel dark reference vector
    white_ref.npy        — 18-channel white reference vector
    crosstalk_report.txt — UV405 crosstalk per channel (%)
    calibration.json     — summary metadata
"""

import sys, os, json, argparse, logging, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.as7265x_driver import AS7265xDriver, AS7265X_CENTRES_NM
from src.calibration import validate_calibration_pair


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol functions
# ---------------------------------------------------------------------------

def step_dark(driver: AS7265xDriver, n: int = 10) -> np.ndarray:
    """Step 2: Dark calibration — cover sensor completely."""
    print("\n" + "=" * 60)
    print("STEP 2: DARK CALIBRATION")
    print("=" * 60)
    print("  Action required: Cover the sensor window completely")
    print("  (light-tight cap, black tape, or opaque cloth)")
    input("  Press ENTER when sensor is covered and LEDs are OFF > ")
    dark = driver.measure_dark(n_readings=n)
    print(f"  Dark reference captured. Mean counts: {dark.mean():.1f}")
    # Assert
    max_dark = dark.max()
    if max_dark > 200:
        print(f"  WARNING: Max dark count {max_dark:.0f} exceeds 200. "
              "Check for light leaks or reduce integration time.")
    else:
        print(f"  OK: Max dark count {max_dark:.0f} < 200.")
    return dark


def step_white(driver: AS7265xDriver, n: int = 10) -> np.ndarray:
    """Step 3: White reference calibration — Teflon / Spectralon tile."""
    print("\n" + "=" * 60)
    print("STEP 3: WHITE REFERENCE CALIBRATION")
    print("=" * 60)
    print("  Action required: Place Spectralon or Teflon tile 5mm below sensor")
    input("  Press ENTER when tile is in position > ")
    white = driver.measure_white(n_readings=n)
    print(f"  White reference captured. Mean counts: {white.mean():.1f}")
    is_valid, val_msg = validate_calibration_pair(driver._dark_ref, white)
    if not is_valid:
        print(f"  WARNING: Calibration validation warning: {val_msg}")
    else:
        print(f"  OK: {val_msg}")
    return white



def step_uv_crosstalk(
    driver: AS7265xDriver,
    dark_ref: np.ndarray,
    n: int = 5,
) -> dict:
    """
    Step 4: UV405 crosstalk test.
    Protocol: scan white tile, then black card, then white tile again.
    Returns per-channel crosstalk percentages.
    """
    print("\n" + "=" * 60)
    print("STEP 4: UV405 CROSSTALK TEST")
    print("=" * 60)
    print("  This test checks if the NIR chip's 405nm LED bleeds into other channels.")

    input("  Place WHITE tile under sensor. Press ENTER > ")
    white_readings = np.stack([driver.read_raw() for _ in range(n)]).mean(axis=0)

    input("  Replace tile with BLACK card. Press ENTER > ")
    black_readings = np.stack([driver.read_raw() for _ in range(n)]).mean(axis=0)

    input("  Replace back to WHITE tile. Press ENTER > ")
    white2_readings = np.stack([driver.read_raw() for _ in range(n)]).mean(axis=0)

    # Crosstalk = |ch_410_black - ch_410_white| / ch_410_white * 100
    white_avg = (white_readings + white2_readings) / 2
    crosstalk_pct = np.abs(black_readings - white_avg) / (white_avg - dark_ref + 1e-6) * 100

    report = {}
    print("\n  UV Crosstalk Report:")
    print(f"  {'Channel':>8s} {'Crosstalk %':>12s} {'Status':>10s}")
    print(f"  {'-'*35}")
    for i, nm in enumerate(AS7265X_CENTRES_NM):
        ct = float(crosstalk_pct[i])
        status = "PASS" if ct < 5.0 else "WARN >5% — apply correction"
        report[f"ch_{nm}nm"] = {"crosstalk_pct": ct, "status": status}
        print(f"  {nm:>5d} nm  {ct:>10.2f}%  {status}")

    return report


def step_sweep(driver: AS7265xDriver, n_apples: int = 5) -> list:
    """
    Step 5: Real apple data collection sweep.
    Prompts operator to position each apple and records n_apples scans.
    """
    print("\n" + "=" * 60)
    print("STEP 5: APPLE SCAN SWEEP (Real Data Collection)")
    print("=" * 60)
    print(f"  Recording {n_apples} apples. For each: assign fruit_id, scan 3 times.")
    records = []
    for apple_num in range(1, n_apples + 1):
        fruit_id = f"fruit_{apple_num:04d}"
        treatment = input(f"  Apple {apple_num}/{n_apples} — Enter treatment class (0=control, 1=low, 2=high): ").strip()
        scans = []
        for rep in range(1, 4):
            input(f"    Scan {rep}/3 — Position apple and press ENTER > ")
            raw  = driver.read_raw()
            cal  = driver.manual_calibrate(raw)
            temp = driver.get_device_temperature()
            scans.append({
                "rep": rep,
                "timestamp": datetime.utcnow().isoformat(),
                "temperature_c": temp,
                "raw": raw.tolist(),
                "calibrated": cal.tolist(),
            })
            print(f"    Scan {rep}/3 captured. Temp={temp:.1f}°C")
        records.append({
            "fruit_id": fruit_id,
            "treatment_class": int(treatment) if treatment.isdigit() else -1,
            "scans": scans,
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AS7265x Hardware Calibration Tool — CFRD Project"
    )
    parser.add_argument("--mode", choices=["uart", "i2c", "mock"], default="mock",
                        help="Sensor communication mode (default: mock)")
    parser.add_argument("--port", default="/dev/serial0",
                        help="Serial port for UART mode (default: /dev/serial0)")
    parser.add_argument("--bus", type=int, default=1,
                        help="I2C bus for I2C mode (default: 1)")
    parser.add_argument("--addr", type=int, default=0x49,
                        help="I2C address (default: 0x49)")
    parser.add_argument("--gain", type=int, choices=[1, 3, 16, 64], default=64,
                        help="Sensor gain (default: 64)")
    parser.add_argument("--integration", type=int, default=50,
                        help="Integration time in ms (default: 50)")
    parser.add_argument("--n-dark", type=int, default=10,
                        help="Dark calibration readings (default: 10)")
    parser.add_argument("--n-white", type=int, default=10,
                        help="White calibration readings (default: 10)")
    parser.add_argument("--sweep-apples", type=int, default=0,
                        help="Number of apples to scan in sweep (0 = skip)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: data/calibrations/<timestamp>)")
    args = parser.parse_args()

    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "data" / "calibrations" / session_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  AS7265x HARDWARE CALIBRATION TOOL — CFRD Project")
    print("=" * 60)
    print(f"  Mode:     {args.mode}")
    print(f"  Gain:     {args.gain}x")
    print(f"  IntTime:  {args.integration} ms")
    print(f"  Output:   {out_dir}")
    print()

    with AS7265xDriver(
        mode=args.mode,
        port=args.port,
        i2c_bus=args.bus,
        i2c_addr=args.addr,
    ) as driver:

        # Step 1: Configuration
        print("\n" + "=" * 60)
        print("STEP 1: SENSOR CONFIGURATION")
        print("=" * 60)
        driver.set_gain(args.gain)
        driver.set_integration_time(args.integration)
        print(f"  Gain={args.gain}x, IntTime={args.integration}ms — OK")

        # Step 2: Dark
        dark_ref = step_dark(driver, n=args.n_dark)
        np.save(out_dir / "dark_ref.npy", dark_ref)
        print(f"  Saved: {out_dir / 'dark_ref.npy'}")

        # Step 3: White
        white_ref = step_white(driver, n=args.n_white)
        np.save(out_dir / "white_ref.npy", white_ref)
        print(f"  Saved: {out_dir / 'white_ref.npy'}")

        # Step 4: UV Crosstalk
        crosstalk = step_uv_crosstalk(driver, dark_ref)
        crosstalk_file = out_dir / "crosstalk_report.txt"
        with open(crosstalk_file, "w") as f:
            f.write("UV405 Crosstalk Report\n")
            f.write(f"Session: {session_id}\n\n")
            for ch, val in crosstalk.items():
                f.write(f"{ch}: {val['crosstalk_pct']:.2f}%  [{val['status']}]\n")
        print(f"  Saved: {crosstalk_file}")

        # Step 5: Sweep (optional)
        sweep_records = []
        if args.sweep_apples > 0:
            sweep_records = step_sweep(driver, n_apples=args.sweep_apples)

        # Summary JSON
        summary = {
            "session_id": session_id,
            "mode": args.mode,
            "gain": args.gain,
            "integration_time_ms": args.integration,
            "dark_mean": float(dark_ref.mean()),
            "dark_max":  float(dark_ref.max()),
            "white_mean": float(white_ref.mean()),
            "white_signal_mean": float((white_ref - dark_ref).mean()),
            "crosstalk": crosstalk,
            "sweep_apples": sweep_records,
            "centres_nm": AS7265X_CENTRES_NM,
        }
        summary_file = out_dir / "calibration.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved: {summary_file}")

    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE")
    print(f"  All outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
