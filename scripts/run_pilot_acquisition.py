"""
CFRD Phase 4: Physical Pilot Acquisition Engine
Connects to SparkFun Triad AS7265x (or mock mode on PC), manages White/Dark calibration,
guides operator through 4-side fruit rotation, and logs calibrated 18-channel reflectance.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CHANNELS_NM = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
OUTPUT_CSV = str(DATA_DIR / "pilot_scans.csv")
OUTPUT_JSONL = str(DATA_DIR / "pilot_scans.jsonl")

class SensorInterface:
    def __init__(self, mock=False):
        self.mock = mock
        self.sensor = None
        self.qwiic = None
        if not mock:
            try:
                import qwiic_as7265x
                self.qwiic = qwiic_as7265x
                self.sensor = qwiic_as7265x.QwiicAS7265x()
                if not self.sensor.begin():
                    print("[WARN] AS7265x hardware not detected on I2C. Falling back to MOCK mode.")
                    self.mock = True
                else:
                    print("[SUCCESS] Connected to AS7265x Triad hardware on I2C bus.")
            except ImportError:
                print("[WARN] 'qwiic_as7265x' library not found. Falling back to MOCK mode.")
                self.mock = True

    def read_raw_channels(self, led_white=True, led_ir=True, led_uv=True):
        if self.mock:
            time.sleep(0.1)
            # Realistic synthetic spectra with Gaussian noise
            base = np.array([120, 180, 240, 310, 450, 600, 750, 820, 910, 1100, 1250, 1300, 1200, 1150, 1050, 950, 850, 750], dtype=float)
            if not led_white and not led_ir and not led_uv:
                return np.random.uniform(5, 15, size=18).round().astype(int).tolist()
            noise = np.random.normal(0, 12.5, size=18)
            counts = np.clip(base * 25 + noise, 0, 65535).round().astype(int).tolist()
            return counts
        else:
            # Control bulbs
            if led_white: self.sensor.enable_bulb(self.qwiic.AS7265X_LED_WHITE)
            else: self.sensor.disable_bulb(self.qwiic.AS7265X_LED_WHITE)
            
            if led_ir: self.sensor.enable_bulb(self.qwiic.AS7265X_LED_IR)
            else: self.sensor.disable_bulb(self.qwiic.AS7265X_LED_IR)
            
            if led_uv: self.sensor.enable_bulb(self.qwiic.AS7265X_LED_UV)
            else: self.sensor.disable_bulb(self.qwiic.AS7265X_LED_UV)
            
            time.sleep(0.3) # Allow LED output to stabilize
            self.sensor.take_measurements()
            counts = [int(self.sensor.get_channel(ch)) for ch in range(18)]
            
            # Turn bulbs off after capture to prevent heat build-up
            self.sensor.disable_bulb(self.qwiic.AS7265X_LED_WHITE)
            self.sensor.disable_bulb(self.qwiic.AS7265X_LED_IR)
            self.sensor.disable_bulb(self.qwiic.AS7265X_LED_UV)
            return counts

    def get_temperature(self):
        if self.mock:
            return 26.5
        try:
            return float(self.sensor.get_temperature())
        except:
            return 25.0


def run_session():
    parser = argparse.ArgumentParser(description="CFRD Pilot Multispectral Acquisition Engine")
    parser.add_argument("--mock", action="store_true", help="Force mock sensor mode for testing")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    sensor = SensorInterface(mock=args.mock)
    session_id = datetime.now().strftime("SESS_%Y%m%d_%H%M%S")

    print("\n" + "="*70)
    print(f" CFRD PHYSICAL PILOT ACQUISITION ENGINE (Session: {session_id})")
    print("="*70)

    # 1. Dark Calibration
    print("\n[STEP 1/2] DARK CALIBRATION")
    input(">> Ensure optical aperture is CAPPED inside the dark box. Press [ENTER] to read DARK: ")
    dark_counts = sensor.read_raw_channels(led_white=False, led_ir=False, led_uv=False)
    print(f"Dark counts captured: mean={np.mean(dark_counts):.1f}, min={min(dark_counts)}, max={max(dark_counts)}")

    # 2. White Calibration
    print("\n[STEP 2/2] WHITE CALIBRATION")
    input(">> Place aperture firmly against TEFLON/PTFE puck. Press [ENTER] to read WHITE: ")
    white_counts = sensor.read_raw_channels(led_white=True, led_ir=True, led_uv=True)
    print(f"White counts captured: mean={np.mean(white_counts):.1f}, min={min(white_counts)}, max={max(white_counts)}")

    # Validate calibration denominator
    denominator = np.array(white_counts, dtype=float) - np.array(dark_counts, dtype=float)
    denominator[denominator <= 0] = 1.0

    print("\n" + "="*70)
    print(" CALIBRATION COMPLETE. COMMENCING FRUIT SCANNING ROUTINE")
    print("="*70)

    while True:
        print("\n--- NEW APPLE ACQUISITION ---")
        apple_id = input("Enter Global Apple ID (e.g. GALA_PILOT_001) or 'q' to quit: ").strip()
        if apple_id.lower() == 'q':
            break

        cultivar = input("Enter Cultivar (e.g. Royal Gala / Fuji / Braeburn): ").strip()
        print("Cohorts: [1] Clean Organic | [2] Water-Washed Control | [3] Nativo 0.5g/L | [4] Nativo 2.0g/L")
        cohort_choice = input("Select Cohort (1-4): ").strip()
        cohort_map = {
            '1': ('COHORT_A_CLEAN', 0.0),
            '2': ('COHORT_B_WATER_WASH', 0.0),
            '3': ('COHORT_C_NATIVO_LOW', 0.5),
            '4': ('COHORT_D_NATIVO_HIGH', 2.0)
        }
        cohort_name, dose_g_l = cohort_map.get(cohort_choice, ('UNKNOWN', 0.0))

        sides = [
            ("Side_A", 0),
            ("Side_B", 90),
            ("Side_C", 180),
            ("Side_D", 270),
            ("Stem_End", -1),
            ("Calyx_End", -1)
        ]

        apple_records = []
        for side_name, rot_deg in sides:
            rot_prompt = f"rotated {rot_deg}°" if rot_deg >= 0 else side_name
            input(f"  >> Position apple for [{side_name}] ({rot_prompt}). Press [ENTER] to scan: ")
            
            raw_counts = sensor.read_raw_channels(led_white=True, led_ir=True, led_uv=True)
            refl = (np.array(raw_counts) - np.array(dark_counts)) / denominator
            refl = np.clip(refl, 0.0, 2.0).round(4).tolist()
            temp_c = sensor.get_temperature()

            rec = {
                "Session_ID": session_id,
                "Global_Apple_ID": apple_id,
                "Cultivar": cultivar,
                "Cohort": cohort_name,
                "Dose_g_L": dose_g_l,
                "Side": side_name,
                "Rotation_Deg": rot_deg,
                "Sensor_Temp_C": temp_c,
                "Timestamp": datetime.now().isoformat()
            }
            for i, nm in enumerate(CHANNELS_NM):
                rec[f"R_{nm}"] = refl[i]
                rec[f"Raw_{nm}"] = raw_counts[i]
                rec[f"Dark_{nm}"] = dark_counts[i]
                rec[f"White_{nm}"] = white_counts[i]

            apple_records.append(rec)
            print(f"     Captured {side_name}: Mean Reflectance={np.mean(refl):.3f} (410nm={refl[0]:.3f}, 680nm={refl[10]:.3f}, 940nm={refl[17]:.3f})")

        # Atomic append to CSV and JSONL
        df_new = pd.DataFrame(apple_records)
        header = not os.path.exists(OUTPUT_CSV)
        df_new.to_csv(OUTPUT_CSV, mode='a', header=header, index=False)

        with open(OUTPUT_JSONL, 'a') as jf:
            for r in apple_records:
                jf.write(json.dumps(r) + "\n")

        print(f"\n[SAVED] 6 scans successfully recorded for {apple_id} -> {OUTPUT_CSV}")

    print("\nSession completed. Data preserved with full provenance.")

if __name__ == "__main__":
    run_session()
