"""
as7265x_driver.py
=================
Complete driver for the AMS AS7265x 18-channel multi-spectral sensor triad.

Supports two communication modes:
  1. UART / Serial (AT command interface) — recommended for Raspberry Pi
  2. I2C (address 0x49) — direct register access via smbus2

Physical sensor architecture:
  - AS72651 (NIR Master):  730, 760, 810, 860, 900, 940 nm  (channels V-L)
  - AS72652 (Green/Red):   560, 585, 610, 645, 680, 705 nm  (channels G-U)
  - AS72653 (UV/Blue):     410, 435, 460, 485, 510, 535 nm  (channels A-F)

Usage (UART):
    driver = AS7265xDriver(mode="uart", port="/dev/serial0", baud=115200)
    driver.set_gain(64)
    driver.set_integration_time(50)
    driver.set_led(white=True, uv=False)
    raw = driver.read_raw()
    calibrated = driver.read_calibrated()
    driver.close()

Usage (I2C):
    driver = AS7265xDriver(mode="i2c", i2c_bus=1, i2c_addr=0x49)
    ...

Usage (Mock — simulation only, must be EXPLICIT):
    driver = AS7265xDriver(mode="mock")  # You must explicitly request this.

HARDWARE FALLBACK POLICY (Audit Issue 1 — corrected Sep 24 2026):
  This driver does NOT silently fall back to simulation mode.
  If pyserial or smbus2 is not installed, or the port/bus is not found,
  an ImportError or RuntimeError is raised immediately.
  Simulation data must NEVER be mistaken for physical hardware measurements.
  See: data/hardware/SIMULATOR_SMOKE_TESTS_README.md

  To run in simulation, you MUST pass mode='mock' explicitly.
  Data files produced by mode='mock' must be labelled
  data_provenance='software_simulation_only'.
"""

from __future__ import annotations
import time
import struct
import logging
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel definitions (ordered 410→940 nm, matching CHANNEL_COLS)
# ---------------------------------------------------------------------------
AS7265X_CENTRES_NM: List[int] = [
    410, 435, 460, 485, 510, 535,   # AS72653 (UV/Blue) — channels A-F
    560, 585, 610, 645, 680, 705,   # AS72652 (Green/Red) — channels G-U
    730, 760, 810, 860, 900, 940,   # AS72651 (NIR Master) — channels V-L
]

# UART AT command constants
_AT_CMD_TIMEOUT = 2.0   # seconds per command
_AT_BAUD        = 115200
_AT_TERMINATOR  = "\r\n"

# I2C register map (AS72651 master registers)
_I2C_ADDR_DEFAULT   = 0x49
_REG_STATUS         = 0x00
_REG_WRITE          = 0x01
_REG_READ           = 0x02
_REG_DEVICE_TYPE    = 0x00
_REG_HW_VERSION     = 0x01
_REG_FW_VERSION_L   = 0x02
_REG_FW_VERSION_H   = 0x03
_REG_CONTROL_SETUP  = 0x04
_REG_INT_T          = 0x05
_REG_DEVICE_TEMP    = 0x06
_REG_LED_CONFIG     = 0x07
_REG_RAW_VALUE_START = 0x08  # 18 × 2-byte raw values
_REG_CAL_START      = 0x14   # 18 × 4-byte float calibrated values

# Gain settings (AS7265x CONTROL_SETUP[5:4])
GAIN_OPTIONS: Dict[int, int] = {1: 0b00, 3: 0b01, 16: 0b10, 64: 0b11}

# Realistic 18-channel apple baseline reflectance profile (410 to 940 nm)
# Matches SpectroFood / Essex empirical reflectance curves:
# Low blue (410-485nm), green peak (535-560nm), chlorophyll dip (680nm), NIR plateau (730-940nm)
MOCK_APPLE_REFLECTANCE: np.ndarray = np.array([
    0.09, 0.10, 0.11, 0.14, 0.22, 0.38,  # 410, 435, 460, 485, 510, 535 nm
    0.44, 0.42, 0.40, 0.35, 0.22, 0.48,  # 560, 585, 610, 645, 680, 705 nm
    0.72, 0.76, 0.79, 0.81, 0.82, 0.83,  # 730, 760, 810, 860, 900, 940 nm
], dtype=np.float32)

_CURRENT_MAP: Dict[float, int] = {
    12.0: 0b00,
    12.5: 0b00,
    25.0: 0b01,
    50.0: 0b10,
    100.0: 0b11,
}


# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------

class AS7265xDriver:
    """
    Unified driver for the AMS AS7265x 18-channel spectral sensor triad.

    Parameters
    ----------
    mode : str
        "uart"  — UART/Serial AT command mode (recommended for Raspberry Pi via /dev/serial0)
        "i2c"   — Direct I2C register access (requires smbus2)
        "mock"  — Software mock using src/sensor_simulator.py (no hardware needed)
    port : str
        Serial port path for UART mode (e.g. "/dev/serial0", "COM3")
    baud : int
        UART baud rate (default 115200)
    i2c_bus : int
        I2C bus number for I2C mode (default 1 = /dev/i2c-1 on Pi)
    i2c_addr : int
        I2C address (default 0x49)
    """

    def __init__(
        self,
        mode: str = "uart",
        port: str = "/dev/serial0",
        baud: int = _AT_BAUD,
        i2c_bus: int = 1,
        i2c_addr: int = _I2C_ADDR_DEFAULT,
    ):
        self.mode     = mode
        self._ser     = None
        self._bus     = None
        self._addr    = i2c_addr
        self._gain    = 64
        self._int_time = 50  # ms
        self._dark_ref: Optional[np.ndarray] = None
        self._white_ref: Optional[np.ndarray] = None

        if mode == "uart":
            self._init_uart(port, baud)
        elif mode == "i2c":
            self._init_i2c(i2c_bus, i2c_addr)
        elif mode == "mock":
            logger.info("AS7265xDriver: Running in mock/simulation mode.")
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'uart', 'i2c', or 'mock'.")

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _init_uart(self, port: str, baud: int) -> None:
        try:
            import serial
        except ImportError as e:
            raise ImportError(
                "pyserial is required for UART mode. Install: pip install pyserial"
            ) from e
        self._ser = serial.Serial(port, baud, timeout=_AT_CMD_TIMEOUT)
        time.sleep(0.5)
        # Flush startup banner
        self._ser.flushInput()
        fw = self._at_cmd("ATFV")
        logger.info(f"AS7265x firmware version: {fw.strip()}")

    def _init_i2c(self, bus: int, addr: int) -> None:
        try:
            import smbus2
        except ImportError as e:
            raise ImportError(
                "smbus2 is required for I2C mode. Install: pip install smbus2"
            ) from e
        self._bus = smbus2.SMBus(bus)
        # Confirm device presence
        dev_type = self._i2c_read_virtual(0x00)
        logger.info(f"AS7265x I2C device type register: 0x{dev_type:02X}")

    # -----------------------------------------------------------------------
    # UART AT command interface
    # -----------------------------------------------------------------------

    def _at_cmd(self, cmd: str, value: Optional[str] = None) -> str:
        """Send an AT command and return the response string."""
        if self._ser is None:
            raise RuntimeError("Serial port not open (UART mode only)")
        full_cmd = f"{cmd}={value}" if value is not None else cmd
        self._ser.write((full_cmd + "\r\n").encode())
        response_lines = []
        deadline = time.time() + _AT_CMD_TIMEOUT
        while time.time() < deadline:
            line = self._ser.readline().decode("ascii", errors="ignore").strip()
            if line:
                response_lines.append(line)
                if line in ("OK", "ERROR"):
                    break
        return "\n".join(response_lines)

    def _at_read_raw(self) -> np.ndarray:
        """Read 18 raw channel counts via UART (ATCDATA command)."""
        resp = self._at_cmd("ATCDATA")
        # Response format: comma-separated 18 integers on one line before "OK"
        for line in resp.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 18:
                try:
                    return np.array([int(x) for x in parts], dtype=np.float32)
                except ValueError:
                    continue
        raise IOError(f"Could not parse raw data from ATCDATA response: {resp!r}")

    def _at_read_calibrated(self) -> np.ndarray:
        """Read 18 calibrated channel values via UART (ATDATA command)."""
        resp = self._at_cmd("ATDATA")
        for line in resp.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 18:
                try:
                    return np.array([float(x) for x in parts], dtype=np.float32)
                except ValueError:
                    continue
        raise IOError(f"Could not parse calibrated data from ATDATA response: {resp!r}")

    # -----------------------------------------------------------------------
    # I2C register interface
    # -----------------------------------------------------------------------

    def _i2c_status(self) -> int:
        return self._bus.read_byte_data(self._addr, _REG_STATUS)

    def _i2c_write_virtual(self, virtual_reg: int, value: int) -> None:
        """Write to a virtual register through the AS72651 I2C bridge."""
        # Wait until TX buffer is empty
        for _ in range(100):
            if not (self._i2c_status() & 0x02):
                break
            time.sleep(0.001)
        self._bus.write_byte_data(self._addr, _REG_WRITE, virtual_reg | 0x80)
        for _ in range(100):
            if not (self._i2c_status() & 0x02):
                break
            time.sleep(0.001)
        self._bus.write_byte_data(self._addr, _REG_WRITE, value)

    def _i2c_read_virtual(self, virtual_reg: int) -> int:
        """Read from a virtual register through the AS72651 I2C bridge."""
        for _ in range(100):
            if not (self._i2c_status() & 0x02):
                break
            time.sleep(0.001)
        self._bus.write_byte_data(self._addr, _REG_WRITE, virtual_reg)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if self._i2c_status() & 0x01:
                return self._bus.read_byte_data(self._addr, _REG_READ)
            time.sleep(0.001)
        raise TimeoutError(f"I2C read timeout for virtual register 0x{virtual_reg:02X}")

    def _i2c_read_raw(self) -> np.ndarray:
        """Read all 18 raw channel values (uint16) via I2C."""
        values = np.zeros(18, dtype=np.float32)
        for i in range(18):
            reg = _REG_RAW_VALUE_START + (i * 2)
            hi  = self._i2c_read_virtual(reg)
            lo  = self._i2c_read_virtual(reg + 1)
            values[i] = float((hi << 8) | lo)
        return values

    def _i2c_read_calibrated(self) -> np.ndarray:
        """Read all 18 calibrated channel values (float32) via I2C."""
        values = np.zeros(18, dtype=np.float32)
        for i in range(18):
            reg  = _REG_CAL_START + (i * 4)
            raw_bytes = bytes([
                self._i2c_read_virtual(reg),
                self._i2c_read_virtual(reg + 1),
                self._i2c_read_virtual(reg + 2),
                self._i2c_read_virtual(reg + 3),
            ])
            values[i] = struct.unpack(">f", raw_bytes)[0]
        return values

    # -----------------------------------------------------------------------
    # Public API — works across all modes
    # -----------------------------------------------------------------------

    def set_gain(self, gain: int) -> None:
        """
        Set sensor electronic gain.
        Valid values: 1, 3, 16, 64 (sensor specification).
        Higher gain = more sensitivity in dim conditions; more noise risk.
        """
        if gain not in GAIN_OPTIONS:
            raise ValueError(f"gain must be one of {list(GAIN_OPTIONS.keys())}, got {gain}")
        self._gain = gain
        if self.mode == "uart":
            self._at_cmd("ATGAIN", str(GAIN_OPTIONS[gain]))
        elif self.mode == "i2c":
            ctrl = self._i2c_read_virtual(_REG_CONTROL_SETUP)
            ctrl = (ctrl & 0xCF) | (GAIN_OPTIONS[gain] << 4)
            self._i2c_write_virtual(_REG_CONTROL_SETUP, ctrl)
        logger.info(f"Gain set to {gain}x")

    def set_integration_time(self, ms: int) -> None:
        """
        Set integration time in milliseconds.
        Valid range: 2.8 ms to 714 ms (register value = ms / 2.8).
        Longer integration = better SNR; slower scan rate.
        """
        if not (3 <= ms <= 714):
            raise ValueError(f"integration_time must be 3–714 ms, got {ms}")
        self._int_time = ms
        reg_val = max(1, min(255, int(ms / 2.8)))
        if self.mode == "uart":
            self._at_cmd("ATINTTIME", str(reg_val))
        elif self.mode == "i2c":
            self._i2c_write_virtual(_REG_INT_T, reg_val)
        logger.info(f"Integration time set to {ms} ms (register={reg_val})")

    def set_led(
        self,
        white: bool = True,
        uv: bool = False,
        ir: bool = False,
        led_current_ma: float = 12.5,
    ) -> None:
        """
        Enable/disable illumination LEDs across the three sensor chips:
          - white: 5700K broadband white LED on AS72652 (Visible)
          - uv:    365/405nm UV LED on AS72653 (UV/Blue)
          - ir:    875nm IR LED on AS72651 (NIR Master)
          - led_current_ma: 12.5, 25, 50, or 100 mA (12 mA supported for backward compatibility)
        """
        cur_val = float(led_current_ma)
        if cur_val not in _CURRENT_MAP:
            raise ValueError(f"led_current_ma must be one of [12.5, 25, 50, 100], got {led_current_ma}")
        cur_bits = _CURRENT_MAP[cur_val]

        if self.mode == "uart":
            # Per AMS OSRAM AS7265x AT Command specification (UG000389):
            # ATLED0 = IR (AS72651), ATLED1 = White (AS72652), ATLED2 = UV (AS72653)
            try:
                self._at_cmd("ATLED0", "1" if ir else "0")
                self._at_cmd("ATLED1", "1" if white else "0")
                self._at_cmd("ATLED2", "1" if uv else "0")
            except Exception:
                self._at_cmd("ATLED", "3" if (white or ir or uv) else "0")
        elif self.mode == "i2c":
            # In I2C: each chip has its own virtual register 0x07 (LED_CONFIG)
            # Register 0x07 bits: [5:4]=current limit, [3]=LED enable
            cfg_ir = (cur_bits << 4) | (int(ir) << 3)
            self._i2c_write_virtual(_REG_LED_CONFIG, cfg_ir)
            try:
                # Sub-device 1 (AS72652 - White)
                self._i2c_write_virtual(0x4F, 1)
                self._i2c_write_virtual(_REG_LED_CONFIG, (cur_bits << 4) | (int(white) << 3))
                # Sub-device 2 (AS72653 - UV)
                self._i2c_write_virtual(0x4F, 2)
                self._i2c_write_virtual(_REG_LED_CONFIG, (cur_bits << 4) | (int(uv) << 3))
                # Restore sub-device 0 (AS72651 - NIR)
                self._i2c_write_virtual(0x4F, 0)
            except Exception:
                cfg_composite = (cur_bits << 4) | (int(white) << 3) | (int(uv) << 2) | (int(ir) << 1)
                self._i2c_write_virtual(_REG_LED_CONFIG, cfg_composite)
        logger.info(f"LEDs: white={white}, uv={uv}, ir={ir}, current={led_current_ma}mA")

    def read_raw(self) -> np.ndarray:
        """
        Read 18 raw channel counts (ADC values, uint16).
        In mock mode: uses realistic sensor_simulator.py physical model.
        Returns np.ndarray of shape (18,), dtype float32.
        """
        if self.mode == "uart":
            return self._at_read_raw()
        elif self.mode == "i2c":
            return self._i2c_read_raw()
        elif self.mode == "mock":
            from sensor_simulator import simulate_scan
            sim = simulate_scan(
                X=MOCK_APPLE_REFLECTANCE,
                gain=self._gain / 16.0,
                read_noise_std=0.008,
                dark_offset=0.02,
                rng=np.random.default_rng(),
            )
            return (sim["raw"] * 30000.0).astype(np.float32)
        raise RuntimeError("Unknown mode")

    def read_calibrated(self) -> np.ndarray:
        """
        Read 18 calibrated reflectance values (dark/white corrected).
        In mock mode: uses realistic sensor_simulator.py physical model.
        Returns np.ndarray of shape (18,), dtype float32.
        """
        if self.mode == "uart":
            return self._at_read_calibrated()
        elif self.mode == "i2c":
            return self._i2c_read_calibrated()
        elif self.mode == "mock":
            from sensor_simulator import simulate_scan
            sim = simulate_scan(
                X=MOCK_APPLE_REFLECTANCE,
                gain=self._gain / 16.0,
                read_noise_std=0.008,
                dark_offset=0.02,
                rng=np.random.default_rng(),
            )
            return sim["calibrated"].astype(np.float32)
        raise RuntimeError("Unknown mode")

    def manual_calibrate(self, raw_scan: np.ndarray) -> np.ndarray:
        """
        Apply stored dark/white reference calibration manually:
          R_k = (S_k - D_k) / (W_k - D_k + ε)

        Must call measure_dark() and measure_white() before this.
        Delegated to src/calibration.py single source of truth.
        """
        if self._dark_ref is None or self._white_ref is None:
            raise RuntimeError(
                "Dark and white references not set. "
                "Call measure_dark() and measure_white() first."
            )
        try:
            from calibration import calibrate_reflectance
        except ImportError:
            from src.calibration import calibrate_reflectance
        return calibrate_reflectance(raw_scan, self._dark_ref, self._white_ref, epsilon=1e-6, clip_min=0.0)


    def measure_dark(self, n_readings: int = 10) -> np.ndarray:
        """
        Measure and store the dark reference vector.
        Cover the sensor window completely before calling.
        Returns mean dark vector (18,).
        """
        self.set_led(white=False, uv=False, ir=False)
        time.sleep(0.05 if self.mode != "mock" else 0)
        if self.mode == "mock":
            from sensor_simulator import simulate_scan
            rng = np.random.default_rng()
            darks = [
                (simulate_scan(X=np.zeros(18), dark_offset=0.02, read_noise_std=0.005, rng=rng)["raw"] * 30000.0)
                for _ in range(n_readings)
            ]
            self._dark_ref = np.mean(darks, axis=0).astype(np.float32)
        else:
            readings = np.stack([self.read_raw() for _ in range(n_readings)])
            self._dark_ref = readings.mean(axis=0)
        logger.info(f"Dark reference measured: mean={self._dark_ref.mean():.1f}")
        return self._dark_ref

    def measure_white(self, n_readings: int = 10) -> np.ndarray:
        """
        Measure and store the white reference vector (Teflon / Spectralon tile).
        Place white reference tile 5mm under sensor window before calling.
        Returns mean white vector (18,).
        """
        self.set_led(white=True, uv=False, ir=False, led_current_ma=12.5)
        time.sleep(0.1 if self.mode != "mock" else 0)
        if self.mode == "mock":
            from sensor_simulator import simulate_scan
            rng = np.random.default_rng()
            whites = [
                (simulate_scan(X=np.ones(18), dark_offset=0.02, read_noise_std=0.005, rng=rng)["raw"] * 30000.0)
                for _ in range(n_readings)
            ]
            self._white_ref = np.mean(whites, axis=0).astype(np.float32)
        else:
            readings = np.stack([self.read_raw() for _ in range(n_readings)])
            self._white_ref = readings.mean(axis=0)
        self.set_led(white=False, uv=False, ir=False)
        if self._dark_ref is not None:
            diff = self._white_ref - self._dark_ref
            if (diff <= 0).any():
                logger.warning("White reference has channels <= dark reference. Check tile placement.")
            signal = self._white_ref - self._dark_ref
            low_channels = np.where(signal < 100)[0]
            if len(low_channels) > 0:
                logger.warning(
                    f"White-Dark signal < 100 counts in channels: {low_channels.tolist()}. "
                    "Consider increasing LED brightness or integration time."
                )
        logger.info(f"White reference measured: mean={self._white_ref.mean():.1f}")
        return self._white_ref

    def read_with_illumination_sequence(
        self,
        sequence: Optional[List[str]] = None,
        settle_time_s: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """
        Cycle through multiple illumination modes and record calibrated spectra:
          - 'white': 5700K broadband white LED on AS72652
          - 'uv':    405nm UV LED on AS72653
          - 'ir':    875nm NIR LED on AS72651
          - 'dark':  LEDs OFF (ambient baseline)

        Returns dictionary mapping mode_name -> 18-channel calibrated reflectance.
        """
        seq = sequence or ["white", "uv", "ir", "dark"]
        results: Dict[str, np.ndarray] = {}

        for mode in seq:
            mode_lower = mode.lower().strip()
            if mode_lower == "white":
                self.set_led(white=True, uv=False, ir=False)
            elif mode_lower == "uv":
                self.set_led(white=False, uv=True, ir=False)
            elif mode_lower == "ir":
                self.set_led(white=False, uv=False, ir=True)
            elif mode_lower == "dark":
                self.set_led(white=False, uv=False, ir=False)
            else:
                raise ValueError(f"Unknown illumination mode: {mode!r}. Choose white, uv, ir, or dark.")

            if self.mode != "mock" and settle_time_s > 0:
                time.sleep(settle_time_s)

            results[mode_lower] = self.read_calibrated()

        # Turn off LEDs after acquisition
        self.set_led(white=False, uv=False, ir=False)
        return results

    def get_device_temperature(self) -> float:
        """
        Read the internal device temperature (°C) from AS72651 master chip.
        Useful for detecting LED thermal drift during long acquisition sessions.
        """
        if self.mode == "i2c":
            return float(self._i2c_read_virtual(_REG_DEVICE_TEMP))
        elif self.mode == "uart":
            resp = self._at_cmd("ATTEMP")
            for line in resp.splitlines():
                try:
                    return float(line.strip())
                except ValueError:
                    continue
        return float("nan")

    def close(self) -> None:
        """Release hardware resources."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
        logger.info("AS7265xDriver closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
