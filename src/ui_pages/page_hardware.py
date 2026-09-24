"""page_hardware.py — Page 6: Hardware Arrival Plan"""
import streamlit as st
from ui_theme import show_disclaimer, warning_box


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    st.title("🔧 Hardware Arrival Plan")
    st.subheader("AS7265x Calibration and Data Collection Protocol")

    st.markdown("""
    ### When hardware arrives — exact steps

    #### Step 1: Physical setup and wiring check
    - Connect AS7265x to Raspberry Pi via I²C (addr 0x49 master)
    - Verify 3.3V power rail stability
    - Confirm SDA/SCL pull-ups (4.7kΩ to 3.3V)
    - Read firmware version via `get_firmware_version()` command

    #### Step 2: Dark calibration
    - Cover sensor window completely (light-tight cap or black tape)
    - Collect 10 consecutive dark readings at target integration time
    - Store mean dark vector `D` (18 values)
    - Assert: all D values < 200 raw counts (for 16-bit ADC)

    #### Step 3: White reference calibration
    - Place Spectralon or Teflon tile 5mm under sensor window
    - Collect 10 consecutive white readings at same integration time
    - Store mean white vector `W` (18 values)
    - Assert: W > D for all channels (fails → hardware issue)
    - Assert: W − D > 100 counts per channel (too dim → increase LED brightness or integration time)

    #### Step 4: UV405 crosstalk test
    """)

    warning_box(
        "⚠️ <strong>Known risk:</strong> The AS72653 UV chip (410nm channel) may receive optical "
        "crosstalk from the 405nm LED used by the NIR chip. Test by scanning: (a) white tile, "
        "(b) black card, (c) white tile again. If channel ch_410 changes by >5% between (a) and (b) "
        "with only UV LED active, apply a per-session correction factor before classification."
    )

    st.markdown("""
    #### Step 5: Real apple data collection
    - Minimum 20 physical apples per treatment class
    - 3 repeated scans per apple (repositioning between scans)
    - Assign fruit_id before first scan — never reassign
    - Record: scan timestamp, integration time, ambient temperature
    - Freeze 20% of fruits as blind test set from day one

    #### Step 6: Retrain from scratch
    - Do NOT mix Roomi synthetic rows with real AS7265x rows
    - Apply StandardScaler → PCA → LDA (same pipeline) to real data
    - GroupKFold by real fruit_id
    - Report real device performance separately from synthetic benchmark
    """)
