# src/ui_pages/__init__.py
# Exposes all page render functions for clean import in app.py
from .page_overview    import render as page_overview
from .page_hsi_demo    import render as page_hsi_demo
from .page_sim_scan    import render as page_sim_scan
from .page_ml_evidence import render as page_ml_evidence
from .page_binary      import render as page_binary
from .page_hardware    import render as page_hardware
from .page_limits      import render as page_limits
from .page_roadmap     import render as page_roadmap
from .page_teacher     import render as page_teacher
from .page_virtual     import render as page_virtual
from .page_live_guided_scan import render_guided_scan_page as page_live_guided_scan

__all__ = [
    "page_overview", "page_hsi_demo", "page_sim_scan",
    "page_ml_evidence", "page_binary", "page_hardware",
    "page_limits", "page_roadmap", "page_teacher", "page_virtual",
    "page_live_guided_scan",
]
