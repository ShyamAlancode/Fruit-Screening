"""
api.py  —  CFRD FastAPI inference service
==========================================
Wraps the ApplePredictor in a production-ready REST API.

Endpoints:
    GET  /              -> health check
    GET  /health        -> health check with model status
    POST /predict       -> 18-channel spectrum -> full prediction JSON
    POST /predict/batch -> list of spectra -> list of predictions
    GET  /docs          -> Swagger UI (auto-generated)
    GET  /redoc         -> ReDoc UI (auto-generated)

Run locally:
    pip install fastapi uvicorn
    uvicorn src.api:app --reload --port 8000

Run in Docker:
    docker build -t cfrd-api .
    docker run -p 8000:8000 cfrd-api

Resume note: shows you can ship ML, not just train it.
"""

from __future__ import annotations
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# FastAPI — graceful import
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define stubs so the module imports without FastAPI
    class BaseModel:
        pass
    def Field(*a, **kw):
        return None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

logger = logging.getLogger("cfrd.api")

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumRequest(BaseModel):
    """18-channel AS7265x reflectance spectrum (410-940 nm)."""
    spectrum: List[float] = Field(
        ...,
        min_items=18,
        max_items=18,
        description="18 calibrated reflectance values [ch_A .. ch_W], range [0.0, 1.5]",
        example=[0.09, 0.10, 0.11, 0.14, 0.22, 0.38,
                 0.44, 0.42, 0.40, 0.35, 0.22, 0.48,
                 0.72, 0.76, 0.79, 0.81, 0.82, 0.83],
    )
    model: str = Field(
        "synthetic",
        description="Pesticide model variant: 'synthetic' (240-row) or 'real' (26-row Roomi)",
    )


class BatchSpectrumRequest(BaseModel):
    """Batch of spectra for bulk prediction."""
    spectra: List[List[float]] = Field(
        ...,
        description="List of 18-channel spectra",
    )
    model: str = Field("synthetic", description="Pesticide model variant")


class PesticideResult(BaseModel):
    label: str
    confidence: float
    ood_status: str
    probabilities: Dict[str, float]
    scientific_disclaimer: str


class RipenessResult(BaseModel):
    stage: str
    confidence: float
    probabilities: Dict[str, float]


class VarietyResult(BaseModel):
    cultivar: str
    confidence: float
    probabilities: Dict[str, float]


class BrixResult(BaseModel):
    estimated_brix: float
    sweetness_tier: str
    r2_authoritative: float
    provenance_status: str


class PredictionResponse(BaseModel):
    """Full apple prediction with scientific metadata."""
    request_id: str
    latency_ms: float
    pesticide: PesticideResult
    ripeness: RipenessResult
    variety: VarietyResult
    brix: BrixResult
    # Required disclaimers — always present
    disclaimer: str = (
        "CFRD is a pre-hardware research prototype. "
        "No pesticide detection claim is scientifically validated. "
        "Do not use for food safety decisions."
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    sklearn_version: str
    torch_available: bool
    physical_sensor: str
    uptime_s: float


# ─────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ─────────────────────────────────────────────────────────────────────────────

if not FASTAPI_AVAILABLE:
    raise ImportError(
        "FastAPI not installed. Run: pip install fastapi uvicorn\n"
        "Or: pip install -e '.[api]'"
    )

app = FastAPI(
    title="CFRD — Compact Food Reflectance Device API",
    description=(
        "REST API for apple quality screening using 18-channel multispectral reflectance.\n\n"
        "**Status:** Pre-hardware research prototype. All predictions are from simulation-trained models.\n\n"
        "**Primary result:** LOAO scan-level accuracy = 75% (sensitivity 100%, specificity 30%, AUROC 0.59, p≈0.19).\n\n"
        "**Do not use for food safety decisions.**"
    ),
    version="0.1.0-pre-hardware",
    contact={"name": "CFRD Project", "url": "https://github.com/cfrd-project"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "prediction", "description": "Spectral inference endpoints"},
        {"name": "health", "description": "Service health and model status"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
_START_TIME = time.time()

# Lazy-loaded predictor (loaded on first request, not at import time)
_predictor_cache: Dict[str, Any] = {}


def get_predictor(model: str = "synthetic"):
    """Get or create a cached predictor instance."""
    if model not in _predictor_cache:
        from predictor import ApplePredictor
        _predictor_cache[model] = ApplePredictor(default_pesticide_model=model)
    return _predictor_cache[model]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "service": "CFRD Apple Screening API",
        "version": "0.1.0-pre-hardware",
        "status": "ok",
        "docs": "/docs",
        "disclaimer": "Pre-hardware research prototype. Not for food safety decisions.",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Check service health and model loading status."""
    import sklearn
    try:
        import torch
        torch_ok = True
    except ImportError:
        torch_ok = False

    # Check which models resolve from the manifest
    from model_registry import get_model_path
    model_keys = ["treatment_3class_synthetic", "treatment_3class_real",
                  "ripeness_classifier", "variety_classifier"]
    models_loaded = {k: (get_model_path(k) is not None and get_model_path(k).exists())
                     for k in model_keys}

    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        sklearn_version=sklearn.__version__,
        torch_available=torch_ok,
        physical_sensor="NOT_CONNECTED — 0 physical scans; all data is simulation",
        uptime_s=round(time.time() - _START_TIME, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(req: SpectrumRequest):
    """
    Predict pesticide treatment, ripeness, cultivar, and Brix from
    an 18-channel AS7265x reflectance spectrum.

    Input: 18 calibrated reflectance values [ch_A..ch_W] in range [0.0, 1.5]
    Output: Structured prediction with scientific disclaimers
    """
    if len(req.spectrum) != 18:
        raise HTTPException(status_code=422, detail="Exactly 18 spectral channels required.")

    t0 = time.time()
    try:
        predictor = get_predictor(req.model)
        result = predictor.predict_spectrum(req.spectrum)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    latency_ms = (time.time() - t0) * 1000
    import uuid

    return PredictionResponse(
        request_id=str(uuid.uuid4())[:8],
        latency_ms=round(latency_ms, 2),
        pesticide=PesticideResult(
            label=result.pesticide.label,
            confidence=round(result.pesticide.confidence, 4),
            ood_status=result.pesticide.ood_status,
            probabilities={k: round(v, 4) for k, v in result.pesticide.probabilities.items()},
            scientific_disclaimer=(
                "LOAO scan-level accuracy 75% (simulation only). "
                "AUROC=0.59, p≈0.19 (not statistically significant). "
                "Held-out test Macro-F1=35-42%. Not validated on physical hardware."
            ),
        ),
        ripeness=RipenessResult(
            stage=result.ripeness.stage,
            confidence=round(result.ripeness.confidence, 4),
            probabilities={k: round(v, 4) for k, v in result.ripeness.probabilities.items()},
        ),
        variety=VarietyResult(
            cultivar=result.variety.cultivar,
            confidence=round(result.variety.confidence, 4),
            probabilities={k: round(v, 4) for k, v in result.variety.probabilities.items()},
        ),
        brix=BrixResult(
            estimated_brix=round(result.brix.estimated_brix, 2),
            sweetness_tier=result.brix.sweetness_tier,
            r2_authoritative=result.brix.r2_verified,
            provenance_status="UNDER_RECONCILIATION",
        ),
    )


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(req: BatchSpectrumRequest):
    """
    Batch prediction for multiple spectra.
    Returns a list of PredictionResponse objects.
    """
    if not req.spectra:
        raise HTTPException(status_code=422, detail="At least one spectrum required.")
    if len(req.spectra) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 spectra per batch.")

    predictor = get_predictor(req.model)
    results = []
    for i, spectrum in enumerate(req.spectra):
        if len(spectrum) != 18:
            raise HTTPException(
                status_code=422,
                detail=f"Spectrum {i}: expected 18 channels, got {len(spectrum)}"
            )
        try:
            r = predictor.predict_spectrum(spectrum)
            results.append({
                "index": i,
                "pesticide_label": r.pesticide.label,
                "pesticide_confidence": round(r.pesticide.confidence, 4),
                "ood_status": r.pesticide.ood_status,
                "ripeness": r.ripeness.stage,
                "cultivar": r.variety.cultivar,
                "brix_estimate": round(r.brix.estimated_brix, 2),
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {
        "n_spectra": len(req.spectra),
        "n_errors": sum(1 for r in results if "error" in r),
        "disclaimer": "Pre-hardware research prototype. Not for food safety decisions.",
        "results": results,
    }
