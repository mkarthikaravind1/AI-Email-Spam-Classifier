import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from classifiers.ensemble import combine, load_weights
from classifiers.llm_classifier import predict_llm
from classifiers.regex_classifier import predict_regex
from classifiers.statistical_ml_classifier import load_model, predict_ml

logger = logging.getLogger("spam_classifier")

# Fail at startup with a clear message if the ML model hasn't been trained yet
load_model()
WEIGHTS = load_weights()

# Max seconds to wait for all classifiers; a model that is slower is reported
# as an error and the verdict is made from the models that finished.
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))

# Only the frontend's own origin(s) may call the API. Override with a
# comma-separated list, e.g. ALLOWED_ORIGINS=https://myapp.example.com
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

CLASSIFIERS = {
    "regex": predict_regex,
    "ml": predict_ml,
    "llm": predict_llm,
}

app = FastAPI(title="AI Email Spam Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

executor = ThreadPoolExecutor(max_workers=len(CLASSIFIERS))


class EmailRequest(BaseModel):
    email_text: str

    @field_validator("email_text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("email_text cannot be empty")
        if len(v) > 5000:
            raise ValueError("email_text exceeds 5000 character limit")
        return v


@app.get("/")
def home():
    return {"message": "Backend running", "weights": WEIGHTS}


@app.post("/analyze")
def analyze_email(request: EmailRequest):
    start_time = time.perf_counter()
    deadline = start_time + MODEL_TIMEOUT_SECONDS

    # All three classifiers run in parallel
    futures = {name: executor.submit(fn, request.email_text) for name, fn in CLASSIFIERS.items()}

    results = {}
    for name, future in futures.items():
        try:
            remaining = max(0.0, deadline - time.perf_counter())
            results[name] = {"status": "ok", **future.result(timeout=remaining)}
        except FutureTimeoutError:
            results[name] = {"status": "error", "error": f"timed out after {MODEL_TIMEOUT_SECONDS:g}s"}
        except Exception as exc:  # one broken model must not take down the request
            logger.exception("%s classifier failed", name)
            results[name] = {"status": "error", "error": str(exc) or type(exc).__name__}

    verdict = combine(results, WEIGHTS)
    if verdict is None:
        raise HTTPException(status_code=503, detail="All classifiers failed; check the backend logs.")

    return {
        **results,
        **verdict,
        "processing_time": round(time.perf_counter() - start_time, 2),
    }
