from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from preprocess import load_and_preprocess
from model import get_models, predict_age
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_models()   # load all 3 at startup
    yield

app = FastAPI(title="Brain Age Prediction API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED = {".nii", ".nii.gz"}
MAX_MB  = 200

class PredictionResult(BaseModel):
    predicted_age: float
    filename: str
    processing_time_s: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    name = file.filename or ""
    if not any(name.lower().endswith(ext) for ext in ALLOWED):
        raise HTTPException(400, "Only .nii and .nii.gz files are supported.")

    contents = await file.read()
    if len(contents) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB}MB limit.")

    t0 = time.time()
    try:
        volume = load_and_preprocess(contents, name)
        age    = predict_age(volume)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    age = max(0.0, min(age, 100.0))
    return PredictionResult(
        predicted_age=round(age, 1),
        filename=name,
        processing_time_s=round(time.time() - t0, 2),
    )