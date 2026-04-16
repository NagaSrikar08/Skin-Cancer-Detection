import io
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from model import build_model


MODEL_PATH = os.getenv("MODEL_PATH", "models/densenet_skin_cancer.pth")
TRAIN_ON_STARTUP = os.getenv("TRAIN_ON_STARTUP", "false").lower() == "true"
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "Train.py")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CLASS_NAMES = ["non-cancer", "cancer"]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    image_size: int
    device: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    class_names: List[str]


def run_training_if_enabled() -> None:
    if not TRAIN_ON_STARTUP:
        return

    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(
            f"Training script not found at '{TRAIN_SCRIPT}'."
        )

    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Training failed on startup.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.class_names = DEFAULT_CLASS_NAMES
        self.image_size = DEFAULT_IMAGE_SIZE
        self.model_path = MODEL_PATH
        self.transform = None

    def load(self) -> None:
        model, device = build_model()
        self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Set MODEL_PATH or place the trained .pth file in models/."
            )

        checkpoint = torch.load(self.model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
            self.image_size = checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)

        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            self.class_names = DEFAULT_CLASS_NAMES
            self.image_size = DEFAULT_IMAGE_SIZE

        else:
            raise ValueError("Unsupported model file format.")

        model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.model = model

    def predict_pil_image(self, image: Image.Image) -> PredictionResponse:
        if self.model is None or self.transform is None:
            raise RuntimeError("Model is not loaded.")

        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        probabilities = {
            self.class_names[i]: float(probs[i].item())
            for i in range(len(self.class_names))
        }

        return PredictionResponse(
            predicted_class=self.class_names[pred_idx],
            confidence=float(probs[pred_idx].item()),
            probabilities=probabilities,
            image_size=self.image_size,
            device=self.device,
        )


model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_training_if_enabled()
    model_service.load()
    yield


app = FastAPI(
    title="Skin Cancer Detection API",
    description=(
        "FastAPI backend for DenseNet-121 binary skin lesion classification. "
        "This is a research and educational prototype, not a diagnostic device."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.model is not None,
        model_path=model_service.model_path,
        device=model_service.device,
        class_names=model_service.class_names,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.model is not None,
        model_path=model_service.model_path,
        device=model_service.device,
        class_names=model_service.class_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use .jpg, .jpeg, .png, or .webp.",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image = Image.open(io.BytesIO(file_bytes))
        result = model_service.predict_pil_image(image)
        return result

    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(_, exc: FileNotFoundError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Run locally with:
# uvicorn skin_cancer_backend_main:app --host 0.0.0.0 --port 8000
