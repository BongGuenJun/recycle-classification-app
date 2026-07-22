from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas import HealthResponse, PredictResponse
from app.services.one_stage import OneStagePredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def create_app(predictor: OneStagePredictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.one_stage_predictor = predictor or OneStagePredictor.load()
        yield

    app = FastAPI(
        title="Recycle Classification API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", tags=["system"])
    def home():
        return {"message": "Recycle AI Server Running"}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        loaded_predictor = app.state.one_stage_predictor
        return {
            "status": "ok",
            "models": {
                "one_stage": {
                    "loaded": True,
                    "classes": loaded_predictor.class_names,
                }
            },
        }

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(
        image: UploadFile = File(...),
        pipeline: Literal["one_stage"] = "one_stage",
        confidence_threshold: float = Query(0.25, ge=0.001, le=1.0),
    ):
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Only JPEG, PNG, and WebP images are supported.",
            )

        contents = await image.read(MAX_IMAGE_BYTES + 1)
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        try:
            pil_image = Image.open(BytesIO(contents))
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            pil_image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image file.") from exc

        loaded_predictor = app.state.one_stage_predictor
        result = await run_in_threadpool(
            loaded_predictor.predict,
            pil_image,
            confidence_threshold,
        )

        return {
            "pipeline": pipeline,
            "confidence_threshold": confidence_threshold,
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            **result,
        }

    return app


app = create_app()
