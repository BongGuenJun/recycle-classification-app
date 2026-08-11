from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from app.schemas import CompareResponse, HealthResponse, PredictResponse
from app.services.one_stage import OneStagePredictor
from app.services.two_stage import TwoStagePredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
PipelineName = Literal["one_stage", "two_stage"]


async def read_uploaded_image(image: UploadFile) -> Image.Image:
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
        return pil_image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(min(annotated.size) / 300))
    text_padding = max(4, line_width)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'

        draw.rectangle((x1, y1, x2, y2), outline="lime", width=line_width)
        text_box = draw.textbbox((x1, y1), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - text_height - (text_padding * 2))
        label_right = min(annotated.width, x1 + text_width + (text_padding * 2))

        draw.rectangle((x1, label_top, label_right, y1), fill="lime")
        draw.text(
            (x1 + text_padding, label_top + text_padding),
            label,
            fill="black",
        )

    return annotated


def create_app(
    one_stage_predictor: OneStagePredictor | None = None,
    two_stage_predictor: TwoStagePredictor | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.one_stage_predictor = (
            one_stage_predictor or OneStagePredictor.load()
        )
        app.state.two_stage_predictor = (
            two_stage_predictor or TwoStagePredictor.load()
        )
        yield

    app = FastAPI(
        title="Recycle Classification API",
        version="0.2.0",
        lifespan=lifespan,
    )

    def get_predictor(pipeline: PipelineName):
        if pipeline == "one_stage":
            return app.state.one_stage_predictor
        return app.state.two_stage_predictor

    @app.get("/", tags=["system"])
    def home():
        return {"message": "Recycle AI Server Running"}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        one_stage = app.state.one_stage_predictor
        two_stage = app.state.two_stage_predictor
        return {
            "status": "ok",
            "models": {
                "one_stage": {
                    "loaded": True,
                    "classes": one_stage.class_names,
                },
                "two_stage": {
                    "loaded": True,
                    "classes": two_stage.class_names,
                },
            },
        }

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(
        image: UploadFile = File(...),
        pipeline: PipelineName = "one_stage",
        confidence_threshold: float = Query(0.25, ge=0.001, le=1.0),
    ):
        pil_image = await read_uploaded_image(image)
        predictor = get_predictor(pipeline)
        result = await run_in_threadpool(
            predictor.predict,
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

    @app.post(
        "/predict/compare",
        response_model=CompareResponse,
        tags=["inference"],
    )
    async def compare_predictions(
        image: UploadFile = File(...),
        one_stage_threshold: float = Query(0.25, ge=0.001, le=1.0),
        two_stage_threshold: float = Query(0.25, ge=0.001, le=1.0),
    ):
        pil_image = await read_uploaded_image(image)
        started_at = perf_counter()

        # 순차 실행으로 동일 장치에서의 메모리 경합과 시간 왜곡을 피한다.
        one_result = await run_in_threadpool(
            app.state.one_stage_predictor.predict,
            pil_image,
            one_stage_threshold,
        )
        two_result = await run_in_threadpool(
            app.state.two_stage_predictor.predict,
            pil_image,
            two_stage_threshold,
        )
        total_ms = (perf_counter() - started_at) * 1000

        return {
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            "total_ms": round(total_ms, 2),
            "one_stage": {
                "pipeline": "one_stage",
                "confidence_threshold": one_stage_threshold,
                **one_result,
            },
            "two_stage": {
                "pipeline": "two_stage",
                "confidence_threshold": two_stage_threshold,
                **two_result,
            },
        }

    @app.post(
        "/predict/visualize",
        tags=["inference"],
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def visualize_prediction(
        image: UploadFile = File(...),
        pipeline: PipelineName = "one_stage",
        confidence_threshold: float = Query(0.25, ge=0.001, le=1.0),
    ):
        pil_image = await read_uploaded_image(image)
        predictor = get_predictor(pipeline)
        result = await run_in_threadpool(
            predictor.predict,
            pil_image,
            confidence_threshold,
        )
        annotated = draw_detections(pil_image, result["detections"])

        output = BytesIO()
        annotated.save(output, format="JPEG", quality=90)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": 'inline; filename="prediction.jpg"',
                "X-Pipeline": pipeline,
                "X-Detection-Count": str(len(result["detections"])),
            },
        )

    return app


app = create_app()
