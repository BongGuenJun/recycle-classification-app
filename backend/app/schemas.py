from pydantic import BaseModel, Field


class Detection(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    class_id: int
    class_name: str
    material: str
    dirtiness: str
    confidence: float = Field(ge=0.0, le=1.0)
    detector_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    pipeline: str
    confidence_threshold: float = Field(ge=0.001, le=1.0)
    inference_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]


class PipelineComparison(BaseModel):
    pipeline: str
    confidence_threshold: float = Field(ge=0.001, le=1.0)
    inference_ms: float
    detections: list[Detection]


class CompareResponse(BaseModel):
    image_width: int
    image_height: int
    total_ms: float
    one_stage: PipelineComparison
    two_stage: PipelineComparison


class ModelHealth(BaseModel):
    loaded: bool
    classes: list[str]


class ModelsHealth(BaseModel):
    one_stage: ModelHealth
    two_stage: ModelHealth


class HealthResponse(BaseModel):
    status: str
    models: ModelsHealth
