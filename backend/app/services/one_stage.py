from __future__ import annotations

import threading
from pathlib import Path
from time import perf_counter

from PIL import Image

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "recycle_yolo_1stage_9class_bboxfixed.pt"
)


class OneStagePredictor:
    def __init__(
        self,
        model,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        max_detections: int = 10,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> "OneStagePredictor":
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        from ultralytics import YOLO

        return cls(YOLO(str(MODEL_PATH)))

    @property
    def class_names(self) -> list[str]:
        names = self.model.names
        if isinstance(names, dict):
            return [str(names[index]) for index in sorted(names)]
        return [str(name) for name in names]

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float | None = None,
    ) -> dict:
        started_at = perf_counter()
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )

        with self._lock:
            result = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=threshold,
                max_det=self.max_detections,
                verbose=False,
            )[0]

        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            class_ids = result.boxes.cls.detach().cpu().tolist()
            confidences = result.boxes.conf.detach().cpu().tolist()

            for bbox, class_id_value, confidence in zip(
                boxes, class_ids, confidences
            ):
                class_id = int(class_id_value)
                class_name = self.class_names[class_id]
                material, dirtiness = self._split_class_name(class_name)

                detections.append(
                    {
                        "bbox": [round(float(value), 2) for value in bbox],
                        "class_id": class_id,
                        "class_name": class_name,
                        "material": material,
                        "dirtiness": dirtiness,
                        "confidence": round(float(confidence), 4),
                    }
                )

        inference_ms = (perf_counter() - started_at) * 1000
        return {
            "inference_ms": round(inference_ms, 2),
            "detections": detections,
        }

    @staticmethod
    def _split_class_name(class_name: str) -> tuple[str, str]:
        parts = class_name.split("_", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Expected '<material>_<dirtiness>', got: {class_name}"
            )
        return parts[0], parts[1]
