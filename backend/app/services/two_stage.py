from __future__ import annotations

import math
import threading
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DETECTOR_PATH = MODELS_DIR / "yolo_2stage_material3_bboxfixed.pt"
CLASSIFIER_PATH = (
    MODELS_DIR / "classifier_resnet18_dirty3_bboxfixed_pad005.pt"
)

MATERIAL_NAMES = ["can", "pet", "plastic"]
DIRTINESS_NAMES = ["clean", "outer", "inner"]
FINAL_CLASS_NAMES = [
    f"{material}_{dirtiness}"
    for material in MATERIAL_NAMES
    for dirtiness in DIRTINESS_NAMES
]


class TwoStagePredictor:
    def __init__(
        self,
        detector,
        classifier,
        device: torch.device,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        max_detections: int = 10,
        crop_padding: float = 0.05,
    ):
        self.detector = detector
        self.classifier = classifier
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self.crop_padding = crop_padding
        self._lock = threading.Lock()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @classmethod
    def load(cls) -> "TwoStagePredictor":
        for model_path in (DETECTOR_PATH, CLASSIFIER_PATH):
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        from ultralytics import YOLO

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        detector = YOLO(str(DETECTOR_PATH))
        detector_names = detector.names
        if isinstance(detector_names, dict):
            names = [str(detector_names[index]) for index in sorted(detector_names)]
        else:
            names = [str(name) for name in detector_names]
        if names != MATERIAL_NAMES:
            raise ValueError(
                f"2-stage detector class order mismatch: {names}"
            )

        classifier = models.resnet18(weights=None)
        classifier.fc = nn.Linear(classifier.fc.in_features, 3)
        state_dict = torch.load(
            CLASSIFIER_PATH,
            map_location=device,
            weights_only=True,
        )
        classifier.load_state_dict(state_dict, strict=True)
        classifier.to(device)
        classifier.eval()

        return cls(detector, classifier, device)

    @property
    def class_names(self) -> list[str]:
        return FINAL_CLASS_NAMES.copy()

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
            result = self.detector.predict(
                source=image,
                imgsz=self.image_size,
                conf=threshold,
                max_det=self.max_detections,
                verbose=False,
            )[0]

            valid_items = []
            crop_tensors = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                material_ids = result.boxes.cls.detach().cpu().tolist()
                detector_confidences = result.boxes.conf.detach().cpu().tolist()

                for bbox, material_value, detector_confidence in zip(
                    boxes, material_ids, detector_confidences
                ):
                    crop = self._crop_with_padding(image, bbox)
                    if crop is None:
                        continue
                    valid_items.append(
                        (bbox, int(material_value), float(detector_confidence))
                    )
                    crop_tensors.append(self.transform(crop))

            detections = []
            if crop_tensors:
                batch = torch.stack(crop_tensors).to(self.device)
                with torch.inference_mode():
                    probabilities = torch.softmax(self.classifier(batch), dim=1)
                dirty_confidences, dirty_ids = probabilities.max(dim=1)

                for index, (bbox, material_id, detector_confidence) in enumerate(
                    valid_items
                ):
                    dirty_id = int(dirty_ids[index].item())
                    classifier_confidence = float(dirty_confidences[index].item())
                    final_class_id = material_id * 3 + dirty_id
                    final_confidence = detector_confidence * classifier_confidence
                    detections.append(
                        {
                            "bbox": [round(float(value), 2) for value in bbox],
                            "class_id": final_class_id,
                            "class_name": FINAL_CLASS_NAMES[final_class_id],
                            "material": MATERIAL_NAMES[material_id],
                            "dirtiness": DIRTINESS_NAMES[dirty_id],
                            "confidence": round(final_confidence, 4),
                            "detector_confidence": round(detector_confidence, 4),
                            "classifier_confidence": round(
                                classifier_confidence, 4
                            ),
                        }
                    )

        inference_ms = (perf_counter() - started_at) * 1000
        return {
            "inference_ms": round(inference_ms, 2),
            "detections": detections,
        }

    def _crop_with_padding(
        self,
        image: Image.Image,
        bbox: list[float],
    ) -> Image.Image | None:
        width, height = image.size
        x1, y1, x2, y2 = map(float, bbox)
        box_width = x2 - x1
        box_height = y2 - y1

        x1 = max(0, math.floor(x1 - box_width * self.crop_padding))
        y1 = max(0, math.floor(y1 - box_height * self.crop_padding))
        x2 = min(width, math.ceil(x2 + box_width * self.crop_padding))
        y2 = min(height, math.ceil(y2 + box_height * self.crop_padding))

        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))
