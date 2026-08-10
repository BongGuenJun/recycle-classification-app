from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakePredictor:
    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.class_names = [
            "can_clean",
            "can_outer",
            "can_inner",
            "pet_clean",
            "pet_outer",
            "pet_inner",
            "plastic_clean",
            "plastic_outer",
            "plastic_inner",
        ]

    def predict(self, image, confidence_threshold=None):
        is_two_stage = self.pipeline == "two_stage"
        return {
            "inference_ms": 20.0 if is_two_stage else 12.34,
            "detections": [
                {
                    "bbox": [10.0, 20.0, 100.0, 160.0],
                    "class_id": 4,
                    "class_name": "pet_outer",
                    "material": "pet",
                    "dirtiness": "outer",
                    "confidence": 0.72 if is_two_stage else 0.82,
                    "detector_confidence": 0.90 if is_two_stage else 0.82,
                    "classifier_confidence": 0.80 if is_two_stage else None,
                }
            ],
        }


def make_app():
    return create_app(FakePredictor("one_stage"), FakePredictor("two_stage"))


def make_jpeg(width=320, height=240):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_both_loaded_models():
    with TestClient(make_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["models"]["one_stage"]["loaded"] is True
    assert response.json()["models"]["two_stage"]["loaded"] is True
    assert len(response.json()["models"]["one_stage"]["classes"]) == 9
    assert len(response.json()["models"]["two_stage"]["classes"]) == 9


def test_predict_defaults_to_one_stage():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "one_stage"
    assert body["confidence_threshold"] == 0.25
    assert body["image_width"] == 320
    assert body["detections"][0]["classifier_confidence"] is None


def test_predict_can_select_two_stage():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict?pipeline=two_stage&confidence_threshold=0.15",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "two_stage"
    assert body["confidence_threshold"] == 0.15
    assert body["detections"][0]["detector_confidence"] == 0.9
    assert body["detections"][0]["classifier_confidence"] == 0.8


def test_compare_returns_both_pipeline_results():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict/compare?one_stage_threshold=0.25&two_stage_threshold=0.20",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["one_stage"]["pipeline"] == "one_stage"
    assert body["two_stage"]["pipeline"] == "two_stage"
    assert body["one_stage"]["confidence_threshold"] == 0.25
    assert body["two_stage"]["confidence_threshold"] == 0.20
    assert body["total_ms"] >= 0


def test_predict_rejects_unknown_pipeline():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict?pipeline=unknown",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 422


def test_predict_rejects_unsupported_file_type():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict",
            files={"image": ("sample.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415


def test_visualize_can_select_two_stage():
    with TestClient(make_app()) as client:
        response = client.post(
            "/predict/visualize?pipeline=two_stage&confidence_threshold=0.15",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-pipeline"] == "two_stage"
    assert response.headers["x-detection-count"] == "1"
    assert Image.open(BytesIO(response.content)).size == (320, 240)
