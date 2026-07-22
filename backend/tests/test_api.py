from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakePredictor:
    class_names = [
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
        return {
            "inference_ms": 12.34,
            "detections": [
                {
                    "bbox": [10.0, 20.0, 100.0, 160.0],
                    "class_id": 4,
                    "class_name": "pet_outer",
                    "material": "pet",
                    "dirtiness": "outer",
                    "confidence": 0.82,
                }
            ],
        }


def make_jpeg(width=320, height=240):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_loaded_model():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["models"]["one_stage"]["loaded"] is True
    assert len(response.json()["models"]["one_stage"]["classes"]) == 9


def test_predict_returns_detection_and_image_size():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/predict",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "one_stage"
    assert body["confidence_threshold"] == 0.25
    assert body["image_width"] == 320
    assert body["image_height"] == 240
    assert body["detections"][0]["class_name"] == "pet_outer"


def test_predict_accepts_custom_confidence_threshold():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/predict?confidence_threshold=0.05",
            files={"image": ("sample.jpg", make_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["confidence_threshold"] == 0.05


def test_predict_rejects_unsupported_file_type():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/predict",
            files={"image": ("sample.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
