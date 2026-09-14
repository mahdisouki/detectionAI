import os
import tempfile
from io import BytesIO

import cv2
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt")
)
CONF = float(os.environ.get("DETECT_CONF", "0.30"))
IOU = float(os.environ.get("DETECT_IOU", "0.45"))
IMGSZ = int(os.environ.get("DETECT_IMGSZ", "640"))
PORT = int(os.environ.get("PORT", "8002"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

model = YOLO(MODEL_PATH)
model.overrides["conf"] = CONF
model.overrides["iou"] = IOU
model.overrides["imgsz"] = IMGSZ


def _box_to_detection(box, names):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    class_id = int(box.cls[0])
    return {
        "class": names[class_id],
        "class_id": class_id,
        "confidence": round(float(box.conf[0]), 4),
        "bbox": {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        },
    }


def _encode_annotated_image(results):
    annotated_bgr = results.plot()
    ok, buffer = cv2.imencode(".jpg", annotated_bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    return buffer.tobytes()


def _apply_conf_filter(results):
    """Drop boxes below CONF. Ultralytics can keep default-threshold (0.25) boxes."""
    if results.boxes is None or len(results.boxes) == 0:
        return results
    results.boxes = results.boxes[results.boxes.conf >= CONF]
    return results


def _run_detection(path):
    results = model.predict(
        path,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        verbose=False,
    )[0]
    results = _apply_conf_filter(results)

    detections = [
        _box_to_detection(box, results.names) for box in results.boxes
    ]
    return results, detections


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        path = tmp.name

    try:
        _, detections = _run_detection(path)
        return jsonify({
            "detections": detections,
            "count": len(detections),
            "conf": CONF,
        })
    finally:
        os.unlink(path)


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        path = tmp.name

    try:
        results, _ = _run_detection(path)
        image_bytes = _encode_annotated_image(results)

        return send_file(
            BytesIO(image_bytes),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="detected.jpg",
        )
    finally:
        os.unlink(path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "conf": CONF,
        "iou": IOU,
        "imgsz": IMGSZ,
        "classes": len(model.names),
        "class_names": list(model.names.values()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
