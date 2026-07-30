# detectionAI

YOLOv8 object detection service exposed over a small Flask HTTP API. Detects 43
household and garden item classes (appliances, furniture, garden equipment,
electronics — see `data.yaml` for the full list).

## API

| Method | Route | Returns |
| --- | --- | --- |
| `POST` | `/detect` | JSON detections, plus a base64 annotated image unless `?annotate=false` |
| `POST` | `/detect/image` | The annotated JPEG itself |
| `GET` | `/health` | Service status, model path, and loaded class names |

Both detect routes take a multipart form upload under the field name `image`.

```bash
curl -F "image=@photo.jpg" http://localhost:8002/detect
curl -F "image=@photo.jpg" http://localhost:8002/detect/image --output detected.jpg
```

A detection looks like this:

```json
{
  "class": "Microwave",
  "class_id": 26,
  "confidence": 0.9134,
  "bbox": { "x1": 214.5, "y1": 88.31, "x2": 476.02, "y2": 265.77 }
}
```

## Configuration

Everything is set through environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/best.pt` | Weights to load |
| `PORT` | `8002` | Listen port |
| `DETECT_CONF` | `0.50` | Confidence threshold |
| `DETECT_IOU` | `0.45` | NMS IoU threshold |
| `DETECT_IMGSZ` | `640` | Inference image size |
| `MAX_UPLOAD_MB` | `25` | Rejects larger uploads |
| `WEB_CONCURRENCY` | `1` | Gunicorn workers |

## Running it

With Docker:

```bash
docker compose up --build
```

Locally, install CPU-only torch first so pip doesn't pull the ~2.5 GB CUDA build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision==0.27.1
pip install -r requirements.txt
python detect_api.py
```

## Deploying to Dokploy

Create an **Application**, point it at this repo, set the build type to
**Dockerfile**, and set the port to **8002**. Override any of the variables
above in the environment tab.

The image runs a warmup inference at build time, so a bad or version-mismatched
checkpoint fails the build instead of surfacing as a runtime error on the first
request. `GET /health` after deploy confirms which weights were baked in.

Scale with `WEB_CONCURRENCY` rather than threads — an ultralytics model object
is not safe to call concurrently, so each worker loads its own copy at roughly
1 GB of RSS. Size the host accordingly.
