FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/app/.matplotlib \
    PORT=8002 \
    WEB_CONCURRENCY=1 \
    GUNICORN_TIMEOUT=120

# libgl1/libglib2.0-0 are the runtime libraries opencv-python links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1 torchvision==0.27.1 \
    && pip install -r requirements.txt

COPY models/ ./models/
COPY detect_api.py .

# Loading the checkpoint and running one inference at build time fails the build
# on a bad model file and bakes the ultralytics cache into the image, so the
# first real request is not paying for it.
RUN mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" \
    && python -c "from ultralytics import settings; settings.update({'sync': False})" \
    && python -c "\
import numpy as np;\
from ultralytics import YOLO;\
m = YOLO('models/best.pt');\
r = m.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)[0];\
r.plot();\
print('warmup ok, classes:', len(m.names))"

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8002') + '/health', timeout=8)"]

# Threads stay at 1: an ultralytics model object is not safe to call concurrently,
# so scale with WEB_CONCURRENCY (one model per worker process) instead.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --threads 1 --timeout ${GUNICORN_TIMEOUT} --graceful-timeout 30 --access-logfile - --error-logfile - detect_api:app"]
