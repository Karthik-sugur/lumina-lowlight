FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GRADIO_ANALYTICS_ENABLED=False MAX_SIDE=1280
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# Test image = the runtime environment + pytest:  docker build --target test -t lumina:test .
FROM base AS test
RUN pip install pytest
CMD ["python", "-m", "pytest", "-q", "tests"]

# Runtime image (the default target):  docker build -t lumina . && docker run -p 7860:7860 lumina
FROM base AS runtime
# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["python", "app.py"]
