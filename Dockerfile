FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached on code-only changes
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code and data (images excluded via .dockerignore)
COPY . .

# HF Spaces requires the app to run as user 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
