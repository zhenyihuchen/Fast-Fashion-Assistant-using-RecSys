FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 user && chown -R user /app
USER 1000

EXPOSE 7860

# startup.py downloads missing binary data files from GitHub on first boot
CMD ["sh", "-c", "python3 startup.py && uvicorn backend.api:app --host 0.0.0.0 --port 7860"]
