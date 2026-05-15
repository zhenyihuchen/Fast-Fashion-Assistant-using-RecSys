FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu, wget for data download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached on code-only changes
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code (binary data files are downloaded below)
COPY . .

# Download binary data files from GitHub (too large for HF Spaces git)
ENV GH=https://raw.githubusercontent.com/zhenyihuchen/Fast-Fashion-Assistant-using-RecSys/main

RUN mkdir -p offline/data/image_embeddings/clip \
             offline/data/image_embeddings/fashion_clip \
             offline/data/text_embeddings/clip \
             offline/data/text_embeddings/fashion_clip \
             offline/data/prompt_embeddings/clip \
             offline/data/prompt_embeddings/fashion_clip

RUN wget -q $GH/offline/data/women_data.parquet -O offline/data/women_data.parquet

RUN wget -q $GH/offline/data/image_embeddings/clip/image_embeddings.npy \
             -O offline/data/image_embeddings/clip/image_embeddings.npy && \
    wget -q $GH/offline/data/image_embeddings/clip/faiss.index \
             -O offline/data/image_embeddings/clip/faiss.index && \
    wget -q $GH/offline/data/image_embeddings/fashion_clip/image_embeddings.npy \
             -O offline/data/image_embeddings/fashion_clip/image_embeddings.npy && \
    wget -q $GH/offline/data/image_embeddings/fashion_clip/faiss.index \
             -O offline/data/image_embeddings/fashion_clip/faiss.index

RUN wget -q $GH/offline/data/text_embeddings/clip/text_embeddings.npy \
             -O offline/data/text_embeddings/clip/text_embeddings.npy && \
    wget -q $GH/offline/data/text_embeddings/fashion_clip/text_embeddings.npy \
             -O offline/data/text_embeddings/fashion_clip/text_embeddings.npy

RUN wget -q $GH/offline/data/prompt_embeddings/clip/occasion_embeddings.npz \
             -O offline/data/prompt_embeddings/clip/occasion_embeddings.npz && \
    wget -q $GH/offline/data/prompt_embeddings/fashion_clip/occasion_embeddings.npz \
             -O offline/data/prompt_embeddings/fashion_clip/occasion_embeddings.npz && \
    wget -q "$GH/offline/data/prompt_embeddings/occasion_clavix_embeddings.npz" \
             -O offline/data/prompt_embeddings/occasion_clavix_embeddings.npz && \
    wget -q "$GH/offline/data/prompt_embeddings/occasion_prompt_embeddings.npz" \
             -O offline/data/prompt_embeddings/occasion_prompt_embeddings.npz

# HF Spaces requires the app to run as user 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
