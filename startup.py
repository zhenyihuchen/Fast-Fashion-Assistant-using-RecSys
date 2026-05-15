"""Downloads binary data files from GitHub at container startup."""
import urllib.request
from pathlib import Path

GH = "https://raw.githubusercontent.com/zhenyihuchen/Fast-Fashion-Assistant-using-RecSys/main"

FILES = [
    "offline/data/women_data.parquet",
    "offline/data/image_embeddings/clip/image_embeddings.npy",
    "offline/data/image_embeddings/clip/faiss.index",
    "offline/data/image_embeddings/fashion_clip/image_embeddings.npy",
    "offline/data/image_embeddings/fashion_clip/faiss.index",
    "offline/data/text_embeddings/clip/text_embeddings.npy",
    "offline/data/text_embeddings/fashion_clip/text_embeddings.npy",
    "offline/data/prompt_embeddings/clip/occasion_embeddings.npz",
    "offline/data/prompt_embeddings/fashion_clip/occasion_embeddings.npz",
]

for rel in FILES:
    dest = Path(rel)
    if dest.exists():
        print(f"[startup] already present: {rel}")
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GH}/{rel}"
    print(f"[startup] downloading {rel} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[startup] done ({dest.stat().st_size:,} bytes)", flush=True)

print("[startup] all data files ready", flush=True)
