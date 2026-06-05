# Prepare and upload embeddings to HuggingFace

import numpy as np
import faiss
import pandas as pd
from huggingface_hub import HfApi, create_repo
import os

# ── Config ────────────────────────────────────────────────────────────────────
data_path  = "D:/Harshita Ajmani/Code_harshu/NLP/data/"
hf_repo  = "harshuajmani/nlp_geoCanada_embeddings"
save_path  = "D:/Harshita Ajmani/Code_harshu/NLP/deploy/"
os.makedirs(save_path, exist_ok=True)

# ── Step 1: Load original embeddings ─────────────────────────────────────────
print(" Loading embeddings...")
embeddings_en = np.load(data_path + "embeddings_en.npy").astype('float32')
embeddings_fr = np.load(data_path + "embeddings_fr.npy").astype('float32')
dataframe = pd.read_csv(data_path + "index_data.csv").fillna("")
print(f"✅ Loaded {len(dataframe):,} records")
print(f"   EN shape: {embeddings_en.shape}")
print(f"   FR shape: {embeddings_fr.shape}")

# ── Step 2: Convert to float16 (halves file size) ────────────────────────────
print("\n🔧 Converting to float16...")
embeddings_en_f16 = embeddings_en.astype('float16')
embeddings_fr_f16 = embeddings_fr.astype('float16')

np.save(save_path + "embeddings_en.npy", embeddings_en_f16)
np.save(save_path + "embeddings_fr.npy", embeddings_fr_f16)

size_en = os.path.getsize(save_path + "embeddings_en.npy") / 1e6
size_fr = os.path.getsize(save_path + "embeddings_fr.npy") / 1e6
print(f"✅ embeddings_en.npy → {size_en:.1f} MB")
print(f"✅ embeddings_fr.npy → {size_fr:.1f} MB")

# ── Step 3: Build and save FAISS indexes ─────────────────────────────────────
print("\n⚙️  Building and saving FAISS indexes...")
dim      = embeddings_en.shape[1]
index_en = faiss.IndexFlatIP(dim)
index_fr = faiss.IndexFlatIP(dim)
index_en.add(embeddings_en)
index_fr.add(embeddings_fr)

faiss.write_index(index_en, save_path + "index_en.faiss")
faiss.write_index(index_fr, save_path + "index_fr.faiss")

size_ien = os.path.getsize(save_path + "index_en.faiss") / 1e6
size_ifr = os.path.getsize(save_path + "index_fr.faiss") / 1e6
print(f"✅ index_en.faiss → {size_ien:.1f} MB")
print(f"✅ index_fr.faiss → {size_ifr:.1f} MB")

# ── Step 4: Save index_data.csv ───────────────────────────────────────────────
dataframe.to_csv(save_path + "index_data.csv", index=False)
size_csv = os.path.getsize(save_path + "index_data.csv") / 1e6
print(f"✅ index_data.csv  → {size_csv:.1f} MB")

# ── Step 5: Upload to HuggingFace ─────────────────────────────────────────────
print(f"\n🚀 Uploading to HuggingFace: {hf_repo}")

api = HfApi()

# Create repo if it doesn't exist
try:
    create_repo(hf_repo, repo_type="dataset", private=False)
    print(f"✅ Created repo: {hf_repo}")
except Exception:
    print(f"✅ Repo already exists: {hf_repo}")

# Upload all files
files = [
    "embeddings_en.npy",
    "embeddings_fr.npy",
    "index_en.faiss",
    "index_fr.faiss",
    "index_data.csv",
]

for file in files:
    print(f"   Uploading {file}...")
    api.upload_file(
        path_or_fileobj = save_path + file,
        path_in_repo    = file,
        repo_id         = hf_repo,
        repo_type       = "dataset",
    )
    print(f"✅ {file} uploaded")

print(f"\n🎉 All files uploaded!")
print(f"   View at: https://huggingface.co/datasets/{hf_repo}")