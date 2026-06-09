# Bilingual Semantic Search: Canadian Geospatial Metadata

> This is a semantic search engine over 46,468 bilingual government datasets from Canada's Open Data Portal, 
> supporting English and French queries including cross lingual searchs (EN query → FR results and vice versa).

---

## Reference Links

| Resource | Link |
|---|---|
| Demo | deployment |
| HuggingFace Dataset (embeddings) | `https://huggingface.co/datasets/harshuajmani/nlp_geoCanada_embeddings` |
| HuggingFace Model | `https://huggingface.co/intfloat/multilingual-e5-base` |
| Canada Open Data Source |`https://open.canada.ca/data/en/dataset/c4c5c7f1-bfa6-4ff6-b4a0-c164cb2060f7/resource/27f49273-d645-4950-9a58-c3086d833d57` |

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
4. [Model](#model)
5. [Project Structure](#project-structure)
6. [Setup and Installation](#setup-and-installation)
7. [Step by Step Pipeline](#step-by-step-pipeline)
8. [Evaluation](#evaluation)
9. [Deployment](#deployment)
10. [Tech Stack](#tech-stack)
11. [Key Concepts Explained](#key-concepts-explained)
12. [Results](#results)
13. [Limitations and Next Steps](#limitations-and-next-steps)

---

## Project Overview

- Qeury input in both English or French.
- Outputs top 5 semantically relevant data.
- Processes cross lingually, French query finds English records and vice versa
- Returns results in under 5ms across 46,468 records

---

The model used (`multilingual-e5-base`) maps both languages into the
**same vector space**

```
"water quality"     → [0.21, 0.84, 0.11, ...]
"qualité de l'eau"  → [0.22, 0.83, 0.12, ...]
```

---

## Dataset
**Canada Open Data Catalog** : Official metadata catalog of the Government of Canada

- Format: JSON Lines (jsonl)
- Size: ~500MB compressed
- Records: 46,468 government datasets

#### About the dataset?

Each record is **metadata**, not the actual data, but information *about* a dataset.

#### Example: 
```json
{
  "id": "000183ed-8864-42f0-ae43-c4313a860720",
  "title_translated": {
    "en": "Principal Mineral Areas, Producing Mines...",
    "fr": "Principales régions minières..."
  },
  "notes_translated": {
    "en": "This dataset is produced and published annually...",
    "fr": "Ce jeu de données est produit et publié..."
  },
  "keywords": {
    "en": ["mines", "minerals", "hydrocarbons", "oil"],
    "fr": ["mines", "minéraux", "hydrocarbures", "pétrole"]
  },
  "organization": {"title": "Natural Resources Canada"},
  "subject": ["economy"]
}
```

#### Fields utilised: 

`id` 
 `title_en`
 `title_fr` 
 `desc_en` 
 `desc_fr` 
 `keywords_en` 
 `keywords_fr`
 `org` 
 `subject` 

Note: The dataset has two kind of fileds `notes_translated` and `notes`. 

```python
record["notes"]            # Plain English string only
record["notes_translated"] # {"en": "...", "fr": "..."}
```

- The `_translated` suffix signals bilingual content in this catalog.

---

## HuggingFace Model Description

- **By:** Microsoft Research
- **Parameters:** 278 million
- **Languages:** 100+ including English and French
- **Embedding dimensions:** 768
- **Base architecture:** XLM-RoBERTa

---

## Setup and Installation

Step 1: Clone the repo

```bash
git clone https://github.com/HarshitaAjmani/Semantic-search-Canada-open-data.git
```

Step 2: Create conda environment

```bash
conda create -n search python -y
conda activate search
```

Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Step 4: Download the raw dataset

---

### Steps:

Run the scripts in following order.

---

Step 1: Clean the data

**File:** `source/load_data.ipynb`

**Output:** `data/cleaned_data.csv` (~46,468 rows, 8 columns)

---

Step 2: Generate embeddings

**File:** `source/embeddings.ipynb`

**GPU check before running:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
```

**Expected time:**
- With NVIDIA GPU: ~4 minutes per language
- With CPU only: ~45 minutes per language

**Output:**
- `data/embeddings_en.npy` — shape (46468, 768), 142MB
- `data/embeddings_fr.npy` — shape (46468, 768), 142MB

---

Step 3: Test search locally

**Run:**
```bash
python source/searchquery.py
```

**Example:**
```
Enter your search query: water quality monitoring
Search in EN or FR index? (en/fr): en

#1 | Score: 0.9121
   Title: Water Quality Monitoring Stations
   Org: Environment and Climate Change Canada
```

---

Step 4: Run evaluation

**File:** `source/evaluate.ipynb`

- **Saves:** `data/evaluation_results.csv` and `data/evaluation_results.png`

---

Step 5: Prepare and upload to HuggingFace

**File:** `source/hf_upload.py`

**Setup HuggingFace login:**
```python
# source/hf_login.py
from huggingface_hub import login
login(token="hf_your_token_here")
```

NEVER push `hf_login.py` to GitHub

**Run:**
```bash
python source/hf_login.py   # login first
python source/hf_upload.py  # then upload
```

Files uploaded:
```
embeddings_en.npy   71.4 MB  ← float16 English vectors
embeddings_fr.npy   71.4 MB  ← float16 French vectors
index_en.faiss     142.7 MB  ← pre-built English FAISS index
index_fr.faiss     142.7 MB  ← pre-built French FAISS index
index_data.csv      69.4 MB  ← lookup table for displaying results
```

---

Step 6: Run the app locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Evaluation

Approach

We evaluated using **100 automatically generated queries** across 4 types:

| Query Type | Count | Description |
|---|---|---|
| EN → EN | 25 | English query → English index (monolingual) |
| FR → FR | 25 | French query → French index (monolingual) |
| EN → FR | 25 | English query → French index (cross-lingual) |
| FR → EN | 25 | French query → English index (cross-lingual) |

**How queries were generated:**
1. Sample records with titles ≥ 5 words
2. Remove redundant articles (the, de, la, le, etc.)
3. Take first 4 meaningful words as query
4. True answer = the source record

### 4 Metrics --------------- CHecked till 

#### Metric 1 — Average Similarity Score
Measures how confident the model is about its results.

**Results:**
| Query Type | Avg Score | Min | Max | Std |
|---|---|---|---|---|
| EN → EN | 0.8749 | 0.845 | 0.900 | 0.0174 |
| FR → FR | 0.8647 | 0.831 | 0.894 | 0.0185 |
| EN → FR | 0.8413 | 0.810 | 0.881 | 0.0157 |
| FR → EN | 0.8405 | 0.805 | 0.863 | 0.0125 |
| **Overall** | **0.8554** | 0.805 | 0.900 | 0.0165 |

#### Metric 2 — EN vs FR Score Gap
Measures bilingual balance — is the model equally good in both languages?

```
EN avg score : 0.8577
FR avg score : 0.8530
Gap          : 0.0047

```

⚠️ Important: Gap alone is meaningless. Gap = 0 could mean both scores
are 0.85 (excellent) OR both are 0.00 (broken). Always read alongside
absolute scores.

#### Metric 3 — Cross-lingual Consistency
Measures whether EN and FR searches for the same concept return the same results.

```
For each record pair:
  EN query → EN index → top 5 IDs: [id1, id2, id3, id4, id5]
  FR query → FR index → top 5 IDs: [id1, id2, id6, id7, id8]
  Overlap = 2/5 = 40%

Average across 25 pairs: 40.8%
```

40.8% is expected — EN and FR indexes use different text descriptions
so some variation is correct behaviour.

#### Metric 4 — Score Distribution
Measures consistency — are scores reliable or scattered?

```
All std devs below 0.02 → scores are tightly clustered ✅
Low std = consistent, predictable search quality
High std = unpredictable, unreliable results
```

### Evaluation Limitations

- The data is not annotated, can't measure true precision/recall.
- Queries generated from dataset titles, may not reflect real user behaviour
---


### Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Set:
   - Repository: `your-username/your-repo`
   - Branch: `main`
   - Main file: `app.py`
5. Click **Deploy**

**First load:** ~2-3 minutes (downloading from HuggingFace)
**After that:** Instant (Streamlit caches everything)


---


## Results

```
Overall avg similarity score : 0.8554  ✅ High confidence
EN vs FR gap                 : 0.0047  ✅ Perfect bilingual balance
Cross-lingual consistency    : 40.8%   ⚠️ Moderate (expected)
Score std deviation          : < 0.02  ✅ Highly consistent

All 100 queries scored above 0.70 threshold ✅
```

---


## Quick Reference

### Run the full pipeline from scratch

```bash
# 1. Setup
conda create -n semantic-search python=3.10 -y
conda activate semantic-search
pip install -r requirements.txt

# 2. Download data
curl -L -o data/od-do-canada.jsonl.gz "https://open.canada.ca/static/od-do-canada.jsonl.gz"
gunzip data/od-do-canada.jsonl.gz

# 3. Clean data
python source/load_data.py

# 4. Generate embeddings (needs GPU, takes ~8 mins)
python source/embeddings.py

# 5. Test search
python source/searchquery.py

# 6. Evaluate
python source/evaluate.py

# 7. Upload to HuggingFace
python source/hf_login.py
python source/hf_upload.py

# 8. Run app
streamlit run app.py
```

### Skip the pipeline (use pre-built files from HuggingFace)

```bash
# App downloads everything automatically on first run
streamlit run app.py
# → opens http://localhost:8501
# → downloads from HuggingFace on first load (~2-3 mins)
# → instant on subsequent runs
```

---


Links = 
https://sbert.net/
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
