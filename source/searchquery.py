# Using FAISS Search
#"FAISS (Facebook AI Similarity Search) is an open-source ", "library" ", for efficient similarity search and clustering of dense vectors. Developed by Meta AI Research, it is highly optimized for handling large-scale, high-dimensional datasets, enabling applications like image search, recommendation systems, and semantic search in NLP. "
#"FAISS provides various algorithms for indexing and searching, including flat (brute-force), inverted file, and product quantization methods, allowing users to balance between search accuracy and speed based on their specific needs."


import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Load the data and embeddings
data_folder= "D:/Harshita Ajmani/Code_harshu/NLP/data"
model_name = "intfloat/multilingual-e5-base"

embeddings_en = np.load(f"{data_folder}/embeddings_en.npy").astype('float32')
embeddings_fr = np.load(f"{data_folder}/embeddings_fr.npy").astype('float32')

df = pd.read_csv(f"{data_folder}/index_data.csv")
print(f"Loaded {len(df):,} records")
print(f"English embeddings shape: {embeddings_en.shape}")
print(f"French embeddings shape: {embeddings_fr.shape}")

# ".astype('float32') converts numbers to 32-bit floats — FAISS strictly requires float32, not float64"



#Building the FAISS Index

dimension = embeddings_en.shape[1]  

print(f"Building FAISS index with dimension: {dimension}")

index_en = faiss.IndexFlatIP(dimension)  
index_fr = faiss.IndexFlatIP(dimension)  

index_en.add(embeddings_en)
index_fr.add(embeddings_fr)

print(f"English index built — {index_en.ntotal:,} vectors")
print(f"French index built — {index_fr.ntotal:,} vectors")

#Loading the Model (for Query Encoding)
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer(model_name , device=device)



# Search function

def search(query, language="en", top_k=5):
    """
    Search the index with a query string.
    
    Args:
        query : user search query (any language)
        lang  : which index to search — "en" or "fr"
        top_k : number of results to return
    
    Returns:
        list of dicts with search results
    """
    
    #Encode the query
    query_embedding = model.encode( f"query: {query}", normalize_embeddings=True).astype('float32').reshape(1, -1)
    
    # Search FAISS index
    index  = index_en if language == "en" else index_fr
    scores, indices = index.search(query_embedding, top_k)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        results.append({
            "rank":     len(results) + 1,
            "score":    round(float(score), 4),
            "title_en": row["title_en"],
            "title_fr": row["title_fr"],
            "desc_en":  str(row["desc_en"])[:200],
            "desc_fr":  str(row["desc_fr"])[:200],
            "org":      row["org"],
            "subject":  row["subject"],
        })
    return results


# teasting the search function

def print_results(query, lang, results):
    print(f"\n{'='*60}")
    print(f"Query  : '{query}'")
    print(f"Lang   : {lang.upper()} index")
    print(f"{'='*60}")
    
    for r in results:
        print(f"\n  #{r['rank']} | Score: {r['score']}")
        print(f"  Title EN : {r['title_en']}")
        print(f"  Title FR : {r['title_fr']}")
        print(f"  Org      : {r['org']}")
        print(f"  Desc     : {r['desc_en'][:120]}...")



if __name__ == "__main__":
    print("\n Semantic Search — Canada Open Data Catalog")
    print("=" * 60)
    print("Languages: 'en' or 'fr'")
    print("=" * 60)

    query = input("\n Enter your search query: ").strip()
    lang = input(" Search in EN or FR index? (en/fr): ").strip().lower()

    results = search(query, language=lang, top_k=5)
    print_results(query, lang, results)