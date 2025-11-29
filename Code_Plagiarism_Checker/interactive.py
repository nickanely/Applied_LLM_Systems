#!/usr/bin/env python
# coding: utf-8

# In[19]:


import json
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tiktoken
import tree_sitter_python as tspython
from openai import OpenAI
from tqdm import tqdm
from tree_sitter import Language, Parser

# In[20]:


# I have stored my keys in run configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key missing. Fix your setup.")

client = OpenAI(api_key=OPENAI_API_KEY)
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
tokenizer = tiktoken.get_encoding("cl100k_base")

INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5-nano"
BATCH_SIZE = 128


# In[21]:


def load_indexes():
    """Load pre-built indexes and corpus"""
    dense_index = faiss.read_index(str(INDEX_DIR / "dense.faiss"))

    with open(INDEX_DIR / "bm25.pkl", 'rb') as f:
        sparse_index = pickle.load(f)

    corpus = pd.read_csv(DATA_DIR / "reference_corpus.csv")

    return dense_index, sparse_index, corpus


# ## Helpers

# In[22]:


def embed_batch(texts, batch_size=BATCH_SIZE):
    """Generate embeddings in batches using OpenAI API"""
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeds.extend([d.embedding for d in resp.data])
    return np.array(all_embeds, dtype=np.float32)

def tokenize_code(code):
    """Tokenize code for BM25 (alphanumeric + symbols)"""
    return re.findall(r'[A-Za-z0-9_]+|[^A-Za-z0-9_\s]', code)


# ## Retrievers

# In[23]:


def dense_retrieve(query_code, dense_index, corpus, top_k=10):
    """Retrieve using dense embeddings only"""
    query_emb = embed_batch([query_code])[0].reshape(1, -1)
    faiss.normalize_L2(query_emb)

    scores, indices = dense_index.search(query_emb, top_k)

    results = corpus.iloc[indices[0]].copy()
    results['similarity'] = scores[0]
    return results.reset_index(drop=True)

def sparse_retrieve(query_code, sparse_index, corpus, top_k=10):
    """Retrieve using BM25 only"""
    query_tokens = tokenize_code(query_code)
    scores = sparse_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = corpus.iloc[top_indices].copy()
    results['bm25_score'] = scores[top_indices]
    return results.reset_index(drop=True)

def hybrid_retrieve(query_code, dense_index, sparse_index, corpus, top_k=10, rrf_k=60):
    """Retrieve using hybrid fusion (RRF)"""
    # Dense retrieval
    dense_results = dense_retrieve(query_code, dense_index, corpus, top_k * 2)
    dense_ranks = {row['id']: i for i, (_, row) in enumerate(dense_results.iterrows())}

    # Sparse retrieval
    sparse_results = sparse_retrieve(query_code, sparse_index, corpus, top_k * 2)
    sparse_ranks = {row['id']: i for i, (_, row) in enumerate(sparse_results.iterrows())}

    # RRF fusion
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    fused_scores = {}

    for doc_id in all_ids:
        score = 0
        if doc_id in dense_ranks:
            score += 1 / (rrf_k + dense_ranks[doc_id])
        if doc_id in sparse_ranks:
            score += 1 / (rrf_k + sparse_ranks[doc_id])
        fused_scores[doc_id] = score

    # Sort and select top-k
    top_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = corpus[corpus['id'].isin([doc_id for doc_id, _ in top_ids])].copy()
    results['fused_score'] = results['id'].map(dict(top_ids))
    results = results.sort_values('fused_score', ascending=False)

    return results.reset_index(drop=True)


# ## Ask LLM

# In[24]:


def ask_llm_plagiarism(query_code, context_codes):
    """Ask LLM to determine plagiarism given context"""
    context_str = "\n\n---\n\n".join([
        f"REFERENCE {i+1}:\n```python\n{code}\n```"
        for i, code in enumerate(context_codes)
    ])

    prompt = f"""You are a code plagiarism detector. Analyze if the QUERY code is plagiarized from any REFERENCE code.

QUERY CODE:
```python
{query_code}
```

REFERENCE CODES:
{context_str}

Determine:
1. Is the query plagiarized? (YES/NO)
2. If yes, which reference(s)?
3. Plagiarism confidence (0.0-1.0)
4. Brief explanation

Return JSON:
{{
  "is_plagiarized": true/false,
  "confidence": 0.0-1.0,
  "matched_references": [1, 2, ...],
  "explanation": "brief reason"
}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],

    )

    result = response.choices[0].message.content.strip()
    result = result.replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(result)
    except:
        return {"is_plagiarized": False, "confidence": 0.0, "matched_references": [], "explanation": "Parse error"}


# ## 4 Callable Functions - Detections

# In[25]:


def detect_embedding(query_code, threshold=0.8):
    """System 1: Pure embedding search with threshold"""
    dense_index, _, corpus = load_indexes()

    results = dense_retrieve(query_code, dense_index, corpus, top_k=5)

    # Check if any result exceeds threshold
    is_plagiarized = (results['similarity'].max() >= threshold)

    return {
        "method": "embedding",
        "is_plagiarized": bool(is_plagiarized),
        "max_similarity": float(results['similarity'].max()),
        "threshold": threshold,
        "top_matches": results[['function_name', 'similarity', 'file_path']].to_dict('records')
    }

def detect_llm(query_code, max_context_functions=5000):
    """System 2: Direct LLM analysis with full corpus context
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    NOTE!!!! May be limited due to model context limitations.
    During testing, I couldn't pass whole corpus
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    _, _, corpus = load_indexes()

    # Sample corpus (or use full if small enough)
    context_sample = corpus.sample(n=min(max_context_functions, len(corpus)))
    context_codes = context_sample['code'].tolist()

    llm_result = ask_llm_plagiarism(query_code, context_codes)

    return {
        "method": "direct_llm",
        "is_plagiarized": llm_result["is_plagiarized"],
        "confidence": llm_result["confidence"],
        "explanation": llm_result["explanation"],
        "context_size": len(context_codes)
    }

def detect_rag(query_code, top_k=5):
    """System 3: Standard RAG (dense retrieval + LLM)"""
    dense_index, _, corpus = load_indexes()

    # Retrieve relevant context
    results = dense_retrieve(query_code, dense_index, corpus, top_k=top_k)
    context_codes = results['code'].tolist()

    # LLM analysis
    llm_result = ask_llm_plagiarism(query_code, context_codes)

    return {
        "method": "rag",
        "is_plagiarized": llm_result["is_plagiarized"],
        "confidence": llm_result["confidence"],
        "explanation": llm_result["explanation"],
        "retrieved_functions": results[['function_name', 'similarity']].to_dict('records')
    }

def detect_hybrid_rag(query_code, top_k=5):
    """System 4: Hybrid RAG (dense + BM25 + LLM)"""
    dense_index, sparse_index, corpus = load_indexes()

    # Hybrid retrieval
    results = hybrid_retrieve(query_code, dense_index, sparse_index, corpus, top_k=top_k)
    context_codes = results['code'].tolist()

    # LLM analysis
    llm_result = ask_llm_plagiarism(query_code, context_codes)

    return {
        "method": "hybrid_rag",
        "is_plagiarized": llm_result["is_plagiarized"],
        "confidence": llm_result["confidence"],
        "explanation": llm_result["explanation"],
        "retrieved_functions": results[['function_name', 'fused_score']].to_dict('records')
    }


# In[26]:


# Step 2: Test detection
test_code = """
def vector_components(size: float, direction: float, in_radians: bool = False) -> list[float]:\n    if in_radians: \n        return [size * cos(direction), size * sin(direction)]\n    return [size * cos(radians(direction)), size * sin(radians(direction))]",

"""

print(detect_embedding(test_code))
print(detect_llm(test_code, 40))
print(detect_rag(test_code))
print(detect_hybrid_rag(test_code))


# In[ ]:




