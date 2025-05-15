# rag_utils.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"

# Load once
_model = SentenceTransformer(model_name)

# Used to remember index and chunks during runtime
_faiss_index = None
_faiss_chunks = None


def chunk_transcript(text, chunk_size=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def build_faiss_index(chunks):
    global _faiss_index, _faiss_chunks
    embeddings = _model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    _faiss_index = index
    _faiss_chunks = chunks
    return index, chunks


def query_faiss_index(question, top_k=1):
    global _faiss_index, _faiss_chunks
    if _faiss_index is None or _faiss_chunks is None:
        raise ValueError("FAISS index not built yet.")
    query_embedding = _model.encode([question], convert_to_numpy=True)
    D, I = _faiss_index.search(query_embedding, top_k)
    return [ _faiss_chunks[i] for i in I[0] ]


def save_index(index, chunks, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(save_dir) / "faiss.index"))
    with open(Path(save_dir) / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index(save_dir):
    global _faiss_index, _faiss_chunks
    index = faiss.read_index(str(Path(save_dir) / "faiss.index"))
    with open(Path(save_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    _faiss_index = index
    _faiss_chunks = chunks
    return index, chunks
