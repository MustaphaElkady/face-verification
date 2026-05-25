import numpy as np

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))

def is_duplicate_submission(emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.999) -> bool:
    return cosine_similarity(emb1, emb2) >= threshold