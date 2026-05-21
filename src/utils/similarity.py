import numpy as np


def normalize_embedding(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def cosine_similarity(a, b):

    return np.dot(a, b)