from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from lexical import read_pairs, write_scores
from methods_config import (PAIRS_PATH, SEMANTIC_OUTPUT_PATH, BATCH_SIZE, MAX_SEQ_LENGTH, MINILM_MODEL, E5_MODEL, BGE_M3_MODEL)

def minilm_scores(texts_a: list[str], texts_b: list[str]) -> list[float]:
    model = SentenceTransformer(MINILM_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    vectors_a = model.encode(texts_a, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    vectors_b = model.encode(texts_b, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    return np.sum(vectors_a * vectors_b, axis=1).tolist()


def e5_scores(texts_a: list[str], texts_b: list[str]) -> list[float]:
    model = SentenceTransformer(E5_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    texts_a = ["query: " + text for text in texts_a]
    texts_b = ["query: " + text for text in texts_b]
    vectors_a = model.encode(texts_a, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    vectors_b = model.encode(texts_b, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    return np.sum(vectors_a * vectors_b, axis=1).tolist()


def bge_m3_scores(texts_a: list[str], texts_b: list[str]) -> list[float]:
    model = SentenceTransformer(BGE_M3_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    vectors_a = model.encode(texts_a, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    vectors_b = model.encode(texts_b, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    return np.sum(vectors_a * vectors_b, axis=1).tolist()


def main() -> None:
    pairs = read_pairs(Path(PAIRS_PATH))
    texts_a = [pair["text_a"] for pair in pairs]
    texts_b = [pair["text_b"] for pair in pairs]
    scores = {
        "minilm": minilm_scores(texts_a, texts_b),
        "e5_base": e5_scores(texts_a, texts_b),
        "bge_m3": bge_m3_scores(texts_a, texts_b),
    }
    write_scores([pair["pair_id"] for pair in pairs], scores, Path(SEMANTIC_OUTPUT_PATH))
    print("пар:", len(pairs), " результат:", SEMANTIC_OUTPUT_PATH)


if __name__ == "__main__":
    main()