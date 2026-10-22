import csv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash

from methods_config import (OUTPUT_PATH, PAIRS_PATH, TFIDF_VARIANTS, MINHASH_NGRAM, MINHASH_PERMUTATIONS, SEED)


def read_pairs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def tfidf_scores(texts_a: list[str], texts_b: list[str], analyzer: str, ngram_range: tuple[int, int]) -> list[float]:
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    vectorizer.fit(texts_a + texts_b)
    vectors_a = vectorizer.transform(texts_a)
    vectors_b = vectorizer.transform(texts_b)
    return np.asarray(vectors_a.multiply(vectors_b).sum(axis=1)).ravel().tolist()


def shingles(text: str) -> set[str]:
    text = text.lower()
    return {text[start:start + MINHASH_NGRAM] for start in range(max(len(text) - MINHASH_NGRAM + 1, 1))}


def minhash(text: str) -> MinHash:
    signature = MinHash(num_perm=MINHASH_PERMUTATIONS, seed=SEED)
    for shingle in shingles(text):
        signature.update(shingle.encode("utf-8"))
    return signature


def minhash_scores(texts_a: list[str], texts_b: list[str]) -> list[float]:
    return [minhash(text_a).jaccard(minhash(text_b)) for text_a, text_b in zip(texts_a, texts_b)]


def write_scores(pair_ids: list[str], scores: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pair_id", *scores])
        for index, pair_id in enumerate(pair_ids):
            writer.writerow([pair_id, *(round(values[index], 4) for values in scores.values())])


def main() -> None:
    pairs = read_pairs(Path(PAIRS_PATH))
    texts_a = [pair["text_a"] for pair in pairs]
    texts_b = [pair["text_b"] for pair in pairs]
    scores = {name: tfidf_scores(texts_a, texts_b, analyzer, ngram_range)
              for name, (analyzer, ngram_range) in TFIDF_VARIANTS.items()}
    scores["minhash"] = minhash_scores(texts_a, texts_b)
    write_scores([pair["pair_id"] for pair in pairs], scores, Path(OUTPUT_PATH))
    print("пар:", len(pairs), " результат:", OUTPUT_PATH)


if __name__ == "__main__":
    main()