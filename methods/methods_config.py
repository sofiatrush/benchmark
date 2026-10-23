PAIRS_PATH = "data/labeled_pairs.csv"
OUTPUT_PATH = "results/lexical_scores.csv"
SEMANTIC_OUTPUT_PATH = "results/semantic_scores.csv"

TFIDF_VARIANTS = {
    "tfidf_word": ("word", (1, 1)),
    "tfidf_char3": ("char_wb", (3, 3)),
    "tfidf_char4": ("char_wb", (4, 4)),
    "tfidf_char5": ("char_wb", (5, 5)),
}

SEED = 42

MINHASH_NGRAM = 5
MINHASH_PERMUTATIONS = 128
LSH_THRESHOLD = 0.5

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512
MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_MODEL = "intfloat/multilingual-e5-base"
BGE_M3_MODEL = "BAAI/bge-m3"