from datetime import datetime, timezone

REPO = "aisbergpublicorganization/telegram-news-ua-dataset"
REVISION: str | None = "2caff5263980833761ce291181bf732678c2fdac"
MONTHS = ["07_2026", "08_2026"]
START_MS = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
DAY_MS = 24 * 60 * 60 * 1000
SEED = 42

MIN_KEY_CHARS = 15
FOOTER_MIN_POSTS = 20

TEMPLATE_WORDS = 4
TEMPLATE_MIN_POSTS = 15
TEMPLATE_MIN_DAYS = 5
MAX_PAIRS_PER_TEMPLATE = 3
MAX_PAIRS_PER_POST = 2

CLUSTER_PREFIX_MIN_CHARS = 20
LEXICAL_MIN_SIMILARITY = 0.4
RANDOM_CANDIDATES_PER_QUOTA = 10
RANDOM_ATTEMPTS_PER_QUOTA = 100

QUOTAS = {"exact": 1700, "cluster": 1500, "lexical": 1500, "random": 1300}
OUTPUT_DIR = "data_preparation/output"
