import random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data_config import (CLUSTER_PREFIX_MIN_CHARS, DAY_MS, LEXICAL_MIN_SIMILARITY, QUOTAS,
                    RANDOM_ATTEMPTS_PER_QUOTA, RANDOM_CANDIDATES_PER_QUOTA)
from text_utils import clean_text, match_key

type Post = dict
type Candidate = tuple[str | int, Post, Post]


def exact_groups(posts: list[Post]) -> list[list[Post]]:
    groups_by_key = defaultdict(list)
    for post in posts:
        groups_by_key[post["key"]].append(post)
    groups = list(groups_by_key.values())
    random.shuffle(groups)
    return groups


def match_cluster_posts(cluster: dict, posts_by_channel_day: dict[tuple[str, int], list[Post]]) -> list[Post]:
    found = []
    for cluster_post in cluster.get("posts", []):
        published = cluster_post.get("published_at")
        prefix = " ".join(match_key(clean_text(cluster_post.get("text") or "")).split()[:-1])
        if not published or len(prefix) < CLUSTER_PREFIX_MIN_CHARS:
            continue
        published_ms = int(datetime.fromisoformat(published).replace(tzinfo=timezone.utc).timestamp() * 1000)
        day = published_ms // DAY_MS
        channel = cluster_post.get("channel_name") or ""
        matches = [post for nearby_day in (day - 1, day, day + 1)
                   for post in posts_by_channel_day[(channel, nearby_day)]
                   if post["key"].startswith(prefix) or prefix.startswith(post["key"])]
        if len(matches) == 1 and matches[0] not in found:
            found.append(matches[0])
    return found


def cluster_pairs(posts: list[Post], clusters: list[dict]) -> list[Candidate]:
    posts_by_channel_day = defaultdict(list)
    for post in posts:
        posts_by_channel_day[(post["channel"], post["date"] // DAY_MS)].append(post)

    candidates = []
    for cluster in clusters:
        found = match_cluster_posts(cluster, posts_by_channel_day)
        pairs = [(first, second) for index, first in enumerate(found) for second in found[index + 1:]
                 if first["channel"] != second["channel"] and first["key"] != second["key"]]
        if pairs:
            first, second = random.choice(pairs)
            candidates.append((cluster["cluster_id"], first, second))
    random.shuffle(candidates)
    return candidates


def lexical_pairs(posts: list[Post]) -> list[Candidate]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5), min_df=2, sublinear_tf=True)
    vectors = vectorizer.fit_transform([post["key"] for post in posts])
    days = np.array([post["date"] // DAY_MS for post in posts])
    channels = np.array([post["channel"] for post in posts])
    keys = np.array([post["key"] for post in posts])

    pairs = set()
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        neighbours = np.flatnonzero(np.abs(days - day) <= 1)
        similarity = (vectors[rows] @ vectors[neighbours].T).toarray()
        similarity[channels[rows][:, None] == channels[neighbours][None, :]] = -1
        similarity[keys[rows][:, None] == keys[neighbours][None, :]] = -1
        for row, column in enumerate(similarity.argmax(axis=1)):
            if similarity[row, column] >= LEXICAL_MIN_SIMILARITY:
                pairs.add(tuple(sorted((int(rows[row]), int(neighbours[column])))))

    candidates = [("", posts[first], posts[second]) for first, second in pairs]
    random.shuffle(candidates)
    return candidates


def random_pairs(posts: list[Post]) -> list[Candidate]:
    posts_by_day = defaultdict(list)
    for post in posts:
        posts_by_day[post["date"] // DAY_MS].append(post)

    target = QUOTAS["random"] * RANDOM_CANDIDATES_PER_QUOTA
    max_attempts = QUOTAS["random"] * RANDOM_ATTEMPTS_PER_QUOTA
    pairs, attempts = set(), 0
    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        day_posts = random.choice(list(posts_by_day.values()))
        if len(day_posts) < 2:
            continue
        first, second = random.sample(day_posts, 2)
        if first["channel"] != second["channel"] and first["key"] != second["key"]:
            pairs.add(tuple(sorted((first["id"], second["id"]))))

    candidates = [("", posts[first], posts[second]) for first, second in pairs]
    random.shuffle(candidates)
    return candidates
