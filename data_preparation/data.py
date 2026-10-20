import json
from collections import Counter, defaultdict

from huggingface_hub import HfApi, hf_hub_download

from config import (DAY_MS, END_MS, FOOTER_MIN_POSTS, MIN_KEY_CHARS, MONTHS, REPO, REVISION, START_MS,
                    TEMPLATE_MIN_DAYS, TEMPLATE_MIN_POSTS)
from text_utils import clean_text, is_ukrainian, match_key, non_empty_lines, strip_footer, template_signature


def download(path: str) -> str:
    return hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION)


def read_records() -> list[dict]:
    records = []
    for month in MONTHS:
        with open(download(f"data/{month}.jsonl"), encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                records.append({
                    "channel": record.get("channel") or "",
                    "date": record["date"],
                    "raw": record.get("post_text_redacted") or "",
                })
    return records


def find_channel_footers(records: list[dict]) -> set[tuple[str, str]]:
    counts = Counter()
    for record in records:
        lines = non_empty_lines(record["raw"])
        if len(lines) > 1:
            counts[(record["channel"], lines[-1])] += 1
    return {footer for footer, count in counts.items() if count >= FOOTER_MIN_POSTS}


def load_posts() -> list[dict]:
    records = read_records()
    footers = find_channel_footers(records)
    print("повторюваних footer-рядків:", len(footers))

    posts = []
    for record in records:
        text = clean_text(strip_footer(record["raw"], record["channel"], footers))
        key = match_key(text)
        if len(key) >= MIN_KEY_CHARS and is_ukrainian(text):
            posts.append({"id": len(posts), "channel": record["channel"], "date": record["date"],
                          "text": text, "key": key})
    return posts


def mark_templates(posts: list[dict]) -> None:
    for post in posts:
        post["template"] = template_signature(post["key"])
    counts, days = Counter(), defaultdict(set)
    for post in posts:
        counts[post["template"]] += 1
        days[post["template"]].add(post["date"] // DAY_MS)
    frequent = {template for template, count in counts.items()
                if count >= TEMPLATE_MIN_POSTS and len(days[template]) >= TEMPLATE_MIN_DAYS}
    for post in posts:
        if post["template"] not in frequent:
            post["template"] = None
    print("масових шаблонів:", len(frequent))


def load_clusters() -> list[dict]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    names = [name for name in files if name.startswith("clusters/cl_") and name.endswith(".json")
             and START_MS <= int(name.split("cl_")[1].split("_")[0]) < END_MS]
    clusters = []
    for name in names:
        with open(download(name), encoding="utf-8") as file:
            clusters.append(json.load(file))
    return clusters
