import csv
import random
from collections import Counter
from pathlib import Path

from data_config import MAX_PAIRS_PER_POST, MAX_PAIRS_PER_TEMPLATE, OUTPUT_DIR, QUOTAS, SEED
from data import load_clusters, load_posts, mark_templates
from source import Candidate, Post, cluster_pairs, exact_groups, lexical_pairs, random_pairs

type Row = tuple[str, str | int, Post, Post]


def select_pairs(groups: list[list[Post]], candidates: dict[str, list[Candidate]]) -> list[Row]:
    rows = []
    used_pairs = set()
    post_usage = Counter()
    template_usage = Counter()

    def try_add(source: str, cluster_id: str | int, first: Post, second: Post) -> bool:
        pair = tuple(sorted((first["id"], second["id"])))
        templates = {template for template in (first["template"], second["template"]) if template}
        if (pair in used_pairs
                or post_usage[first["id"]] >= MAX_PAIRS_PER_POST
                or post_usage[second["id"]] >= MAX_PAIRS_PER_POST
                or any(template_usage[template] >= MAX_PAIRS_PER_TEMPLATE for template in templates)):
            return False
        used_pairs.add(pair)
        post_usage.update((first["id"], second["id"]))
        template_usage.update(templates)
        rows.append((source, cluster_id, first, second))
        return True

    exact_count = 0
    for group in groups:
        if exact_count >= QUOTAS["exact"]:
            break
        pairs = [(first, second) for index, first in enumerate(group) for second in group[index + 1:]
                 if first["channel"] != second["channel"]]
        random.shuffle(pairs)
        for first, second in pairs:
            if try_add("exact", "", first, second):
                exact_count += 1
                break
    print("exact:", exact_count)

    for source in ("cluster", "lexical", "random"):
        count = 0
        for cluster_id, first, second in candidates[source]:
            if count >= QUOTAS[source]:
                break
            if try_add(source, cluster_id, first, second):
                count += 1
        print(f"{source}:", count)
    return rows


def write_pairs(rows: list[Row], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "pairs_full.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["pair_id", "source", "cluster_id", "channel_a", "channel_b", "text_a", "text_b", "label"])
        for pair_id, (source, cluster_id, first, second) in enumerate(rows):
            writer.writerow([pair_id, source, cluster_id, first["channel"], second["channel"],
                             first["text"], second["text"], ""])
    with open(output_dir / "pairs_label.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["pair_id", "text_a", "text_b", "label"])
        for pair_id, (_, _, first, second) in enumerate(rows):
            writer.writerow([pair_id, first["text"], second["text"], ""])


def main() -> None:
    random.seed(SEED)
    posts = load_posts()
    mark_templates(posts)
    print("постів:", len(posts))
    candidates = {
        "cluster": cluster_pairs(posts, load_clusters()),
        "lexical": lexical_pairs(posts),
        "random": random_pairs(posts),
    }
    rows = select_pairs(exact_groups(posts), candidates)
    random.shuffle(rows)
    write_pairs(rows, Path(OUTPUT_DIR))
    print("пар всього:", len(rows))


if __name__ == "__main__":
    main()
