import re
import unicodedata

from data_config import TEMPLATE_WORDS

MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\([^\n)]*\)")
TRUNCATED_MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\([^\n)]*$")
URL = re.compile(r"https?://\S+")


def is_ukrainian(text: str) -> bool:
    lowered = text.lower()
    return not any(letter in lowered for letter in "ыэъё") and any(letter in lowered for letter in "іїєґ")


def non_empty_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def strip_footer(raw: str, channel: str, footers: set[tuple[str, str]]) -> str:
    lines = non_empty_lines(raw)
    while len(lines) > 1 and (channel, lines[-1]) in footers:
        lines.pop()
    return "\n".join(lines)


def keep_link_text(line: str) -> str:
    return TRUNCATED_MARKDOWN_LINK.sub(r"\1", MARKDOWN_LINK.sub(r"\1", line))


def clean_text(raw: str) -> str:
    text = "\n".join(keep_link_text(line) for line in raw.splitlines())
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = URL.sub("", text).replace("[url]", "")
    return re.sub(r"\s+", " ", text).strip(" []|🔗\t\n")


def match_key(text: str) -> str:
    key = unicodedata.normalize("NFKC", text).lower().replace("’", "'").replace("ʼ", "'")
    key = re.sub(r"@\w+", " ", key)
    key = re.sub(r"[^\w\s']", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def template_signature(key: str) -> str:
    return " ".join(re.sub(r"\d+", "0", key).split()[:TEMPLATE_WORDS])
