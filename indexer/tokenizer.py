import re
import string

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "their", "our", "your", "his", "her", "as", "if", "about", "into",
    "through", "during", "than", "more", "also", "other", "which",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens
