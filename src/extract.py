"""
Candidate aspect-phrase extraction.

Extracts POS-pattern bigrams (ADJ+NOUN, NOUN+NOUN) from review text as
candidate aspect phrases, e.g. "fast shipping", "delivery time",
"customer support". Frequency filtering removes noise before these
candidates are embedded and clustered downstream.
"""

from collections import Counter
from typing import Iterable

import spacy

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_bigrams(text: str) -> list[str]:
    """Extract ADJ+NOUN / NOUN+NOUN bigrams from a single piece of text."""
    nlp = get_nlp()
    doc = nlp(text)
    bigrams = []
    for i in range(len(doc) - 1):
        w1, w2 = doc[i], doc[i + 1]
        if w1.pos_ in ("ADJ", "NOUN") and w2.pos_ == "NOUN":
            if w1.is_stop or w2.is_stop:
                continue
            bigrams.append(f"{w1.text} {w2.text}".lower())
    return bigrams


def extract_corpus_bigrams(
    texts: Iterable[str], min_frequency: int = 5
) -> dict[str, int]:
    """
    Extract bigrams across a full corpus and filter by minimum frequency.

    Returns a dict of {bigram: frequency} for bigrams appearing at least
    `min_frequency` times across the corpus — this is the main noise filter.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(extract_bigrams(text))
    return {bg: count for bg, count in counter.items() if count >= min_frequency}
