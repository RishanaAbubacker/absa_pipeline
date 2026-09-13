"""
End-to-end ABSA pipeline orchestration.

    reviews.csv -> extract candidate bigrams -> embed in context ->
    cluster into aspects -> score sentiment -> aspect_sentiment.csv

Usage:
    python -m src.pipeline --input data/reviews.csv --output data/aspect_sentiment.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from .cluster import CLUSTER_LABELS, label_clusters, reduce_and_cluster
from .embed import embed_batch
from .extract import extract_bigrams, extract_corpus_bigrams
from .sentiment import SentimentScorer


def run_pipeline(
    input_path: str,
    output_path: str,
    sentiment_model_path: str | None = None,
    min_bigram_frequency: int = 5,
) -> pd.DataFrame:
    reviews = pd.read_csv(input_path)
    if not {"review_id", "text"}.issubset(reviews.columns):
        raise ValueError("Input CSV must have 'review_id' and 'text' columns")

    # 1. Candidate extraction (corpus-level frequency filter)
    valid_bigrams = extract_corpus_bigrams(
        reviews["text"], min_frequency=min_bigram_frequency
    )

    rows = []
    for _, row in reviews.iterrows():
        for bg in extract_bigrams(row["text"]):
            if bg in valid_bigrams:
                rows.append({"review_id": row["review_id"], "text": row["text"], "bigram": bg})
    candidates = pd.DataFrame(rows)

    if candidates.empty:
        print("No candidate bigrams survived frequency filtering.", file=sys.stderr)
        candidates.to_csv(output_path, index=False)
        return candidates

    # 2. Contextual embedding
    pairs = list(zip(candidates["text"], candidates["bigram"]))
    embeddings = embed_batch(pairs).numpy()

    # 3. Clustering
    labels = reduce_and_cluster(embeddings)
    candidates["aspect_cluster_id"] = labels
    cluster_top_terms = label_clusters(candidates["bigram"].tolist(), labels)
    candidates["aspect_cluster"] = candidates["aspect_cluster_id"].map(
        lambda cid: CLUSTER_LABELS.get(cid, f"cluster_{cid}")
    )

    # 4. Sentiment scoring (optional — requires a fine-tuned checkpoint)
    if sentiment_model_path:
        scorer = SentimentScorer(sentiment_model_path)
        sentiments = scorer.score_batch(candidates["text"].tolist())
        candidates["sentiment"] = [s for s, _ in sentiments]
        candidates["confidence"] = [c for _, c in sentiments]
    else:
        candidates["sentiment"] = None
        candidates["confidence"] = None
        print(
            "No sentiment model provided — skipping sentiment scoring. "
            "Pass --sentiment-model to enable it.",
            file=sys.stderr,
        )

    # 5. Output
    out = candidates[
        ["review_id", "bigram", "aspect_cluster", "sentiment", "confidence"]
    ]
    out.to_csv(output_path, index=False)

    print(f"Discovered clusters (top terms): {cluster_top_terms}", file=sys.stderr)
    print(f"Wrote {len(out)} rows to {output_path}", file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the ABSA pipeline end to end.")
    parser.add_argument("--input", required=True, help="Path to input reviews CSV (columns: review_id, text)")
    parser.add_argument("--output", required=True, help="Path to write aspect/sentiment output CSV")
    parser.add_argument("--sentiment-model", default=None, help="Path to a fine-tuned sentiment classifier checkpoint")
    parser.add_argument("--min-bigram-frequency", type=int, default=5)
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        sentiment_model_path=args.sentiment_model,
        min_bigram_frequency=args.min_bigram_frequency,
    )


if __name__ == "__main__":
    main()
