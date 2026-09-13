# ABSA Pipeline — Unsupervised Aspect-Based Sentiment Analysis

A BERT-based Aspect-Based Sentiment Analysis (ABSA) pipeline built for Voice-of-Customer review analysis — no external LLM dependency. Aspects are discovered bottom-up via clustering rather than a fixed taxonomy.

## Pipeline stages

1. **Candidate extraction** (`src/extract.py`) — POS-pattern bigram extraction from review text using spaCy (ADJ+NOUN, NOUN+NOUN).
2. **Contextual embedding** (`src/embed.py`) — Each candidate phrase is embedded in its original sentence context using BERT, then pooled over the phrase's token span.
3. **Clustering** (`src/cluster.py`) — UMAP dimensionality reduction followed by HDBSCAN density-based clustering to discover aspect groups without a fixed cluster count.
4. **Sentiment scoring** (`src/sentiment.py`) — A fine-tuned BERT sentence classifier scores sentiment per aspect-bearing sentence.
5. **Aggregation** (`src/pipeline.py`) — Orchestrates the full pipeline end to end and outputs a review-level aspect/sentiment table.

## Why unsupervised clustering instead of a fixed taxonomy

Fixed aspect taxonomies go stale as products, complaints, and language evolve. Clustering surfaces emerging aspect themes directly from the data, and the mapping from cluster to a human-readable label is a one-time manual step per clustering run.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
python -m src.pipeline --input data/reviews.csv --output data/aspect_sentiment.csv
```

## Output schema

| column | description |
|---|---|
| review_id | source review identifier |
| bigram | extracted aspect phrase |
| aspect_cluster | cluster ID / human-assigned aspect label |
| sentiment | positive / negative / neutral |
| confidence | sentiment model confidence score |

## Notes

- Cluster-to-label mapping (`src/cluster.py::CLUSTER_LABELS`) is built once per clustering run by inspecting each cluster's top frequent bigrams, then reused for scoring new data.
- Sentiment classifier should be fine-tuned on a small hand-labeled or SemEval-style dataset before production use (see `notebooks/` for a fine-tuning starting point).
- This is a from-scratch/portfolio implementation, not the original production system.
