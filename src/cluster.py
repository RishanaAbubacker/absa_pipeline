"""
Aspect discovery via clustering.

Bigram embeddings (768-dim BERT vectors) are reduced with UMAP before
HDBSCAN clustering — density-based clustering on the reduced space lets
the number of aspect clusters emerge from the data instead of being
fixed upfront. Cluster -1 (HDBSCAN's noise label) should be routed to
manual review rather than forced into an aspect.
"""

from collections import Counter

import hdbscan
import numpy as np
import umap

# Populate this once per clustering run by inspecting each cluster's
# top frequent bigrams (see `label_clusters`), then reuse it for scoring
# new data without re-running UMAP/HDBSCAN each time.
CLUSTER_LABELS: dict[int, str] = {}


def reduce_and_cluster(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 10,
    min_cluster_size: int = 15,
) -> np.ndarray:
    """Reduce embedding dimensionality and cluster with HDBSCAN.

    Returns an array of cluster labels aligned with the input embeddings;
    label -1 indicates HDBSCAN's noise/outlier assignment.
    """
    reduced = umap.UMAP(
        n_neighbors=n_neighbors, n_components=n_components
    ).fit_transform(embeddings)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    return labels


def label_clusters(
    bigrams: list[str], labels: np.ndarray, top_n: int = 5
) -> dict[int, list[str]]:
    """
    For each cluster, return its top-N most frequent bigrams — use this
    output to manually assign a human-readable aspect name per cluster
    (e.g. cluster dominated by "delivery time", "shipping speed" -> "Delivery").
    """
    clusters: dict[int, Counter[str]] = {}
    for bg, label in zip(bigrams, labels):
        clusters.setdefault(label, Counter())[bg] += 1
    return {
        label: [bg for bg, _ in counter.most_common(top_n)]
        for label, counter in clusters.items()
    }
