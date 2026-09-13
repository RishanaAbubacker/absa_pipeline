"""
Per-aspect sentiment scoring.

Wraps a fine-tuned BERT sequence-classification model. Fine-tune on a
small hand-labeled set (or a public ABSA benchmark like SemEval-2014/2016)
before using in production — see notebooks/ for a fine-tuning starting
point. This module only handles inference.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["negative", "neutral", "positive"]


class SentimentScorer:
    def __init__(self, model_path: str):
        """`model_path` should point to a fine-tuned 3-class checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def score(self, sentence: str) -> tuple[str, float]:
        """Return (sentiment_label, confidence) for a sentence."""
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=0)
        idx = int(torch.argmax(probs))
        return LABELS[idx], float(probs[idx])

    def score_batch(self, sentences: list[str]) -> list[tuple[str, float]]:
        return [self.score(s) for s in sentences]
