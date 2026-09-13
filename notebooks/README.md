# Notebooks

Starting points for fine-tuning the sentiment classifier used in `src/sentiment.py`.

- Fine-tune `bert-base-uncased` (or a lighter model like `distilbert-base-uncased`) as a
  3-class sequence classifier (negative / neutral / positive) using HuggingFace `Trainer`.
- Suggested data sources: SemEval-2014/2016 ABSA restaurant/laptop review datasets for a
  warm start, then a small hand-labeled sample (200-300 sentences) from your own domain
  for fine-tuning.
- Save the resulting checkpoint locally and point `--sentiment-model` in
  `src/pipeline.py` at that directory.
