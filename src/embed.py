"""
Contextual BERT embedding for candidate aspect phrases.

Each phrase is embedded within its original sentence context (not in
isolation) so that meaning stays tied to context — e.g. "return policy"
in a complaint reads differently from "return policy" mentioned neutrally.
The phrase's embedding is the mean-pooled hidden state over its token span.
"""

import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "bert-base-uncased"
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed_phrase_in_context(sentence: str, phrase: str) -> torch.Tensor:
    """
    Embed `phrase` using the hidden states produced when encoding the full
    `sentence`, mean-pooled over the phrase's token span.

    Falls back to a full-sentence mean pool if the phrase's token span
    can't be located (e.g. tokenization mismatch).
    """
    tokenizer, model = _load()
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state[0]

    phrase_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase))
    input_ids = inputs["input_ids"][0].tolist()

    start = None
    for i in range(len(input_ids) - len(phrase_ids) + 1):
        if input_ids[i : i + len(phrase_ids)] == phrase_ids:
            start = i
            break

    if start is None:
        return hidden.mean(dim=0)
    return hidden[start : start + len(phrase_ids)].mean(dim=0)


def embed_batch(pairs: list[tuple[str, str]]) -> torch.Tensor:
    """Embed a list of (sentence, phrase) pairs, returning a stacked tensor."""
    return torch.stack([embed_phrase_in_context(s, p) for s, p in pairs])
