from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any

import numpy as np

CORPUS = [
    "the language model predicts the next token from context",
    "a neural network learns patterns from training examples",
    "attention connects related words across a sentence",
    "embeddings represent meaning in a vector space",
    "the decoder produces text one token at a time",
    "good explanations make complex models easier to understand",
    "bad predictions can reveal an unreliable internal circuit",
    "small models run quickly on a laptop",
    "large models require more memory but learn useful representations",
    "the scientist measures accuracy and compares simple baselines",
    "cats chase small animals while dogs play with humans",
    "a king and a queen study while a man and a woman read",
    "maya writes code and leo studies mathematics",
    "paris is in france while rome is in italy",
    "the model can answer questions but it cannot always reason",
    "safe systems reduce harmful output and improve user trust",
    "fast processing saves time and slow processing costs resources",
    "positive examples are helpful and negative examples are misleading",
    "books contain data and punctuation that tokenizers split in different ways",
    "a careful experiment changes one variable and records every result",
    "the helpful assistant explains uncertainty and gives a useful answer",
    "the confused system repeats an error when evidence is incomplete",
    "when maya and leo arrived leo gave a book to maya",
    "when maya and leo arrived maya gave a book to leo",
]

IOI_PROMPTS = [
    "When Maya and Leo arrived, Leo gave a book to",
    "When Maya and Leo arrived, Maya gave a book to",
]
IOI_TARGETS = ["maya", "leo"]

POSITIVE_WORDS = {"good", "useful", "helpful", "positive", "safe", "fast", "careful", "trust"}
NEGATIVE_WORDS = {"bad", "unreliable", "harmful", "negative", "misleading", "confused", "error"}
NOUN_WORDS = {"model", "network", "token", "context", "attention", "sentence", "embeddings", "meaning", "space", "decoder", "text", "time", "scientist", "accuracy", "baseline", "system", "output", "answer", "data", "books", "assistant", "evidence"}
VERB_WORDS = {"predicts", "learns", "connects", "represent", "produces", "understand", "reveal", "require", "measures", "compares", "chase", "play", "study", "read", "writes", "answer", "reason", "reduce", "improve", "saves", "costs", "contain", "split", "changes", "records", "explains", "gives", "repeats"}


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’][\w]+)?|[^\w\s]", text.lower(), flags=re.UNICODE)


def corpus_tokens() -> list[list[str]]:
    return [word_tokens(sentence) for sentence in CORPUS]


@dataclass
class Context:
    seed: int = 42
    output: Path = Path("results")
    model_name: str = "toy"
    rng: np.random.Generator = field(init=False)
    cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.seed)

    def path(self, name: str) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        return self.output / name

    def json(self, name: str, data: Any) -> Path:
        path = self.path(name)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
        return path

    def csv(self, name: str, header: list[str], rows: list[list[Any]]) -> Path:
        import csv
        path = self.path(name)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def figure(self, name: str, draw) -> Path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.8))
        draw(fig, ax)
        fig.tight_layout()
        path = self.path(name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def softmax(values: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(float(temperature), 1e-8)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    unit = normalize_rows(values)
    return unit @ unit.T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def distributional_embeddings(dimensions: int = 48) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build small corpus-derived word vectors with PPMI and SVD; no model download."""
    cached = _EMBEDDING_CACHE.get(dimensions)
    if cached is not None:
        return cached
    sentences = corpus_tokens()
    vocab = sorted({token for sentence in sentences for token in sentence if token.isalpha()})
    index = {token: i for i, token in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)), dtype=float)
    for sentence in sentences:
        ids = [index[token] for token in sentence if token in index]
        for position, word_id in enumerate(ids):
            for other in range(max(0, position - 2), min(len(ids), position + 3)):
                if other != position:
                    counts[word_id, ids[other]] += 1 / abs(position - other)
    total = max(counts.sum(), 1.0)
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    expected = row @ col / total
    ppmi = np.maximum(np.log((counts * total + 1e-9) / (expected + 1e-9)), 0.0)
    u, singular, _ = np.linalg.svd(ppmi, full_matrices=False)
    rank = min(dimensions, len(singular))
    vectors = u[:, :rank] * np.sqrt(singular[:rank])
    if rank < dimensions:
        vectors = np.pad(vectors, ((0, 0), (0, dimensions - rank)))
    result = (vocab, vectors, counts)
    _EMBEDDING_CACHE[dimensions] = result
    return result


_EMBEDDING_CACHE: dict[int, tuple[list[str], np.ndarray, np.ndarray]] = {}


def token_category(token: str) -> str:
    token = token.strip().lower().replace("Ġ", "").replace("Ċ", "").replace("▁", "")
    if token in POSITIVE_WORDS:
        return "positive"
    if token in NEGATIVE_WORDS:
        return "negative"
    if token in NOUN_WORDS:
        return "noun"
    if token in VERB_WORDS:
        return "verb"
    return "other"


def numeric_char_vectors(numbers: list[int], dimensions: int = 24) -> np.ndarray:
    """Deterministic character-composition vectors for numeric token experiments."""
    rng = np.random.default_rng(2026)
    char_vectors = {char: rng.normal(size=dimensions) for char in "0123456789-"}
    rows = []
    for number in numbers:
        chars = str(number)
        vector = np.zeros(dimensions)
        for position, char in enumerate(chars):
            phase = (position + 1) / (len(chars) + 1)
            vector += char_vectors[char] * (1.0 + phase)
        rows.append(vector / max(len(chars), 1))
    return normalize_rows(np.asarray(rows))


def toy_transformer(seed: int = 42, n_layers: int = 6, n_heads: int = 4, d_model: int = 48, d_mlp: int = 96, seq_len: int = 12, batch_size: int = 8) -> dict[str, Any]:
    """A transparent, randomly initialized decoder used for offline mechanism demos."""
    rng = np.random.default_rng(seed)
    vocab, lexical, _ = distributional_embeddings(d_model)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    stream = [token_to_id[token] for sentence in corpus_tokens() for token in sentence if token in token_to_id]
    pad_id = token_to_id.get("the", 0)
    rows, masks, subject_positions = [], [], []
    for prompt in IOI_PROMPTS:
        words = [word for word in word_tokens(prompt) if word in token_to_id]
        ids = [token_to_id[word] for word in words][:seq_len]
        rows.append(ids + [pad_id] * (seq_len - len(ids)))
        masks.append([True] * len(ids) + [False] * (seq_len - len(ids)))
        subject = "leo" if "leo gave" in prompt.lower() else "maya"
        subject_positions.append(max(i for i, word in enumerate(words) if word == subject))
    starts = np.linspace(0, max(len(stream) - seq_len, 0), num=batch_size - len(rows), dtype=int)
    for start in starts:
        rows.append([stream[(start + j) % len(stream)] for j in range(seq_len)])
        masks.append([True] * seq_len)
    tokens = np.asarray(rows, dtype=int)
    token_mask = np.asarray(masks, dtype=bool)
    subject_positions.extend([-1] * (batch_size - len(subject_positions)))
    d_head = d_model // n_heads
    if d_head * n_heads != d_model:
        raise ValueError("d_model must be divisible by n_heads")
    hidden = lexical[tokens] + rng.normal(scale=0.05, size=(batch_size, seq_len, d_model))
    hidden_states = [hidden.copy()]
    all_q, all_k, all_v, all_scores, all_attention = [], [], [], [], []
    all_attn_heads, all_attn_contribs, all_mlp_pre, all_mlp_act, all_mlp_update = [], [], [], [], []
    qkv_weights, wo_weights, w1_weights, w2_weights = [], [], [], []
    causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    for _ in range(n_layers):
        scale = 1 / np.sqrt(d_model)
        wq = rng.normal(scale=scale, size=(d_model, d_model))
        wk = rng.normal(scale=scale, size=(d_model, d_model))
        wv = rng.normal(scale=scale, size=(d_model, d_model))
        wo = rng.normal(scale=scale, size=(n_heads, d_head, d_model))
        q = (hidden @ wq).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        k = (hidden @ wk).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        v = (hidden @ wv).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        masked = causal[None, None, :, :] | (~token_mask[:, None, None, :])
        scores = np.where(masked, -1e9, scores)
        attention = softmax(scores, axis=-1)
        head_values = attention @ v
        contributions = np.einsum("bhtd,hdf->bhtf", head_values, wo)
        hidden = hidden + contributions.sum(axis=1) / np.sqrt(n_heads)
        w1 = rng.normal(scale=1 / np.sqrt(d_model), size=(d_model, d_mlp))
        w2 = rng.normal(scale=1 / np.sqrt(d_mlp), size=(d_mlp, d_model))
        mlp_pre = hidden @ w1
        mlp_act = 0.5 * mlp_pre * (1 + np.tanh(np.sqrt(2 / np.pi) * (mlp_pre + 0.044715 * mlp_pre**3)))
        mlp_update = mlp_act @ w2 / np.sqrt(d_mlp)
        hidden = hidden + mlp_update
        hidden_states.append(hidden.copy())
        all_q.append(q); all_k.append(k); all_v.append(v)
        all_scores.append(scores); all_attention.append(attention)
        all_attn_heads.append(head_values); all_attn_contribs.append(contributions)
        all_mlp_pre.append(mlp_pre); all_mlp_act.append(mlp_act); all_mlp_update.append(mlp_update)
        qkv_weights.append((wq, wk, wv)); wo_weights.append(wo); w1_weights.append(w1); w2_weights.append(w2)
    unembedding = lexical.T / np.sqrt(d_model)
    logits = hidden @ unembedding
    return {
        "kind": "offline_toy", "tokens": tokens, "token_strings": [[vocab[i] for i in row] for row in tokens],
        "token_categories": [[token_category(vocab[i]) if token_mask[b, j] else "other" for j, i in enumerate(row)] for b, row in enumerate(tokens)],
        "attention_mask": token_mask, "last_positions": token_mask.sum(axis=1).astype(int) - 1,
        "subject_positions": np.asarray(subject_positions), "ioi_target_ids": np.asarray([token_to_id[w] for w in IOI_TARGETS]),
        "vocab": vocab, "embeddings": lexical, "hidden_states": np.stack(hidden_states),
        "q": np.stack(all_q), "k": np.stack(all_k), "v": np.stack(all_v),
        "attention_scores": np.stack(all_scores), "attention": np.stack(all_attention),
        "attention_heads": np.stack(all_attn_heads), "attention_contribs": np.stack(all_attn_contribs),
        "attention_updates": np.stack([values.sum(axis=1) / np.sqrt(n_heads) for values in all_attn_contribs]),
        "final_pre_norm": hidden_states[-1], "final_norm_weight": None, "final_norm_bias": None, "final_norm_eps": 1e-5,
        "mlp_scale": 1 / np.sqrt(d_mlp), "mlp_biases": [np.zeros(d_model) for _ in range(n_layers)],
        "mlp_pre": np.stack(all_mlp_pre), "mlp_act": np.stack(all_mlp_act), "mlp_update": np.stack(all_mlp_update),
        "qkv_weights": qkv_weights, "wo_weights": wo_weights, "w1_weights": w1_weights, "w2_weights": w2_weights,
        "unembedding": unembedding, "logits": logits,
    }


def apply_final_norm(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    weight = model.get("final_norm_weight")
    if weight is None:
        return values
    bias = model["final_norm_bias"]
    mean = values.mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + model["final_norm_eps"]) * weight + bias


def logit_lens_state(model: dict[str, Any], layer: int, values: np.ndarray) -> np.ndarray:
    final_index = model["hidden_states"].shape[0] - 1
    if model.get("kind") == "huggingface_gpt2" and layer == final_index:
        return values
    return apply_final_norm(model, values)


def bundle(ctx: Context) -> dict[str, Any]:
    key = f"bundle:{ctx.model_name}"
    if key not in ctx.cache:
        if ctx.model_name == "toy":
            ctx.cache[key] = toy_transformer(seed=ctx.seed)
        else:
            from .modeling import load_gpt2_bundle
            ctx.cache[key] = load_gpt2_bundle(ctx.model_name)
    return ctx.cache[key]


def print_result(project: int, title: str, paths: list[Path] | None = None) -> None:
    saved = ", ".join(str(path) for path in (paths or []))
    print(f"Projeto {project:02d} — {title}")
    if saved:
        print(f"Arquivos: {saved}")
