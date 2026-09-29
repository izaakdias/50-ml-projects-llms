from __future__ import annotations

from collections import Counter
import re

import numpy as np

from ..core import CORPUS, Context, corpus_tokens, print_result, word_tokens


class CharBPE:
    """A small character-pair learner for demonstrations; not a production tokenizer."""
    def __init__(self, merges: int = 40):
        self.n_merges = merges
        self.rules: list[tuple[str, str]] = []

    def fit(self, texts: list[str]) -> "CharBPE":
        sequences = [list(text) for text in texts]
        self.rules = []
        for _ in range(self.n_merges):
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts or counts.most_common(1)[0][1] < 2:
                break
            left, right = counts.most_common(1)[0][0]
            self.rules.append((left, right))
            merged = left + right
            for i, seq in enumerate(sequences):
                out = []
                j = 0
                while j < len(seq):
                    if j + 1 < len(seq) and (seq[j], seq[j + 1]) == (left, right):
                        out.append(merged)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                sequences[i] = out
        return self

    def encode(self, text: str) -> list[str]:
        tokens = list(text)
        for left, right in self.rules:
            out, i = [], 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                    out.append(left + right)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return "".join(tokens)


def _bpe() -> CharBPE:
    return CharBPE(merges=80).fit(CORPUS)


def _save_bars(ctx: Context, name: str, labels: list[str], values: list[float], ylabel: str) -> None:
    def draw(fig, ax):
        ax.bar(labels, values, color="#5376a6")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
    ctx.figure(name, draw)


def p01(ctx: Context) -> None:
    text = "The way you learn one token at a time."
    bpe = _bpe()
    encoded = {"character": list(text), "word": text.split(), "char_bpe": bpe.encode(text)}
    rows = [[scheme, len(tokens), " | ".join(tokens)] for scheme, tokens in encoded.items()]
    paths = [ctx.csv("p01_tokenizations.csv", ["scheme", "token_count", "tokens"], rows)]
    _save_bars(ctx, "p01_token_counts.png", list(encoded), [len(x) for x in encoded.values()], "Number of tokens")
    paths.append(ctx.path("p01_token_counts.png"))
    print_result(1, "Three tokenization schemes", paths)


def p02(ctx: Context) -> None:
    bpe = _bpe()
    rows = []
    for i, text in enumerate(CORPUS, 1):
        rows.append([i, len(text), len(text.split()), len(bpe.encode(text))])
    totals = np.sum(np.asarray([row[1:] for row in rows]), axis=0).tolist()
    paths = [ctx.csv("p02_text_lengths.csv", ["text_id", "characters", "words", "char_bpe_tokens"], rows)]
    _save_bars(ctx, "p02_total_lengths.png", ["characters", "words", "BPE tokens"], totals, "Total units in the local corpus")
    paths.append(ctx.path("p02_total_lengths.png"))
    print_result(2, "Text lengths in characters, words and tokens", paths)


def p03(ctx: Context) -> None:
    bpe = _bpe()
    lengths = [len(token.encode("utf-8")) for text in CORPUS for token in bpe.encode(text)]
    counts = Counter(lengths)
    rows = [[length, counts[length]] for length in sorted(counts)]
    paths = [ctx.csv("p03_token_length_frequency.csv", ["utf8_bytes", "frequency"], rows)]
    _save_bars(ctx, "p03_token_length_frequency.png", [str(row[0]) for row in rows], [row[1] for row in rows], "Frequency")
    paths.append(ctx.path("p03_token_length_frequency.png"))
    print_result(3, "Frequency table of token lengths", paths)


def p04(ctx: Context) -> None:
    examples = ["café", "ação", "naïve", "東京", "🙂", "model" ]
    chart_labels = ["café", "ação", "naïve", "CJK", "emoji", "model"]
    rows = [[token, len(token), len(token.encode("utf-8")), len(list(token.encode("utf-8")))] for token in examples]
    paths = [ctx.csv("p04_character_byte_lengths.csv", ["text", "unicode_characters", "utf8_bytes", "byte_values"], rows)]
    def draw(fig, ax):
        x = np.arange(len(examples)); width = .36
        ax.bar(x - width / 2, [r[1] for r in rows], width, label="Unicode code points")
        ax.bar(x + width / 2, [r[2] for r in rows], width, label="UTF-8 bytes")
        ax.set_xticks(x, chart_labels); ax.legend(); ax.set_ylabel("Length")
    ctx.figure("p04_characters_vs_bytes.png", draw)
    paths.append(ctx.path("p04_characters_vs_bytes.png"))
    print_result(4, "Character and byte lengths", paths)


def p05(ctx: Context) -> None:
    bpe = _bpe()
    char_count = sum(len(text.encode("utf-8")) for text in CORPUS)
    methods = {"characters": sum(len(text) for text in CORPUS), "words": sum(len(text.split()) for text in CORPUS), "char BPE": sum(len(bpe.encode(text)) for text in CORPUS)}
    rows = [[name, count, char_count / max(count, 1)] for name, count in methods.items()]
    paths = [ctx.csv("p05_compression.csv", ["tokenizer", "tokens", "utf8_bytes_per_token"], rows)]
    _save_bars(ctx, "p05_compression.png", [r[0] for r in rows], [r[2] for r in rows], "UTF-8 bytes per token")
    paths.append(ctx.path("p05_compression.png"))
    print_result(5, "Is tokenization compression?", paths)


def p06(ctx: Context) -> None:
    samples = {
        "Português": "modelos de linguagem aprendem padrões úteis",
        "English": "language models learn useful patterns",
        "Español": "los modelos de lenguaje aprenden patrones útiles",
        "Français": "les modèles de langue apprennent des motifs utiles",
        "Deutsch": "Sprachmodelle lernen nützliche Muster",
    }
    bpe = CharBPE(merges=80).fit(list(samples.values()))
    rows = [[language, len(text), len(text.split()), len(bpe.encode(text)), len(text.encode("utf-8"))] for language, text in samples.items()]
    paths = [ctx.csv("p06_language_tokenization.csv", ["language", "characters", "words", "char_bpe_tokens", "utf8_bytes"], rows)]
    _save_bars(ctx, "p06_tokens_by_language.png", [r[0] for r in rows], [r[3] for r in rows], "Tokens in short sample")
    paths.append(ctx.path("p06_tokens_by_language.png"))
    print_result(6, "Tokenization and compression across languages", paths)


def p07(ctx: Context) -> None:
    text = "attention helps the model connect words"
    bpe = _bpe()
    pieces = bpe.encode(text)
    char_ids = {char: i for i, char in enumerate(sorted(set(text)))}
    bpe_ids = {piece: i for i, piece in enumerate(sorted(set(pieces)))}
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append([piece, cursor, cursor + len(piece), bpe_ids[piece]])
        cursor += len(piece)
    rows = [[piece, start, end, token_id, [char_ids[c] for c in text[start:end]]] for piece, start, end, token_id in offsets]
    paths = [ctx.csv("p07_token_alignment.csv", ["bpe_piece", "char_start", "char_end", "bpe_id", "char_ids"], rows)]
    ctx.json("p07_round_trip.json", {"source": text, "reconstructed": CharBPE.decode(pieces), "bpe_tokens": pieces})
    paths.append(ctx.path("p07_round_trip.json"))
    print_result(7, "Translation and alignment between tokenizers", paths)


PROJECTS = {1: p01, 2: p02, 3: p03, 4: p04, 5: p05, 6: p06, 7: p07}
