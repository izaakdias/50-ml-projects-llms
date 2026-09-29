from __future__ import annotations

from collections import Counter

import numpy as np

from ..core import Context, cosine, cosine_matrix, corpus_tokens, distributional_embeddings, normalize_rows, numeric_char_vectors, print_result, token_category


def p08(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    sim = cosine_matrix(vectors)
    vals = sim[np.triu_indices(len(vocab), k=1)]
    rows = [[float(x)] for x in vals]
    path = ctx.csv("p08_cosine_similarity_values.csv", ["cosine"], rows)
    def draw(fig, ax):
        ax.hist(vals, bins=24, color="#6285a8", edgecolor="white")
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Word pairs")
    ctx.figure("p08_cosine_distribution.png", draw)
    print_result(8, "Distribution of cosine similarities", [path, ctx.path("p08_cosine_distribution.png")])


def p09(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    lookup = {word: vectors[i] for i, word in enumerate(vocab)}
    sequential = [cosine(lookup[a], lookup[b]) for sent in corpus_tokens() for a, b in zip(sent, sent[1:]) if a in lookup and b in lookup]
    shuffled_tokens = [token for sent in corpus_tokens() for token in sent if token in lookup]
    shuffled = ctx.rng.permutation(shuffled_tokens)
    baseline = [cosine(lookup[a], lookup[b]) for a, b in zip(shuffled, shuffled[1:])]
    rows = [["adjacent", x] for x in sequential] + [["shuffled", x] for x in baseline]
    path = ctx.csv("p09_sequential_cosines.csv", ["condition", "cosine"], rows)
    def draw(fig, ax):
        ax.boxplot([sequential, baseline], tick_labels=["adjacent", "shuffled"])
        ax.set_ylabel("Cosine similarity")
    ctx.figure("p09_sequential_similarity.png", draw)
    print_result(9, "Sequential cosine similarity", [path, ctx.path("p09_sequential_similarity.png")])


def p10(ctx: Context) -> None:
    numbers = list(range(0, 100))
    vectors = numeric_char_vectors(numbers)
    adjacent = np.sum(vectors[:-1] * vectors[1:], axis=1)
    rows = [[n, float(adjacent[n])] for n in range(len(adjacent))]
    path = ctx.csv("p10_number_cosines.csv", ["left_number", "cosine_to_next"], rows)
    def draw(fig, ax):
        ax.plot(numbers[:-1], adjacent, marker=".", linewidth=1)
        ax.axvline(9.5, color="gray", linestyle="--", label="digit-width boundary")
        ax.set_xlabel("Number n (compared with n+1)"); ax.set_ylabel("Cosine similarity"); ax.legend()
    ctx.figure("p10_number_cosines.png", draw)
    print_result(10, "Sequential number cosine similarity", [path, ctx.path("p10_number_cosines.png")])


def p11(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    sim = cosine_matrix(vectors)
    k = 3
    edges = set()
    for i in range(len(vocab)):
        neighbors = np.argsort(sim[i])[-k-1:-1]
        for j in neighbors:
            edges.add(tuple(sorted((i, int(j)))))
    degree = np.zeros(len(vocab), dtype=int)
    for i, j in edges:
        degree[i] += 1; degree[j] += 1
    rows = [[vocab[i], int(degree[i]), ", ".join(vocab[j] for j in np.argsort(sim[i])[-k-1:-1][::-1])] for i in range(len(vocab))]
    path = ctx.csv("p11_similarity_network.csv", ["word", "degree", "nearest_neighbors"], rows)
    def draw(fig, ax):
        order = np.argsort(degree)[-20:]
        ax.barh([vocab[i] for i in order], degree[order], color="#597da0")
        ax.set_xlabel("Degree in mutual top-3 similarity graph")
    ctx.figure("p11_network_degree.png", draw)
    print_result(11, "Network graph of cosine similarities", [path, ctx.path("p11_network_degree.png")])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def p12(ctx: Context) -> None:
    vocab, ppmi_vectors, counts = distributional_embeddings()
    raw_vectors = np.log1p(counts)
    raw_sim = cosine_matrix(raw_vectors)
    ppmi_sim = cosine_matrix(ppmi_vectors)
    keep = np.triu_indices(len(vocab), k=1)
    x, y = raw_sim[keep], ppmi_sim[keep]
    rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    rows = [[float(a), float(b)] for a, b in zip(x, y)]
    path = ctx.csv("p12_rsa_similarity_pairs.csv", ["raw_count_cosine", "ppmi_svd_cosine"], rows)
    def draw(fig, ax):
        ax.scatter(x, y, s=10, alpha=.4)
        ax.set_xlabel("Raw co-occurrence cosine"); ax.set_ylabel("PPMI + SVD cosine")
        ax.set_title(f"Toy representation RSA, rank correlation = {rho:.2f}")
    ctx.figure("p12_rsa_comparison.png", draw)
    ctx.json("p12_rsa_summary.json", {"vocabulary_size": len(vocab), "pair_rank_correlation": rho, "representation_a": "log co-occurrence", "representation_b": "PPMI with truncated SVD"})
    print_result(12, "Representational similarity analysis", [path, ctx.path("p12_rsa_comparison.png"), ctx.path("p12_rsa_summary.json")])


def p13(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    rng = ctx.rng
    pairs = rng.integers(0, len(vocab), size=(1500, 2))
    a, b = vectors[pairs[:, 0]], vectors[pairs[:, 1]]
    cos = np.sum(normalize_rows(a) * normalize_rows(b), axis=1)
    euclidean = np.linalg.norm(a - b, axis=1)
    path = ctx.csv("p13_distance_comparison.csv", ["cosine", "euclidean"], [[float(x), float(y)] for x, y in zip(cos, euclidean)])
    def draw(fig, ax):
        ax.scatter(cos, euclidean, s=10, alpha=.3, color="#b36948")
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Euclidean distance")
    ctx.figure("p13_cosine_vs_distance.png", draw)
    print_result(13, "Word similarity by distance and cosine", [path, ctx.path("p13_cosine_vs_distance.png")])


def p14(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    idx = {word: i for i, word in enumerate(vocab)}
    axes = {"positive – negative": ("good", "bad"), "fast – slow": ("fast", "slow"), "helpful – harmful": ("helpful", "harmful")}
    rows, scores_by_axis = [], {}
    for label, (positive, negative) in axes.items():
        if positive not in idx or negative not in idx:
            continue
        axis = vectors[idx[positive]] - vectors[idx[negative]]
        axis /= max(np.linalg.norm(axis), 1e-12)
        scores = vectors @ axis
        scores_by_axis[label] = scores
        for word, score in zip(vocab, scores):
            rows.append([label, word, float(score)])
    path = ctx.csv("p14_semantic_axis_scores.csv", ["axis", "word", "projection"], rows)
    def draw(fig, ax):
        label, scores = next(iter(scores_by_axis.items()))
        order = np.argsort(scores)
        selected = np.r_[order[:7], order[-7:]]
        ax.barh([vocab[i] for i in selected], scores[selected], color=["#b36d5d" if s < 0 else "#548879" for s in scores[selected]])
        ax.set_title(label); ax.set_xlabel("Projection onto normalized difference vector")
    ctx.figure("p14_semantic_axis.png", draw)
    print_result(14, "Linear semantic axes", [path, ctx.path("p14_semantic_axis.png")])


def p15(ctx: Context) -> None:
    vocab, vectors, _ = distributional_embeddings()
    idx = {word: i for i, word in enumerate(vocab)}
    analogies = [("king", "man", "woman", "queen"), ("paris", "france", "italy", "rome")]
    rows = []
    for a, b, c, expected in analogies:
        if not all(word in idx for word in (a, b, c, expected)):
            continue
        query = vectors[idx[a]] - vectors[idx[b]] + vectors[idx[c]]
        sims = normalize_rows(vectors) @ (query / max(np.linalg.norm(query), 1e-12))
        for word in (a, b, c):
            sims[idx[word]] = -np.inf
        best = np.argsort(sims)[-5:][::-1]
        rank = int(np.where(np.argsort(sims)[::-1] == idx[expected])[0][0] + 1)
        rows.append([f"{a} - {b} + {c}", expected, rank, ", ".join(vocab[i] for i in best)])
    path = ctx.csv("p15_analogy_results.csv", ["query", "expected", "expected_rank", "top_five"], rows)
    print_result(15, "Analogy vectors", [path])


PROJECTS = {8: p08, 9: p09, 10: p10, 11: p11, 12: p12, 13: p13, 14: p14, 15: p15}
