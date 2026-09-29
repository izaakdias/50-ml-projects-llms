from __future__ import annotations

import numpy as np

from ..core import Context, apply_final_norm, bundle, cosine, cosine_matrix, print_result, softmax, token_category


def p33(ctx: Context) -> None:
    model = bundle(ctx); rows = []
    for layer, matrices in enumerate(model["qkv_weights"]):
        for name, matrix in zip(("Q", "K", "V"), matrices):
            rows.append([layer, name, float(np.mean(matrix)), float(np.std(matrix)), float(np.linalg.norm(matrix)), float(np.max(np.abs(matrix)))])
    path = ctx.csv("p33_qkv_weight_statistics.csv", ["layer", "projection", "mean", "std", "frobenius_norm", "max_abs"], rows)
    def draw(fig, ax):
        for projection in ("Q", "K", "V"):
            group = [r for r in rows if r[1] == projection]
            ax.plot([r[0] for r in group], [r[4] for r in group], marker="o", label=projection)
        ax.set_xlabel("Layer"); ax.set_ylabel("Weight Frobenius norm"); ax.legend()
    ctx.figure("p33_qkv_weights.png", draw)
    print_result(33, "QKV weight characteristics", [path, ctx.path("p33_qkv_weights.png")])


def p34(ctx: Context) -> None:
    model = bundle(ctx); rows = []
    for name in ("q", "k", "v"):
        values = model[name]
        for layer in range(values.shape[0]):
            row = values[layer].ravel()
            rows.append([layer, name.upper(), float(row.mean()), float(row.std()), float(np.quantile(row, .05)), float(np.quantile(row, .95))])
    path = ctx.csv("p34_qkv_activation_statistics.csv", ["layer", "activation", "mean", "std", "p05", "p95"], rows)
    def draw(fig, ax):
        for name in ("Q", "K", "V"):
            group = [r for r in rows if r[1] == name]
            ax.plot([r[0] for r in group], [r[3] for r in group], marker="o", label=name)
        ax.set_xlabel("Layer"); ax.set_ylabel("Activation standard deviation"); ax.legend()
    ctx.figure("p34_qkv_activation_spread.png", draw)
    print_result(34, "QKV activation characteristics", [path, ctx.path("p34_qkv_activation_spread.png")])


def p35(ctx: Context) -> None:
    model = bundle(ctx); layer, head, batch = 2, 0, 0
    raw = model["attention_scores"][layer, batch, head]
    weights = model["attention"][layer, batch, head]
    valid_length = int(model["attention_mask"][batch].sum())
    raw = raw[:valid_length, :valid_length]; weights = weights[:valid_length, :valid_length]
    tokens = model["token_strings"][batch][:valid_length]
    path = ctx.csv("p35_attention_matrices.csv", ["query_position", "key_position", "raw_score", "softmax_weight"], [[i, j, float(raw[i, j]), float(weights[i, j])] for i in range(len(tokens)) for j in range(len(tokens))])
    def draw(fig, ax):
        im = ax.imshow(weights, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(tokens)), tokens, rotation=70, fontsize=7)
        ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
        ax.set_xlabel("Key token"); ax.set_ylabel("Query token"); fig.colorbar(im, ax=ax, label="Attention probability")
    ctx.figure("p35_attention_heatmap.png", draw)
    print_result(35, "Raw and softmax attention scores", [path, ctx.path("p35_attention_heatmap.png")])


def p36(ctx: Context) -> None:
    model = bundle(ctx)
    magnitude_by_position = np.linalg.norm(model["attention_contribs"], axis=-1)
    mask = model["attention_mask"][None, :, None, :]
    magnitudes = (magnitude_by_position * mask).sum(axis=(1, 3)) / np.maximum(mask.sum(axis=(1, 3)), 1)
    rows = [[layer, head, float(magnitudes[layer, head])] for layer in range(magnitudes.shape[0]) for head in range(magnitudes.shape[1])]
    path = ctx.csv("p36_attention_adjustment_magnitudes.csv", ["layer", "head", "mean_update_norm"], rows)
    def draw(fig, ax):
        im = ax.imshow(magnitudes, aspect="auto", cmap="magma")
        ax.set_xlabel("Attention head"); ax.set_ylabel("Layer"); fig.colorbar(im, ax=ax, label="Mean update norm")
    ctx.figure("p36_attention_adjustment_heatmap.png", draw)
    print_result(36, "Magnitudes of attention adjustments", [path, ctx.path("p36_attention_adjustment_heatmap.png")])


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def p37(ctx: Context) -> None:
    model = bundle(ctx); positions = model["last_positions"]
    batches = np.arange(len(positions))
    baseline = softmax(model["logits"][batches, positions, :], axis=-1)
    rows = []
    for layer in range(model["attention_contribs"].shape[0]):
        for head in range(model["attention_contribs"].shape[2]):
            altered = np.stack([model["final_pre_norm"][b, pos] - model["attention_contribs"][layer, b, head, pos] for b, pos in enumerate(positions)])
            probs = softmax(apply_final_norm(model, altered) @ model["unembedding"], axis=-1)
            kl = float(np.mean([_kl(p, q) for p, q in zip(baseline, probs)]))
            rows.append([layer, head, kl, float(np.mean(np.argmax(baseline, axis=-1) != np.argmax(probs, axis=-1)))])
    path = ctx.csv("p37_head_prediction_kl.csv", ["layer", "head", "mean_kl_to_baseline", "top_token_change_rate"], rows)
    matrix = np.zeros((model["attention_contribs"].shape[0], model["attention_contribs"].shape[2]))
    for layer, head, kl, _ in rows: matrix[layer, head] = kl
    def draw(fig, ax):
        im = ax.imshow(matrix, aspect="auto", cmap="inferno")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); fig.colorbar(im, ax=ax, label="KL divergence")
    ctx.figure("p37_prediction_kl_heatmap.png", draw)
    print_result(37, "Token prediction changes and attention-head KL", [path, ctx.path("p37_prediction_kl_heatmap.png")])


def p38(ctx: Context) -> None:
    model = bundle(ctx); vocab = model["vocab"]
    token_ids = model["tokens"].ravel()
    labels = np.asarray([vocab[i] for i in token_ids])
    word_categories = np.asarray([category for row in model["token_categories"] for category in row])
    categories = np.asarray([category if category in {"positive", "negative"} else "other" for category in word_categories])
    pairs = np.triu_indices(len(labels), k=1)
    category_different = (categories[pairs[0]] != categories[pairs[1]]).astype(float)
    rows = []
    for layer, states in enumerate(model["hidden_states"]):
        vectors = states.reshape(-1, states.shape[-1])
        sim = cosine_matrix(vectors)
        distance = 1 - sim[pairs]
        rho = float(np.corrcoef(distance, category_different)[0, 1]) if np.std(category_different) else 0.0
        groups = []
        for cat in ("positive", "negative"):
            idx = np.flatnonzero(categories == cat)
            if len(idx) > 1:
                sub = sim[np.ix_(idx, idx)]
                groups.append(float(np.mean(sub[np.triu_indices(len(idx), k=1)])))
        selectivity = float(np.mean(groups)) if groups else 0.0
        rows.append([layer, rho, selectivity])
    path = ctx.csv("p38_layer_rsa_selectivity.csv", ["layer", "category_rdm_correlation", "positive_negative_within_cosine"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label="RDM correlation")
        ax.plot([r[0] for r in rows], [r[2] for r in rows], marker="s", label="within-category cosine")
        ax.set_xlabel("Layer"); ax.set_ylabel("Summary statistic"); ax.legend()
    ctx.figure("p38_rsa_and_selectivity.png", draw)
    print_result(38, "Layer-wise RSA and category selectivity", [path, ctx.path("p38_rsa_and_selectivity.png")])


def p39(ctx: Context) -> None:
    model = bundle(ctx); vocab = model["vocab"]
    valid = model["attention_mask"]
    ids = model["tokens"][valid]
    freq = np.bincount(ids, minlength=len(vocab))
    mean_attention = model["attention"].mean(axis=(0, 2))
    incoming = np.zeros(len(vocab)); update = np.zeros(len(vocab)); appearances = np.zeros(len(vocab))
    for batch in range(model["tokens"].shape[0]):
        valid_queries = np.flatnonzero(valid[batch])
        for position in valid_queries:
            token_id = int(model["tokens"][batch, position])
            incoming[token_id] += mean_attention[batch, valid_queries, position].sum()
            update[token_id] += np.linalg.norm(model["attention_contribs"][:, batch, :, position], axis=-1).mean()
            appearances[token_id] += 1
    incoming /= np.maximum(appearances, 1); update /= np.maximum(appearances, 1)
    qk = model["attention_scores"].mean(axis=(0, 1, 2)).mean(axis=0)
    rows = [[vocab[i], int(freq[i]), float(incoming[i]), float(update[i])] for i in np.argsort(freq)[::-1][:30]]
    path = ctx.csv("p39_token_frequency_attention.csv", ["token", "frequency", "mean_incoming_attention_proxy", "mean_attention_update_norm_proxy"], rows)
    def draw(fig, ax):
        x = np.asarray([r[1] for r in rows]); y = np.asarray([r[3] for r in rows])
        ax.scatter(x, y, alpha=.75)
        for r in rows[:10]: ax.annotate(r[0], (r[1], r[3]), fontsize=7)
        ax.set_xlabel("Token frequency in toy batch"); ax.set_ylabel("Layer-averaged attention update norm")
    ctx.figure("p39_frequency_vs_attention.png", draw)
    ctx.json("p39_qk_summary.json", {"mean_qk_score": float(np.mean(qk)), "qk_score_std": float(np.std(qk)), "note": "QK and update summaries come from the local random decoder demo."})
    print_result(39, "Token frequency, attention adjustment and QK scores", [path, ctx.path("p39_frequency_vs_attention.png"), ctx.path("p39_qk_summary.json")])


def p40(ctx: Context) -> None:
    model = bundle(ctx); positions = model["last_positions"]
    batches = np.arange(len(positions))
    baseline = model["logits"][batches, positions]
    baseline_prob = softmax(baseline)
    base_state = np.stack([model["final_pre_norm"][b, pos] for b, pos in enumerate(positions)])
    rows = []
    for layer in range(model["attention_contribs"].shape[0]):
        for head in range(model["attention_contribs"].shape[2]):
            modified_pre = np.stack([base_state[b] - model["attention_contribs"][layer, b, head, pos] for b, pos in enumerate(positions)])
            modified = apply_final_norm(model, modified_pre)
            probs = softmax(modified @ model["unembedding"])
            effect = float(np.mean([_kl(p, q) for p, q in zip(baseline_prob, probs)]))
            rows.append([layer, head, effect, float(np.linalg.norm(modified - apply_final_norm(model, base_state)) / len(modified))])
    path = ctx.csv("p40_head_silencing_effects.csv", ["layer", "head", "mean_output_kl", "mean_last_state_change"], rows)
    matrix = np.zeros((model["attention_contribs"].shape[0], model["attention_contribs"].shape[2]))
    for layer, head, effect, _ in rows: matrix[layer, head] = effect
    def draw(fig, ax):
        im = ax.imshow(matrix, aspect="auto", cmap="plasma")
        ax.set_xlabel("Silenced head"); ax.set_ylabel("Layer"); fig.colorbar(im, ax=ax, label="Output KL proxy")
    ctx.figure("p40_head_silencing.png", draw)
    print_result(40, "Downstream impact proxy for attention-head silencing", [path, ctx.path("p40_head_silencing.png")])


def p41(ctx: Context) -> None:
    model = bundle(ctx); contributions = model["attention_contribs"]
    target, foil = [int(value) for value in model["ioi_target_ids"]]
    receiver_pos, donor_pos = [int(value) for value in model["last_positions"][:2]]
    rows = []
    for layer in range(contributions.shape[0]):
        for head in range(contributions.shape[2]):
            receiver_pre = model["final_pre_norm"][0, receiver_pos]
            donor_head = contributions[layer, 1, head, donor_pos]
            receiver = apply_final_norm(model, receiver_pre)
            base_margin = float((receiver @ model["unembedding"])[target] - (receiver @ model["unembedding"])[foil])
            patched = apply_final_norm(model, receiver_pre - contributions[layer, 0, head, receiver_pos] + donor_head)
            patched_margin = float((patched @ model["unembedding"])[target] - (patched @ model["unembedding"])[foil])
            rows.append([layer, head, base_margin, patched_margin, patched_margin - base_margin])
    path = ctx.csv("p41_head_patch_effects.csv", ["layer", "head", "base_margin", "patched_margin", "patch_effect"], rows)
    matrix = np.zeros((contributions.shape[0], contributions.shape[2]))
    for layer, head, _, _, effect in rows: matrix[layer, head] = effect
    def draw(fig, ax):
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
        ax.set_xlabel("Patched head"); ax.set_ylabel("Layer"); fig.colorbar(im, ax=ax, label="Change in target margin")
    ctx.figure("p41_head_patching_heatmap.png", draw)
    print_result(41, "IOI prompt-pair attention-head patching proxy", [path, ctx.path("p41_head_patching_heatmap.png")])


PROJECTS = {33: p33, 34: p34, 35: p35, 36: p36, 37: p37, 38: p38, 39: p39, 40: p40, 41: p41}
