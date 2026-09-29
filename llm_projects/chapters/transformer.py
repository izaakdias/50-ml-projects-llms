from __future__ import annotations

import numpy as np

from ..core import Context, apply_final_norm, logit_lens_state, bundle, cosine, cosine_matrix, distributional_embeddings, print_result, softmax, token_category


def p24(ctx: Context) -> None:
    model = bundle(ctx); states = model["hidden_states"][:, 0]
    valid_length = int(model["attention_mask"][0].sum())
    within, between = [], []
    for layer, state in enumerate(states):
        state = state[:valid_length]
        sim = cosine_matrix(state)
        within.append(float(np.mean(sim[np.triu_indices(len(state), 1)])))
        if layer:
            prev = states[layer - 1]
            between.append(float(np.mean([cosine(a, b) for a, b in zip(state, prev)])))
    rows = [[i, within[i], np.nan if i == 0 else between[i - 1]] for i in range(len(within))]
    path = ctx.csv("p24_layer_cosines.csv", ["layer", "within_layer_mean", "same_position_vs_previous_layer"], rows)
    def draw(fig, ax):
        ax.plot(range(len(within)), within, marker="o", label="within layer / token positions")
        ax.plot(range(1, len(within)), between, marker="s", label="same position / prior layer")
        ax.set_xlabel("Residual-stream layer"); ax.set_ylabel("Mean cosine similarity"); ax.legend()
    ctx.figure("p24_layer_cosines.png", draw)
    print_result(24, "Cosine similarities within and across layers", [path, ctx.path("p24_layer_cosines.png")])


def p25(ctx: Context) -> None:
    model = bundle(ctx); vocab = model["vocab"]
    rows = []
    for layer, state in enumerate(model["hidden_states"]):
        valid = model["attention_mask"].reshape(-1)
        vectors = state.reshape(-1, state.shape[-1])[valid]
        labels = np.asarray([category for row in model["token_categories"] for category in row])[valid]
        categories = sorted(set(labels) - {"other"})
        for category in categories:
            idx = np.array([i for i, label in enumerate(labels) if label == category])
            rest = np.array([i for i, label in enumerate(labels) if label != category])
            if len(idx) > 1 and len(rest):
                own = cosine_matrix(vectors[idx]); own_score = float(np.mean(own[np.triu_indices(len(idx), 1)])) if len(idx) > 1 else 0.0
                cross = float(np.mean([cosine(vectors[i], vectors[j]) for i in idx for j in rest]))
                rows.append([layer, category, own_score, cross, own_score - cross])
    path = ctx.csv("p25_category_selectivity.csv", ["layer", "category", "within_category_cosine", "category_to_rest_cosine", "selectivity_gap"], rows)
    def draw(fig, ax):
        for category in sorted({row[1] for row in rows}):
            values = [row[4] for row in rows if row[1] == category]
            ax.plot(range(len(values)), values, marker="o", label=category)
        ax.set_xlabel("Layer"); ax.set_ylabel("Within minus cross-category cosine"); ax.legend(ncol=2)
    ctx.figure("p25_category_selectivity.png", draw)
    print_result(25, "Category selectivity by cosine similarity", [path, ctx.path("p25_category_selectivity.png")])


def p26(ctx: Context) -> None:
    model = bundle(ctx); states = model["hidden_states"][:, 0]
    valid_length = int(model["attention_mask"][0].sum())
    states = states[:, :valid_length]
    updates = states[1:] - states[:-1]
    rows = []
    for layer, (before, update, after) in enumerate(zip(states[:-1], updates, states[1:])):
        rows.append([layer, float(np.linalg.norm(update)), float(np.linalg.norm(before)), float(np.linalg.norm(after)), float(np.mean(np.sum(before * update, axis=-1)))])
    path = ctx.csv("p26_residual_updates.csv", ["layer", "update_norm", "previous_norm", "current_norm", "mean_residual_dot_update"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label="adjustment")
        ax.plot([r[0] for r in rows], [r[2] for r in rows], marker="s", label="previous state")
        ax.set_xlabel("Layer"); ax.set_ylabel("Vector norm"); ax.legend()
    ctx.figure("p26_residual_decomposition.png", draw)
    print_result(26, "Layer state as residual plus adjustment", [path, ctx.path("p26_residual_decomposition.png")])


def p27(ctx: Context) -> None:
    model = bundle(ctx); states = model["hidden_states"][:, 0]
    valid_length = int(model["attention_mask"][0].sum())
    states = states[:, :valid_length]
    rows = []
    for layer, state in enumerate(states):
        noise = ctx.rng.normal(size=state.shape)
        noise *= np.std(state) * .25
        for scale in (0.5, 1.0, 1.5):
            altered = state * scale + noise
            rows.append([layer, scale, float(np.mean(np.linalg.norm(altered - state, axis=-1))), float(np.mean([cosine(a, b) for a, b in zip(state, altered)]))])
    path = ctx.csv("p27_noise_scaling_sensitivity.csv", ["layer", "scale", "mean_change_norm", "mean_cosine_to_original"], rows)
    def draw(fig, ax):
        for scale in (0.5, 1.0, 1.5):
            group = [r for r in rows if r[1] == scale]
            ax.plot([r[0] for r in group], [r[3] for r in group], marker="o", label=f"scale {scale}")
        ax.set_xlabel("Layer"); ax.set_ylabel("Cosine to unperturbed state"); ax.legend()
    ctx.figure("p27_noise_and_scaling.png", draw)
    print_result(27, "Effect of layer-specific noise and scaling", [path, ctx.path("p27_noise_and_scaling.png")])


def p28(ctx: Context) -> None:
    model = bundle(ctx); states = model["hidden_states"]
    valid = model["attention_mask"].reshape(-1)
    rows = []
    for layer, state in enumerate(states):
        x = state.reshape(-1, state.shape[-1])[valid]; x = x - x.mean(axis=0, keepdims=True)
        eigen = np.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)); eigen = np.maximum(eigen, 0)[::-1]
        pr = float(eigen.sum() ** 2 / max(np.sum(eigen ** 2), 1e-12))
        rows.append([layer, pr, float(eigen[0] / max(eigen.sum(), 1e-12))])
    path = ctx.csv("p28_effective_dimension.csv", ["layer", "participation_ratio", "top_component_variance_share"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o")
        ax.set_xlabel("Layer"); ax.set_ylabel("Effective dimension (participation ratio)")
    ctx.figure("p28_effective_dimension.png", draw)
    print_result(28, "Effective dimensionality of hidden layers", [path, ctx.path("p28_effective_dimension.png")])


def p29(ctx: Context) -> None:
    from sklearn.decomposition import PCA
    model = bundle(ctx); states = model["hidden_states"]
    valid = model["attention_mask"].reshape(-1)
    layer = states[-1].reshape(-1, states.shape[-1])[valid]
    projected = PCA(n_components=2, random_state=ctx.seed).fit_transform(layer)
    token_ids = model["tokens"].reshape(-1)[valid]; labels = [model["vocab"][i] for i in token_ids]
    path = ctx.csv("p29_pca_coordinates.csv", ["pc1", "pc2", "token"], [[float(x), float(y), label] for (x, y), label in zip(projected, labels)])
    def draw(fig, ax):
        ax.scatter(projected[:, 0], projected[:, 1], s=28, alpha=.75)
        for (x, y), label in zip(projected, labels):
            ax.annotate(label, (x, y), fontsize=6, alpha=.8)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title("Offline toy decoder states")
    ctx.figure("p29_hidden_state_pca.png", draw)
    print_result(29, "Hidden-state dimensionality reduction", [path, ctx.path("p29_hidden_state_pca.png")])


def p30(ctx: Context) -> None:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    model = bundle(ctx); x, y = [], []
    for batch, ids in enumerate(model["tokens"]):
        for position, token_id in enumerate(ids):
            category = model["token_categories"][batch][position]
            label = 1 if category == "positive" else 0 if category == "negative" else -1
            if label >= 0:
                x.append(model["hidden_states"][-1, batch, position]); y.append(label)
    x, y = np.asarray(x), np.asarray(y)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=ctx.seed)
    scores = cross_val_score(clf, x, y, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=ctx.seed))
    clf.fit(x, y)
    rows = [["positive-vs-negative token state", float(scores.mean()), float(scores.std()), len(y)]]
    path = ctx.csv("p30_toy_sentiment_probe.csv", ["task", "mean_two_fold_accuracy", "std_accuracy", "examples"], rows)
    print_result(30, "Sentiment decision-tree probe", [path])


def p31(ctx: Context) -> None:
    model = bundle(ctx); token_ids = model["tokens"]
    rows = []
    for layer, state in enumerate(model["hidden_states"]):
        lens_state = logit_lens_state(model, layer, state)
        probs = softmax(lens_state @ model["unembedding"])
        chosen = probs.argmax(axis=-1)
        for batch, position in enumerate(model["last_positions"]):
            rows.append([layer, batch, model["vocab"][int(chosen[batch, position])], float(probs[batch, position].max()), int(token_ids[batch, position])])
    path = ctx.csv("p31_logit_lens.csv", ["layer", "sequence", "top_token", "top_probability", "input_token_id"], rows)
    means = [np.mean([r[3] for r in rows if r[0] == layer]) for layer in range(len(model["hidden_states"]))]
    def draw(fig, ax):
        ax.plot(range(len(means)), means, marker="o")
        ax.set_xlabel("Layer read by shared unembedding"); ax.set_ylabel("Mean top-token probability")
    ctx.figure("p31_logit_lens.png", draw)
    print_result(31, "Logit lens across layers", [path, ctx.path("p31_logit_lens.png")])


def p32(ctx: Context) -> None:
    model = bundle(ctx); states = model["hidden_states"]
    target, foil = [int(value) for value in model["ioi_target_ids"]]
    receiver_pos, donor_pos = [int(value) for value in model["last_positions"][:2]]
    receiver_subject, donor_subject = [int(value) for value in model["subject_positions"][:2]]
    rows = []
    for layer, state in enumerate(states):
        receiver = logit_lens_state(model, layer, state[0, receiver_pos])
        donor = logit_lens_state(model, layer, state[1, donor_pos])
        receiver_margin = float((receiver @ model["unembedding"])[target] - (receiver @ model["unembedding"])[foil])
        donor_margin = float((donor @ model["unembedding"])[target] - (donor @ model["unembedding"])[foil])
        subject_delta = state[1, donor_subject] - state[0, receiver_subject]
        patched = logit_lens_state(model, layer, state[0, receiver_pos] + subject_delta)
        patched_margin = float((patched @ model["unembedding"])[target] - (patched @ model["unembedding"])[foil])
        rows.append([layer, receiver_margin, donor_margin, patched_margin, patched_margin - receiver_margin])
    path = ctx.csv("p32_hidden_state_patch_demo.csv", ["layer", "maya_prompt_margin", "leo_prompt_margin", "linearized_patch_margin", "patch_effect_proxy"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[4] for r in rows], marker="o")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xlabel("Layer"); ax.set_ylabel("Change in target-minus-foil readout")
        ax.set_title("IOI prompt pair; linearized patch proxy")
    ctx.figure("p32_hidden_state_patching.png", draw)
    print_result(32, "Hidden-state patching demonstration", [path, ctx.path("p32_hidden_state_patching.png")])


PROJECTS = {24: p24, 25: p25, 26: p26, 27: p27, 28: p28, 29: p29, 30: p30, 31: p31, 32: p32}
