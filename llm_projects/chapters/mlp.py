from __future__ import annotations

import numpy as np

from ..core import Context, apply_final_norm, bundle, cosine, distributional_embeddings, print_result, token_category


def _gelu(x: np.ndarray) -> np.ndarray:
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def p42(ctx: Context) -> None:
    model = bundle(ctx); rows = []
    for layer, (w1, w2) in enumerate(zip(model["w1_weights"], model["w2_weights"])):
        activation = model["mlp_act"][layer].ravel()
        rows.append([layer, float(np.linalg.norm(w1)), float(np.linalg.norm(w2)), float(activation.mean()), float(activation.std()), float(np.mean(activation == 0))])
    path = ctx.csv("p42_mlp_weight_activation_stats.csv", ["layer", "input_weight_norm", "output_weight_norm", "activation_mean", "activation_std", "zero_fraction"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[4] for r in rows], marker="o", label="activation std")
        ax.plot([r[0] for r in rows], [r[5] for r in rows], marker="s", label="exact zero fraction")
        ax.set_xlabel("Layer"); ax.legend()
    ctx.figure("p42_mlp_activation_profile.png", draw)
    print_result(42, "MLP weights and activations", [path, ctx.path("p42_mlp_activation_profile.png")])


def p43(ctx: Context) -> None:
    model = bundle(ctx); layer = len(model["mlp_pre"]) - 1
    pre = model["mlp_pre"][layer].ravel(); post = model["mlp_act"][layer].ravel()
    path = ctx.csv("p43_mlp_progression.csv", ["pre_activation", "after_gelu"], [[float(a), float(b)] for a, b in zip(pre[::8], post[::8])])
    def draw(fig, ax):
        ax.scatter(pre[::4], post[::4], s=9, alpha=.35)
        ax.set_xlabel("Before GELU"); ax.set_ylabel("After GELU")
    ctx.figure("p43_mlp_pre_vs_post.png", draw)
    print_result(43, "Progression through MLP projections", [path, ctx.path("p43_mlp_pre_vs_post.png")])


def p44(ctx: Context) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    model = bundle(ctx); layer = len(model["mlp_act"]) - 1
    features, labels = [], []
    for batch, ids in enumerate(model["tokens"]):
        for position, token_id in enumerate(ids):
            category = model["token_categories"][batch][position]
            if category in {"noun", "verb"}:
                features.append(model["mlp_act"][layer, batch, position])
                labels.append(int(category == "verb"))
    x, y = np.asarray(features), np.asarray(labels)
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=ctx.seed)
    scores = cross_val_score(clf, x, y, cv=cv)
    clf.fit(x, y)
    rows = [["noun-vs-verb", float(scores.mean()), float(scores.std()), len(y), int(np.sum(y))]]
    path = ctx.csv("p44_grammar_probe.csv", ["classification", "mean_accuracy", "std_accuracy", "examples", "verb_examples"], rows)
    ctx.json("p44_probe_weights.json", {"strongest_neurons": np.argsort(np.abs(clf.coef_[0]))[-10:][::-1].tolist(), "coefficients": clf.coef_[0].tolist()})
    print_result(44, "Grammar tuning in MLP projections", [path, ctx.path("p44_probe_weights.json")])


def p45(ctx: Context) -> None:
    from sklearn.feature_selection import mutual_info_classif
    model = bundle(ctx); states = model["hidden_states"][-1].reshape(-1, model["hidden_states"].shape[-1])
    positions = np.tile(np.arange(model["tokens"].shape[1]), model["tokens"].shape[0])
    bins = np.digitize(positions, [4, 8])
    rows = []
    for p in (1, 2, 4):
        left, right = states[::2], states[1::2]
        d = np.sum(np.abs(left - right) ** p, axis=1) ** (1 / p)
        target = bins[::2][:len(d)]
        mi = float(np.mean(mutual_info_classif(d.reshape(-1, 1), target, discrete_features=False, random_state=ctx.seed)))
        rows.append([p, float(np.mean(d)), float(np.std(d)), mi])
    path = ctx.csv("p45_minkowski_mi.csv", ["p", "mean_distance", "std_distance", "distance_position_mutual_information"], rows)
    def draw(fig, ax):
        ax.plot([r[0] for r in rows], [r[3] for r in rows], marker="o")
        ax.set_xlabel("Minkowski p"); ax.set_ylabel("Mutual information (nats, estimate)")
    ctx.figure("p45_minkowski_mi.png", draw)
    print_result(45, "Minkowski distance and mutual information", [path, ctx.path("p45_minkowski_mi.png")])


def p46(ctx: Context) -> None:
    from scipy.stats import ttest_ind
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    model = bundle(ctx); layer = len(model["mlp_act"]) - 1
    values, labels = [], []
    for batch, ids in enumerate(model["tokens"]):
        for position, token_id in enumerate(ids):
            category = model["token_categories"][batch][position]
            if category == "positive":
                values.append(model["mlp_act"][layer, batch, position]); labels.append(1)
            elif category == "negative":
                values.append(model["mlp_act"][layer, batch, position]); labels.append(0)
    x, y = np.asarray(values), np.asarray(labels)
    t, p = ttest_ind(x[y == 1], x[y == 0], axis=0, equal_var=False, nan_policy="omit")
    p = np.nan_to_num(p, nan=1.0)
    selected = np.argsort(p)[:min(12, x.shape[1])]
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=ctx.seed)
    baseline = cross_val_score(clf, x, y, cv=cv)
    lesioned = x.copy(); lesioned[:, selected] = 0
    after = cross_val_score(clf, lesioned, y, cv=cv)
    path = ctx.csv("p46_statistics_based_lesion.csv", ["condition", "mean_accuracy", "std_accuracy", "neurons_zeroed"], [["before", float(baseline.mean()), float(baseline.std()), 0], ["after", float(after.mean()), float(after.std()), len(selected)]])
    ctx.json("p46_selected_neurons.json", {"selected_indices": selected.tolist(), "p_values": p[selected].tolist(), "t_statistics": t[selected].tolist()})
    print_result(46, "Statistics-guided MLP neuron lesion", [path, ctx.path("p46_selected_neurons.json")])


def p47(ctx: Context) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    model = bundle(ctx); layer = len(model["mlp_act"]) - 1
    x, y = [], []
    for batch, ids in enumerate(model["tokens"]):
        for pos, token_id in enumerate(ids):
            category = model["token_categories"][batch][pos]
            if category in {"noun", "verb"}:
                x.append(model["mlp_act"][layer, batch, pos]); y.append(int(category == "verb"))
    x, y = np.asarray(x), np.asarray(y)
    method = "scikit-learn HistGradientBoosting fallback"
    try:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=60, max_depth=2, learning_rate=.08, subsample=.9, colsample_bytree=.8, eval_metric="logloss", n_jobs=1, random_state=ctx.seed)
        method = "XGBoost"
    except ImportError:
        estimator = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, random_state=ctx.seed)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=ctx.seed)
    scores = cross_val_score(estimator, x, y, cv=cv)
    path = ctx.csv("p47_boosted_probe.csv", ["method", "mean_accuracy", "std_accuracy", "examples"], [[method, float(scores.mean()), float(scores.std()), len(y)]])
    print_result(47, "Supervised boosted-tree probe", [path])


def p48(ctx: Context) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    model = bundle(ctx)
    vocab, vectors, _ = distributional_embeddings(); index = {word: i for i, word in enumerate(vocab)}
    pairs = [
        ("the model can answer questions", 1), ("the model cannot answer every question", 0),
        ("a decoder can predict another token", 1), ("a decoder cannot see future tokens", 0),
        ("the assistant can explain uncertainty", 1), ("the assistant cannot guarantee truth", 0),
        ("a small network can learn this pattern", 1), ("a small network cannot store every fact", 0),
        ("the system can produce a useful summary", 1), ("the system cannot reason in every case", 0),
        ("the classifier can detect a word category", 1), ("the classifier cannot prove causation", 0),
    ]
    w1, w2 = model["w1_weights"][-1], model["w2_weights"][-1]
    x, y = [], []
    for text, label in pairs:
        if model["kind"] == "huggingface_gpt2":
            ids = model["tokenizer"](text, return_tensors="np")["input_ids"][0]
            vector = model["embeddings"][ids].mean(axis=0)
        else:
            ids = [index[word] for word in text.split() if word in index]
            vector = vectors[ids].mean(axis=0)
        x.append(_gelu(vector @ w1) @ w2)
        y.append(label)
    x, y = np.asarray(x), np.asarray(y)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, x, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=ctx.seed))
    path = ctx.csv("p48_can_cannot_probe.csv", ["mean_accuracy", "std_accuracy", "examples"], [[float(scores.mean()), float(scores.std()), len(y)]])
    print_result(48, "Can vs. cannot classification with logistic regression", [path])


def p49(ctx: Context) -> None:
    model = bundle(ctx); layer = len(model["mlp_act"]) - 1
    activation = model["mlp_act"][layer].copy()
    valid = model["attention_mask"]
    median = np.median(activation[valid], axis=0)
    importance = np.mean(np.abs(activation[valid] - median), axis=0)
    order = np.argsort(importance)[::-1]
    residual = model["hidden_states"][-2] + model["attention_updates"][-1]
    positions = model["last_positions"]
    batches = np.arange(len(positions))
    target = model["logits"][batches, positions].argmax(axis=-1)
    rows = []
    for count in sorted(set((0, 1, 2, 4, 8, 16, 32, activation.shape[-1]))):
        changed = activation.copy()
        if count:
            changed[..., order[:count]] = median[order[:count]]
        output = residual + (changed @ model["w2_weights"][layer]) * model["mlp_scale"] + model["mlp_biases"][layer]
        logits = np.stack([apply_final_norm(model, output[b, pos]) @ model["unembedding"] for b, pos in enumerate(positions)])
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True)); probs /= probs.sum(axis=-1, keepdims=True)
        confidence = float(np.mean(probs[batches, target]))
        rows.append([count, count / activation.shape[-1], confidence])
    path = ctx.csv("p49_median_replacement.csv", ["neurons_replaced", "fraction_replaced", "mean_input_token_probability"], rows)
    def draw(fig, ax):
        ax.plot([r[1] for r in rows], [r[2] for r in rows], marker="o")
        ax.set_xlabel("Fraction of neurons median-replaced"); ax.set_ylabel("Input-token probability proxy")
    ctx.figure("p49_successive_median_replacement.png", draw)
    print_result(49, "Successive median replacement of MLP activations", [path, ctx.path("p49_successive_median_replacement.png")])


def p50(ctx: Context) -> None:
    model = bundle(ctx)
    vocab, vectors, _ = distributional_embeddings(); index = {word: i for i, word in enumerate(vocab)}
    item_tokens = {"books": ["books", "helpful"], "coding": ["code", "network"], "mathematics": ["mathematics", "examples"], "language models": ["models", "token"], "attention": ["attention", "sentence"], "data science": ["data", "scientist"], "safety": ["safe", "trust"]}
    w1, w2 = model["w1_weights"][-1], model["w2_weights"][-1]
    item_vecs = {}
    for item, words in item_tokens.items():
        if model["kind"] == "huggingface_gpt2":
            ids = [model["tokenizer"](word, return_tensors="np")["input_ids"][0] for word in words]
            present = [model["embeddings"][token_ids].mean(axis=0) for token_ids in ids if len(token_ids)]
        else:
            present = [vectors[index[word]] for word in words if word in index]
        base = np.mean(present, axis=0)
        item_vecs[item] = _gelu(base @ w1) @ w2
    profiles = {"reader": ["books", "language models"], "builder": ["coding", "mathematics"], "analyst": ["data science", "attention"]}
    rows = []
    for user, liked in profiles.items():
        profile = np.mean([item_vecs[item] for item in liked], axis=0)
        ranked = sorted(((cosine(profile, vector), item) for item, vector in item_vecs.items() if item not in liked), reverse=True)
        for rank, (score, item) in enumerate(ranked, 1):
            rows.append([user, rank, item, score, ", ".join(liked)])
    path = ctx.csv("p50_projection_recommendations.csv", ["profile", "rank", "recommended_item", "cosine_score", "liked_items"], rows)
    print_result(50, "Recommendation from MLP-projected item vectors", [path])


PROJECTS = {42: p42, 43: p43, 44: p44, 45: p45, 46: p46, 47: p47, 48: p48, 49: p49, 50: p50}
