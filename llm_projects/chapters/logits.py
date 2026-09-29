from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..core import CORPUS, Context, corpus_tokens, distributional_embeddings, print_result, softmax, word_tokens


def p16(ctx: Context) -> None:
    logits = ctx.rng.normal(0, 2, size=24)
    temperatures = [0.35, 0.7, 1.0, 1.8]
    probs = [softmax(logits, temperature=t) for t in temperatures]
    entropy = [-float(np.sum(p * np.log(p + 1e-12))) for p in probs]
    paths = [ctx.csv("p16_softmax_by_temperature.csv", ["temperature", "entropy", "top_probability"], [[t, h, float(p.max())] for t, h, p in zip(temperatures, entropy, probs)])]
    def draw(fig, ax):
        x = np.arange(len(logits)); width = .18
        for i, (t, p) in enumerate(zip(temperatures, probs)):
            ax.bar(x + (i - 1.5) * width, p, width, label=f"T={t}")
        ax.set_xlabel("Candidate token id"); ax.set_ylabel("Probability"); ax.legend(ncol=2)
    ctx.figure("p16_softmax_temperature.png", draw)
    paths.append(ctx.path("p16_softmax_temperature.png"))
    print_result(16, "Softmax probability distributions", paths)


def p17(ctx: Context) -> None:
    logits = ctx.rng.normal(size=8)
    probs = softmax(logits, temperature=0.8)
    draws = ctx.rng.choice(np.arange(len(probs)), size=5000, p=probs)
    observed = np.bincount(draws, minlength=len(probs)) / len(draws)
    rows = [[i, float(p), float(o)] for i, (p, o) in enumerate(zip(probs, observed))]
    path = ctx.csv("p17_sampling_frequencies.csv", ["candidate", "softmax_probability", "sampled_frequency"], rows)
    def draw(fig, ax):
        x = np.arange(len(probs)); width = .38
        ax.bar(x - width / 2, probs, width, label="Expected"); ax.bar(x + width / 2, observed, width, label="Observed")
        ax.set_xlabel("Candidate token"); ax.set_ylabel("Probability / frequency"); ax.legend()
    ctx.figure("p17_sampling_demo.png", draw)
    print_result(17, "Probabilistic token selection", [path, ctx.path("p17_sampling_demo.png")])


def _bigram_counts(sentences: list[list[str]]):
    counts = defaultdict(Counter)
    unigram = Counter()
    vocab = set()
    for sentence in sentences:
        words = ["<bos>", *sentence, "<eos>"]
        vocab.update(words)
        unigram.update(words[1:])
        for a, b in zip(words, words[1:]):
            counts[a][b] += 1
    return counts, unigram, sorted(vocab)


def _conditional(counts, vocab, context: str, alpha: float = .1):
    total = sum(counts[context].values()) + alpha * len(vocab)
    return {word: (counts[context][word] + alpha) / total for word in vocab}


def p18(ctx: Context) -> None:
    sentences = corpus_tokens()
    counts, _, vocab = _bigram_counts(sentences)
    examples = [(a, b) for sentence in sentences for a, b in zip(sentence, sentence[1:])]
    top1 = top3 = 0
    rows = []
    for previous, target in examples:
        probs = _conditional(counts, vocab, previous)
        guess = sorted(probs, key=probs.get, reverse=True)
        top1 += target == guess[0]; top3 += target in guess[:3]
        rows.append([previous, target, guess[0], ", ".join(guess[:3]), float(probs[target])])
    path = ctx.csv("p18_next_token_predictions.csv", ["context_word", "target", "top1", "top3", "target_probability"], rows)
    summary = {"examples": len(examples), "top1_accuracy": top1 / len(examples), "top3_accuracy": top3 / len(examples), "model": "additive-smoothed bigram baseline"}
    ctx.json("p18_accuracy_summary.json", summary)
    print_result(18, "Token prediction accuracy", [path, ctx.path("p18_accuracy_summary.json")])


def p19(ctx: Context) -> None:
    logits = ctx.rng.normal(size=(40, 13))
    targets = ctx.rng.integers(0, logits.shape[1], size=logits.shape[0])
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    manual = -float(log_probs[np.arange(len(targets)), targets].mean())
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1)
    stable = float((logsumexp - logits[np.arange(len(targets)), targets]).mean())
    ctx.json("p19_loss_summary.json", {"manual_nll": manual, "stable_logsumexp_nll": stable, "absolute_difference": abs(manual - stable)})
    print_result(19, "Language-model negative log-likelihood", [ctx.path("p19_loss_summary.json")])


def _ngram_perplexity(train: list[list[str]], test: list[list[str]], order: int, alpha: float = .5) -> float:
    vocab = sorted({token for sentence in train for token in sentence} | {"<unk>"})
    vocab_set = set(vocab)
    counts = defaultdict(Counter)
    totals = Counter()
    for sentence in train:
        padded = ["<bos>"] * (order - 1) + [w if w in vocab_set else "<unk>" for w in sentence] + ["<eos>"]
        for pos in range(order - 1, len(padded)):
            context = tuple(padded[pos - order + 1:pos])
            token = padded[pos]
            counts[context][token] += 1; totals[context] += 1
    losses = []
    for sentence in test:
        padded = ["<bos>"] * (order - 1) + [w if w in vocab_set else "<unk>" for w in sentence] + ["<eos>"]
        for pos in range(order - 1, len(padded)):
            context = tuple(padded[pos - order + 1:pos])
            token = padded[pos]
            p = (counts[context][token] + alpha) / (totals[context] + alpha * len(vocab))
            losses.append(-np.log(p))
    return float(np.exp(np.mean(losses)))


def p20(ctx: Context) -> None:
    sentences = corpus_tokens()
    train, test = sentences[:-5], sentences[-5:]
    rows = [[order, _ngram_perplexity(train, test, order)] for order in (1, 2, 3)]
    path = ctx.csv("p20_perplexity.csv", ["context_order", "perplexity"], rows)
    def draw(fig, ax):
        ax.bar([f"{r[0]}-gram" for r in rows], [r[1] for r in rows], color="#7b6592")
        ax.set_ylabel("Held-out perplexity (toy corpus)")
    ctx.figure("p20_perplexity.png", draw)
    print_result(20, "Perplexity for simple language models", [path, ctx.path("p20_perplexity.png")])


def p21(ctx: Context) -> None:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error
    vocab, vectors, _ = distributional_embeddings()
    x, y = [], []
    for sentence in corpus_tokens():
        for position, token in enumerate(sentence):
            if token in vocab:
                x.append(vectors[vocab.index(token)])
                y.append(position / max(len(sentence) - 1, 1))
    x, y = np.asarray(x), np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=ctx.seed)
    reg = LinearRegression().fit(x_train, y_train)
    cuts = (y_train > .5).astype(int); test_class = (y_test > .5).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(x_train, cuts)
    prediction = reg.predict(x_test)
    summary = {"linear_mae": mean_absolute_error(y_test, prediction), "early_late_accuracy": accuracy_score(test_class, clf.predict(x_test)), "examples": len(y)}
    ctx.json("p21_position_prediction.json", summary)
    def draw(fig, ax):
        ax.scatter(y_test, prediction, alpha=.45, s=15)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--")
        ax.set_xlabel("Normalized true token position"); ax.set_ylabel("Linear-regression prediction")
    ctx.figure("p21_position_regression.png", draw)
    print_result(21, "Predicting token position with regression", [ctx.path("p21_position_prediction.json"), ctx.path("p21_position_regression.png")])


def p22(ctx: Context) -> None:
    train = corpus_tokens()
    counts, _, vocab = _bigram_counts(train)
    cases = [
        ("The cook prepared dinner by", ["chopping vegetables", "painting a wall", "sleeping outdoors"], 0),
        ("The student solved the equation by", ["writing each algebra step", "washing a bicycle", "folding a blanket"], 0),
        ("A runner reached the finish line and", ["raised both arms", "opened a suitcase", "turned on a lamp"], 0),
        ("The mechanic repaired the engine after", ["replacing a worn belt", "reading a poem", "planting flowers"], 0),
        ("A person opened the umbrella because", ["rain began falling", "the room became quiet", "the clock stopped"], 0),
    ]
    rows = []
    correct_count = 0
    for prefix, options, correct_index in cases:
        prior = word_tokens(prefix)
        scores = []
        for option in options:
            words = word_tokens(option)
            seq = prior[-1:] + words
            score = 0.0
            for a, b in zip(seq, seq[1:]):
                score += np.log(_conditional(counts, vocab, a).get(b, 1e-12))
            scores.append(score / max(len(words), 1))
        best = int(np.argmax(scores))
        correct_count += best == correct_index
        for i, (option, score) in enumerate(zip(options, scores)):
            rows.append([prefix, option, float(score), i == best, i == correct_index, i == best == correct_index])
    path = ctx.csv("p22_local_completion_ranking.csv", ["prefix", "continuation", "mean_log_probability", "selected_by_baseline", "gold_continuation", "selected_and_correct"], rows)
    summary = {"examples": len(cases), "accuracy": correct_count / len(cases), "scoring_model": "additive-smoothed local bigram baseline", "dataset": "small authored demonstration; not the HellaSwag benchmark"}
    ctx.json("p22_local_completion_summary.json", summary)
    print_result(22, "Commonsense completion ranking (local HellaSwag-style demo)", [path, ctx.path("p22_local_completion_summary.json")])


def p23(ctx: Context) -> None:
    # A deliberately small, balanced synthetic prompt set for checking measurement logic.
    names = {"she": ["Ava", "Mia", "Noah"], "he": ["Leo", "Omar", "Eli"]}
    roles = ["scientist", "artist", "teacher", "engineer"]
    rows = []
    for pronoun, group in names.items():
        for role in roles:
            for name in group:
                template = f"{name} is a {role}; {pronoun}"
                rows.append([pronoun, role, name, len(word_tokens(template)), len(set(word_tokens(template)) & set(role.split()))])
    by_group = {}
    for pronoun in names:
        by_group[pronoun] = {role: np.mean([r[4] for r in rows if r[0] == pronoun and r[1] == role]) for role in roles}
    gaps = {role: float(by_group["he"][role] - by_group["she"][role]) for role in roles}
    path = ctx.csv("p23_balanced_prompt_audit.csv", ["pronoun_group", "occupation", "name", "token_count", "occupation_token_count"], rows)
    ctx.json("p23_measurement_summary.json", {"mean_occupation_token_count_gap": gaps, "interpretation": "Synthetic prompt audit only; it does not measure social bias in a trained model."})
    print_result(23, "Controlled language-bias measurement demo", [path, ctx.path("p23_measurement_summary.json")])


PROJECTS = {16: p16, 17: p17, 18: p18, 19: p19, 20: p20, 21: p21, 22: p22, 23: p23}
