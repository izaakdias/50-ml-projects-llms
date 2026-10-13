# 50 ML Projects to Understand LLMs — Independent Python Implementations

A collection of 50 runnable Python experiments inspired by the topics in Mike X. Cohen's book [*50 ML Projects to Understand LLMs*](https://www.packtpub.com/en-ca/product/50-ml-projects-to-understand-llms-9781808082559).

This is an independent study companion, not an official solution set. The code, examples, small corpus, and visualizations in this repository are original implementations; they are not copied from the book. See the full [project index](PROJECTS.md).

## What you'll explore

The experiments cover tokenization, embeddings, output logits, transformer representations, attention, and MLPs. Each project can be run on its own and writes inspectable outputs such as CSV files, JSON files, and plots.

## Quick start

Python 3.12 is the version verified for this repository. The offline mode supports Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m llm_projects 1
```

Run one project, set a reproducible seed, or run all 50 in the default offline mode:

```bash
python -m llm_projects 14 --output results/project-14 --seed 7
python -m llm_projects all --output results/all
```

Results are written to `results/` by default. Use `--output` to choose another directory. Project implementations are organized by topic in `llm_projects/chapters/`.

## Model modes

- **Offline toy mode (`--model toy`)** is the default. It uses local example data, classical ML methods, and a small causal decoder initialized with random weights. It is designed to demonstrate calculations and visualizations; it does **not** report empirical results from a pretrained GPT-2, BERT, or other trained language model.
- **Pretrained checkpoint mode (`--model distilgpt2`)** uses public DistilGPT-2 weights for analyses that inspect transformer internals, such as hidden states, attention, and MLP activations. The first run downloads the checkpoint and may be slow on CPU.

Install the optional model dependencies before using the checkpoint mode:

```bash
python -m pip install -r requirements-models.txt
python -m llm_projects 35 --model distilgpt2
```

Tokenization, simple embedding, and regression examples continue to use local data. The generated report or plot should identify which mode produced it. Project 22 uses a local continuation-ranking example and does not download HellaSwag; Project 23 audits controlled local prompts and does not measure bias in a trained model. Attention-head and hidden-state interventions use approximate patching or ablation measures, with details recorded in the generated outputs.

## Reproducibility and data

Use `--seed` to control the randomized parts of the educational examples. The scripts do not overwrite source files or upload data. Generated results are written to the selected output directory and ignored by Git by default.

## Book attribution and scope

The project's themes and sequence are based on topics from Mike X. Cohen's [book](https://www.packtpub.com/en-ca/product/50-ml-projects-to-understand-llms-9781808082559). The code and examples here are independently written, and some experiments use smaller educational demonstrations instead of the book's datasets or figures. Consult the book for its full explanations; this repository does not replace or reproduce the author's material.
