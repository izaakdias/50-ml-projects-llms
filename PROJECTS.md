# Project index

All commands run offline by default. For analyses that inspect a pretrained transformer's internals (covered mainly by Projects 24–50), add `--model distilgpt2` after installing `requirements-models.txt`. The default `toy` mode is an educational demonstration with randomly initialized weights, not a pretrained model.

| No. | Topic | Module |
|---:|---|---|
| 01 | Three tokenization schemes | `llm_projects/chapters/tokenization.py` |
| 02 | Text length in characters, words, and tokens | `llm_projects/chapters/tokenization.py` |
| 03 | Token-length frequency tables | `llm_projects/chapters/tokenization.py` |
| 04 | Token length in characters and bytes | `llm_projects/chapters/tokenization.py` |
| 05 | Tokenization as compression | `llm_projects/chapters/tokenization.py` |
| 06 | Tokenization and compression across languages | `llm_projects/chapters/tokenization.py` |
| 07 | Translation and alignment across tokenizers | `llm_projects/chapters/tokenization.py` |
| 08 | Distribution of cosine similarities | `llm_projects/chapters/embeddings.py` |
| 09 | Cosine similarity between sequential tokens | `llm_projects/chapters/embeddings.py` |
| 10 | Cosine similarity of sequential numbers | `llm_projects/chapters/embeddings.py` |
| 11 | Cosine-similarity network | `llm_projects/chapters/embeddings.py` |
| 12 | RSA between two vector representations | `llm_projects/chapters/embeddings.py` |
| 13 | Distance-based and cosine similarity | `llm_projects/chapters/embeddings.py` |
| 14 | Linear semantic axes | `llm_projects/chapters/embeddings.py` |
| 15 | Analogy vectors | `llm_projects/chapters/embeddings.py` |
| 16 | Softmax probability distributions | `llm_projects/chapters/logits.py` |
| 17 | Probabilistic token selection | `llm_projects/chapters/logits.py` |
| 18 | Next-token prediction accuracy | `llm_projects/chapters/logits.py` |
| 19 | Language-model cross-entropy loss | `llm_projects/chapters/logits.py` |
| 20 | N-gram language-model perplexity | `llm_projects/chapters/logits.py` |
| 21 | Regression to predict token position | `llm_projects/chapters/logits.py` |
| 22 | Continuation ranking on a local benchmark | `llm_projects/chapters/logits.py` |
| 23 | Bias audit on controlled prompts | `llm_projects/chapters/logits.py` |
| 24 | Cosine similarities across layers | `llm_projects/chapters/transformer.py` |
| 25 | Category selectivity using cosine similarity | `llm_projects/chapters/transformer.py` |
| 26 | Previous-layer state plus its update | `llm_projects/chapters/transformer.py` |
| 27 | Noise and scale sensitivity across layers | `llm_projects/chapters/transformer.py` |
| 28 | Effective dimensionality of hidden states | `llm_projects/chapters/transformer.py` |
| 29 | Dimensionality reduction of hidden states | `llm_projects/chapters/transformer.py` |
| 30 | Sentiment classification with a decision tree | `llm_projects/chapters/transformer.py` |
| 31 | Logit lens across layers | `llm_projects/chapters/transformer.py` |
| 32 | Hidden-state patching on contrastive IOI prompts | `llm_projects/chapters/transformer.py` |
| 33 | Properties of QKV weights | `llm_projects/chapters/attention.py` |
| 34 | Properties of QKV activations | `llm_projects/chapters/attention.py` |
| 35 | Raw and softmax attention scores | `llm_projects/chapters/attention.py` |
| 36 | Magnitude of attention updates | `llm_projects/chapters/attention.py` |
| 37 | Per-head prediction changes and KL divergence | `llm_projects/chapters/attention.py` |
| 38 | Layer-wise RSA and category selectivity | `llm_projects/chapters/attention.py` |
| 39 | Token frequency, QK scores, and attention updates | `llm_projects/chapters/attention.py` |
| 40 | Effect of ablating attention heads | `llm_projects/chapters/attention.py` |
| 41 | Attention-head patching on contrastive IOI prompts | `llm_projects/chapters/attention.py` |
| 42 | MLP weights and activations | `llm_projects/chapters/mlp.py` |
| 43 | Stages of MLP projections | `llm_projects/chapters/mlp.py` |
| 44 | Probing for nouns and verbs in MLP projections | `llm_projects/chapters/mlp.py` |
| 45 | Minkowski distances and mutual information | `llm_projects/chapters/mlp.py` |
| 46 | Statistics-guided neuron ablation | `llm_projects/chapters/mlp.py` |
| 47 | Supervised probing with boosted trees | `llm_projects/chapters/mlp.py` |
| 48 | Logistic regression to classify “can” and “cannot” | `llm_projects/chapters/mlp.py` |
| 49 | Progressive median replacement in MLP activations | `llm_projects/chapters/mlp.py` |
| 50 | Recommendations using MLP projections | `llm_projects/chapters/mlp.py` |

For example, run Project 35 in offline mode and save its outputs to a dedicated directory:

```bash
python -m llm_projects 35 --output results/project-35
```
