# Índice dos projetos

Todos os comandos abaixo rodam em modo offline. Para os projetos 24–50, acrescente `--model distilgpt2` para analisar um checkpoint real (após instalar `requirements-models.txt`). O modo `toy` é uma demonstração didática com pesos aleatórios.

| Nº | Tema | Módulo |
|---:|---|---|
| 01 | Três esquemas de tokenização | `llm_projects/chapters/tokenization.py` |
| 02 | Comprimento de textos em caracteres, palavras e tokens | `llm_projects/chapters/tokenization.py` |
| 03 | Tabelas de frequência do comprimento de tokens | `llm_projects/chapters/tokenization.py` |
| 04 | Comprimento de tokens em caracteres e bytes | `llm_projects/chapters/tokenization.py` |
| 05 | Tokenização como compressão | `llm_projects/chapters/tokenization.py` |
| 06 | Tokenização e compressão entre idiomas | `llm_projects/chapters/tokenization.py` |
| 07 | Tradução e alinhamento entre tokenizadores | `llm_projects/chapters/tokenization.py` |
| 08 | Distribuição de similaridades cosseno | `llm_projects/chapters/embeddings.py` |
| 09 | Similaridade cosseno entre tokens sequenciais | `llm_projects/chapters/embeddings.py` |
| 10 | Similaridade cosseno de números sequenciais | `llm_projects/chapters/embeddings.py` |
| 11 | Rede de similaridades cosseno | `llm_projects/chapters/embeddings.py` |
| 12 | RSA entre duas representações vetoriais | `llm_projects/chapters/embeddings.py` |
| 13 | Similaridade por distância e cosseno | `llm_projects/chapters/embeddings.py` |
| 14 | Eixos semânticos lineares | `llm_projects/chapters/embeddings.py` |
| 15 | Vetores de analogia | `llm_projects/chapters/embeddings.py` |
| 16 | Distribuições de probabilidade softmax | `llm_projects/chapters/logits.py` |
| 17 | Seleção probabilística de tokens | `llm_projects/chapters/logits.py` |
| 18 | Acurácia de previsão do próximo token | `llm_projects/chapters/logits.py` |
| 19 | Perda de linguagem por entropia cruzada | `llm_projects/chapters/logits.py` |
| 20 | Perplexidade de modelos n-gram | `llm_projects/chapters/logits.py` |
| 21 | Regressão para prever posição de token | `llm_projects/chapters/logits.py` |
| 22 | Ranking de continuações em um benchmark local | `llm_projects/chapters/logits.py` |
| 23 | Auditoria de viés em prompts controlados | `llm_projects/chapters/logits.py` |
| 24 | Similaridades cosseno entre camadas | `llm_projects/chapters/transformer.py` |
| 25 | Seletividade de categorias por cosseno | `llm_projects/chapters/transformer.py` |
| 26 | Estado da camada anterior mais seu ajuste | `llm_projects/chapters/transformer.py` |
| 27 | Sensibilidade a ruído e escala por camada | `llm_projects/chapters/transformer.py` |
| 28 | Dimensionalidade efetiva dos estados ocultos | `llm_projects/chapters/transformer.py` |
| 29 | Redução dimensional de estados ocultos | `llm_projects/chapters/transformer.py` |
| 30 | Classificação de sentimento com árvore de decisão | `llm_projects/chapters/transformer.py` |
| 31 | Logit lens entre camadas | `llm_projects/chapters/transformer.py` |
| 32 | Patching de estado oculto em prompts contrastivos de IOI | `llm_projects/chapters/transformer.py` |
| 33 | Características dos pesos QKV | `llm_projects/chapters/attention.py` |
| 34 | Características das ativações QKV | `llm_projects/chapters/attention.py` |
| 35 | Scores de atenção brutos e após softmax | `llm_projects/chapters/attention.py` |
| 36 | Magnitude dos ajustes de atenção | `llm_projects/chapters/attention.py` |
| 37 | Mudança de previsão e divergência KL por cabeça | `llm_projects/chapters/attention.py` |
| 38 | RSA por camada e seletividade de categoria | `llm_projects/chapters/attention.py` |
| 39 | Frequência de token, QK e ajustes de atenção | `llm_projects/chapters/attention.py` |
| 40 | Efeito de silenciar cabeças de atenção | `llm_projects/chapters/attention.py` |
| 41 | Patching de cabeças de atenção em prompts contrastivos de IOI | `llm_projects/chapters/attention.py` |
| 42 | Características de pesos e ativações MLP | `llm_projects/chapters/mlp.py` |
| 43 | Etapas das projeções MLP | `llm_projects/chapters/mlp.py` |
| 44 | Sondagem de substantivos e verbos nas projeções MLP | `llm_projects/chapters/mlp.py` |
| 45 | Distâncias de Minkowski e informação mútua | `llm_projects/chapters/mlp.py` |
| 46 | Lesão de neurônios guiada por estatística | `llm_projects/chapters/mlp.py` |
| 47 | Sondagem supervisionada com árvores impulsionadas | `llm_projects/chapters/mlp.py` |
| 48 | Classificação de “can” e “cannot” com regressão logística | `llm_projects/chapters/mlp.py` |
| 49 | Substituição progressiva por medianas nas ativações MLP | `llm_projects/chapters/mlp.py` |
| 50 | Recomendação usando projeções MLP | `llm_projects/chapters/mlp.py` |

Para executar, por exemplo:

```bash
python -m llm_projects 35 --output results/projeto-35
```
