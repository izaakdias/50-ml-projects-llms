# 50 projetos originais para entender LLMs

Implementações novas em Python, feitas do zero a partir dos temas e objetivos do livro *50 ML Projects to Understand LLMs*, de Mike X Cohen. Este repositório **não contém notebooks nem soluções copiadas do autor**. O código, os exemplos e o pequeno corpus foram escritos para este projeto.

O objetivo é transformar cada tema em um experimento que rode, gere dados e deixe um resultado examinável. O índice completo está em [PROJECTS.md](PROJECTS.md).

## Começar

Python 3.12 é a versão verificada neste repositório. O modo offline usa Python 3.10 ou superior; o modo de checkpoint também depende da versão de PyTorch compatível com seu Python.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m llm_projects 1
```

O comando salva CSVs, JSONs e gráficos na pasta `results/`. Escolha outra pasta com `--output` e altere a semente com `--seed`:

```bash
python -m llm_projects 14 --output resultados/projeto-14 --seed 7
python -m llm_projects all --output resultados/todos
```

Cada projeto também tem um número e uma função próprios em `llm_projects/chapters/`. É possível rodar um só sem executar os demais.

## Modos de execução

- **Padrão (`--model toy`)**: roda offline, com corpus curto criado para este repositório, algoritmos clássicos e um decoder causal pequeno, implementado neste projeto e inicializado aleatoriamente. Esse modo demonstra como calcular e visualizar as medidas; ele **não representa resultados empíricos de GPT-2, BERT ou outro modelo treinado**.
- **Checkpoint real (`--model distilgpt2`)**: usa pesos públicos de um modelo GPT-2 compatível para os projetos que analisam estados ocultos, atenção e MLP. Instale primeiro as dependências opcionais; a primeira execução baixa os pesos e pode ser mais lenta em CPU.

```bash
python -m pip install -r requirements-models.txt
python -m llm_projects 35 --model distilgpt2
```

Os projetos de tokenização, embeddings simples e regressão continuam usando os dados locais. O modo de checkpoint se aplica aos projetos que leem os estados internos do decoder. O notebook/relatório deve indicar qual modo gerou cada figura. Os projetos 22 e 23 continuam sendo exemplos locais: o 22 não baixa o conjunto HellaSwag e o 23 não mede viés de um modelo treinado. Intervenções em cabeças e estados usam medidas de patch/ablação aproximadas, descritas nos próprios CSVs e gráficos.

## Resultados e reprodutibilidade

Os scripts não sobrescrevem o código-fonte e não fazem upload de dados. `--seed` controla as partes aleatórias do modo didático. Os arquivos produzidos ficam no diretório de saída e são ignorados pelo Git por padrão.

## Créditos e escopo

Os temas e a sequência dos 50 projetos foram inspirados pelo livro de Mike X Cohen. As implementações, os exemplos de texto, o corpus e as visualizações deste repositório são independentes; alguns experimentos usam demonstrações didáticas menores em vez dos dados e das figuras do livro. Consulte o livro para o contexto conceitual completo; este código não substitui as explicações do autor.
