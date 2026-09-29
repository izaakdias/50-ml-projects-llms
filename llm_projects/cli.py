from __future__ import annotations

import argparse
from pathlib import Path

from .core import Context
from .chapters import attention, embeddings, logits, mlp, tokenization, transformer

PROJECTS = {}
for module in (tokenization, embeddings, logits, transformer, attention, mlp):
    PROJECTS.update(module.PROJECTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda uma implementação didática original de um projeto.")
    parser.add_argument("project", help="número do projeto (1–50) ou 'all'")
    parser.add_argument("--output", default="results", help="pasta onde serão salvos CSVs, JSONs e gráficos")
    parser.add_argument("--seed", type=int, default=42, help="semente para resultados reproduzíveis")
    parser.add_argument("--model", default="toy", help="use toy offline ou informe um checkpoint GPT-2 compatível, como distilgpt2")
    args = parser.parse_args()
    ctx = Context(seed=args.seed, output=Path(args.output), model_name=args.model)
    if args.project.lower() == "all":
        selected = range(1, 51)
    else:
        try:
            selected = [int(args.project)]
        except ValueError as exc:
            parser.error(f"projeto inválido: {args.project!r}")
            raise exc
    for number in selected:
        if number not in PROJECTS:
            parser.error(f"projeto {number} não existe; use um número de 1 a 50")
        PROJECTS[number](ctx)


if __name__ == "__main__":
    main()
