from __future__ import annotations

import argparse
from pathlib import Path

from .core import Context
from .chapters import attention, embeddings, logits, mlp, tokenization, transformer

PROJECTS = {}
for module in (tokenization, embeddings, logits, transformer, attention, mlp):
    PROJECTS.update(module.PROJECTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one original educational project implementation.")
    parser.add_argument("project", help="project number (1–50) or 'all'")
    parser.add_argument("--output", default="results", help="directory for CSV, JSON, and plot outputs")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible results")
    parser.add_argument("--model", default="toy", help="use toy mode offline or provide a compatible GPT-2 checkpoint, such as distilgpt2")
    args = parser.parse_args()
    ctx = Context(seed=args.seed, output=Path(args.output), model_name=args.model)
    if args.project.lower() == "all":
        selected = range(1, 51)
    else:
        try:
            selected = [int(args.project)]
        except ValueError as exc:
            parser.error(f"invalid project: {args.project!r}")
            raise exc
    for number in selected:
        if number not in PROJECTS:
            parser.error(f"project {number} does not exist; choose a number from 1 to 50")
        PROJECTS[number](ctx)


if __name__ == "__main__":
    main()
