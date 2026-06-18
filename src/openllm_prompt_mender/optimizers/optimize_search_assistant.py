from __future__ import annotations

from pathlib import Path

import dspy
import typer
from datasets import load_dataset
from dotenv import load_dotenv

from openllm_prompt_mender.apps.search_assistant import (
    GoogleRAG,
    build_llm_judge_metric,
    build_trainset,
    configure_lm,
    make_judge,
)
from openllm_prompt_mender.utils.data_utils import load_trainset, save_trainset

app = typer.Typer(help="Optimize the DSPy search assistant.")


def load_queries(file_path: Path) -> list[str]:
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    return list(dataset["query"])


@app.command()
def main(
    queries_path: Path = typer.Option(Path("data/queries.jsonl"), help="JSONL file with a query field."),
    trainset_path: Path = typer.Option(Path("data/trainset.jsonl"), help="Cached DSPy trainset path."),
    output_path: Path = typer.Option(Path("search_assistant.json"), help="Compiled DSPy program output path."),
    main_model: str = typer.Option("openai/gpt-4.1-mini", help="Model used by the optimized program."),
    judge_model: str = typer.Option("openai/gpt-4o", help="Model used for judging and prompt optimization."),
    auto: str = typer.Option("light", help="MIPROv2 search budget: light, medium, or heavy."),
):
    load_dotenv()
    configure_lm(main_model)
    judge_lm, judge = make_judge(judge_model)
    metric = build_llm_judge_metric(judge_lm=judge_lm, judge=judge)

    if trainset_path.exists():
        trainset = load_trainset(str(trainset_path))
    else:
        trainset = build_trainset(load_queries(queries_path))
        trainset_path.parent.mkdir(parents=True, exist_ok=True)
        save_trainset(trainset, str(trainset_path))

    from dspy.teleprompt import MIPROv2

    teleprompter = MIPROv2(
        metric=metric,
        auto=auto,
        prompt_model=judge_lm,
        teacher_settings={"lm": judge_lm},
    )
    compiled_program = teleprompter.compile(GoogleRAG(), trainset=trainset)
    compiled_program.save(str(output_path))
    typer.echo(f"Saved compiled search assistant to {output_path}")


if __name__ == "__main__":
    app()
