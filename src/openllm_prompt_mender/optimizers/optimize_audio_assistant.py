# Copyright (c) 2025 Loong Ma
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from openllm_prompt_mender.apps.audio_assistant import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAIN_MODEL,
    DEFAULT_MAX_TOKENS,
    VoiceMemoApp,
    build_llm_judge_metric,
    configure_lm,
    make_judge,
)
from openllm_prompt_mender.utils.data_utils import load_trainset

app = typer.Typer(help="Optimize the DSPy audio assistant.")


@app.command()
def main(
    trainset_path: Path = typer.Option(Path("data/requirements.jsonl"), help="JSONL DSPy trainset path."),
    output_path: Path = typer.Option(Path("audio_assistant.json"), help="Compiled DSPy program output path."),
    main_model: str = typer.Option(DEFAULT_MAIN_MODEL, help="Model used by the optimized program."),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, help="Model used for judging and prompt optimization."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, help="Max tokens for the main model."),
    auto: str = typer.Option("light", help="MIPROv2 search budget: light, medium, or heavy."),
):
    load_dotenv()
    if not trainset_path.exists():
        raise typer.BadParameter(f"Trainset not found: {trainset_path}")

    configure_lm(main_model=main_model, max_tokens=max_tokens)
    judge_lm, judge = make_judge(judge_model)
    metric = build_llm_judge_metric(judge_lm=judge_lm, judge=judge)
    trainset = load_trainset(str(trainset_path), input_keys=("requirements",))

    from dspy.teleprompt import MIPROv2

    teleprompter = MIPROv2(
        metric=metric,
        auto=auto,
        prompt_model=judge_lm,
        teacher_settings={"lm": judge_lm},
    )
    compiled_program = teleprompter.compile(VoiceMemoApp(), trainset=trainset)
    compiled_program.save(str(output_path))
    typer.echo(f"Saved compiled audio assistant to {output_path}")


if __name__ == "__main__":
    app()
