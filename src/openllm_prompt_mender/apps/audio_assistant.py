# Copyright (c) 2025 Loong Ma
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAIN_MODEL = "ollama/Qwen3-4B-2507-RL-global-285-0123-fp16"
DEFAULT_JUDGE_MODEL = "openai/gpt-5-mini"
DEFAULT_MAX_TOKENS = 10240


class AnalyzeRequirement(dspy.Signature):
    """Extract structured constraints from a free-form requirement."""

    requirement = dspy.InputField(desc="User requirement text.")
    language = dspy.OutputField(desc="Target language, for example zh, en, ja, or de.")
    style = dspy.OutputField(desc="Speaking style, for example academic, business, or popular science.")
    tone = dspy.OutputField(desc="Tone or attitude, for example formal, friendly, or neutral.")
    audience = dspy.OutputField(desc="Target audience, for example executives, engineers, or beginners.")
    verbosity = dspy.OutputField(desc="Expected level of detail: short, medium, or detailed.")


class TemplateGenerator(dspy.Signature):
    """Generate a structured summary template from user requirements."""

    requirements = dspy.InputField(desc="User input containing either keywords or a descriptive paragraph.")
    template = dspy.OutputField(
        desc=(
            "Generate a practical Markdown template. Use Markdown headers and bullet points by default unless "
            "the user explicitly asks for another format."
        )
    )


class VoiceMemoApp(dspy.Module):
    """DSPy module for generating voice memo or meeting-summary templates."""

    def __init__(self):
        super().__init__()
        self.generate_template = dspy.Predict(TemplateGenerator)

    def forward(self, requirements: str) -> dspy.Prediction:
        return self.generate_template(requirements=requirements)


class AssessTemplateQuality(dspy.Signature):
    """Evaluate generated template quality against user requirements."""

    requirements: str = dspy.InputField(desc="The user's requirements or instructions for the template.")
    template: str = dspy.InputField(desc="The generated template that needs to be evaluated.")

    general_score: float = dspy.OutputField(
        desc="Overall quality score of the template, from 0.0 to 1.0.",
        type_=float,
    )
    tone_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 for how well the template tone matches the intended style or voice.",
        type_=float,
    )
    hierarchy_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 for logical structure and hierarchy. The template needs at least two levels.",
        type_=float,
    )
    scenario_alignment_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 for how well the template fits the specified scenario.",
        type_=float,
    )
    audience_match_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 for whether the template is appropriate for the target audience.",
        type_=float,
    )
    language_consistency_score: float = dspy.OutputField(
        desc=(
            "Score from 0.0 to 1.0 for whether the template language matches the requirements language. "
            "If a specific output language is requested, evaluate against that language instead."
        ),
        type_=float,
    )
    language_appropriateness_score: float = dspy.OutputField(
        desc=(
            "Score from 0.0 to 1.0 for whether any language mixing is appropriate. Common technical terms "
            "such as ROI or KPI are acceptable, but structural placeholders should be localized."
        ),
        type_=float,
    )
    rationale: str = dspy.OutputField(
        desc="Concise explanation justifying the assigned scores and naming the main improvement opportunity.",
        type_=str,
    )


def configure_lm(main_model: str = DEFAULT_MAIN_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> dspy.LM:
    """Configure the default DSPy LM for template generation."""
    main_lm = dspy.LM(model=main_model, max_tokens=max_tokens)
    dspy.configure(lm=main_lm)
    return main_lm


def make_judge(judge_model: str = DEFAULT_JUDGE_MODEL) -> tuple[dspy.LM, dspy.Module]:
    """Create the judge LM and module used by optimizers."""
    judge_lm = dspy.LM(model=judge_model)
    return judge_lm, dspy.ChainOfThought(AssessTemplateQuality)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0)


def build_llm_judge_metric(judge_lm: dspy.LM | None = None, judge: dspy.Module | None = None):
    """Create a DSPy metric that returns both score and textual feedback."""
    if judge_lm is None or judge is None:
        judge_lm, judge = make_judge()

    def llm_judge_metric(example, pred, trace=None):
        with dspy.context(lm=judge_lm):
            assessment = judge(requirements=example.requirements, template=pred.template)

        scores = [
            assessment.general_score,
            assessment.tone_score,
            assessment.hierarchy_score,
            assessment.scenario_alignment_score,
            assessment.audience_match_score,
            assessment.language_consistency_score,
            assessment.language_appropriateness_score,
        ]
        total_score = sum(_coerce_score(score) for score in scores) / len(scores)
        return dspy.Prediction(score=total_score, feedback=assessment.rationale)

    return llm_judge_metric


def load_program(path: str | Path) -> VoiceMemoApp:
    """Load a compiled VoiceMemoApp from disk."""
    program = VoiceMemoApp()
    program.load(str(path))
    return program


def generate_template(
    requirements: str,
    *,
    compiled_path: str | Path | None = None,
    main_model: str = DEFAULT_MAIN_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Generate a template with either a compiled program or the base module."""
    configure_lm(main_model=main_model, max_tokens=max_tokens)
    program = load_program(compiled_path) if compiled_path else VoiceMemoApp()
    return program(requirements=requirements).template


__all__ = [
    "AnalyzeRequirement",
    "AssessTemplateQuality",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAIN_MODEL",
    "DEFAULT_MAX_TOKENS",
    "TemplateGenerator",
    "VoiceMemoApp",
    "build_llm_judge_metric",
    "configure_lm",
    "generate_template",
    "load_program",
    "make_judge",
]


if __name__ == "__main__":
    requirement_text = input("Enter your requirements: ")
    print(generate_template(requirement_text))
