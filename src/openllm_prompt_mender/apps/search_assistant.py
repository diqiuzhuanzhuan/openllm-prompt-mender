# Copyright (c) 2025 Loong Ma
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from typing import Any

import dspy
from dotenv import load_dotenv
from google_cse import GoogleCSE

load_dotenv()


class WebRAGWithCitations(dspy.Signature):
    """
    Answer the question based on the provided web snippets.

    Requirements:
    1. The answer's language and punctuation must strictly match the language used in the user's question.
    2. Cite the source using [[id]] format for every claim, for example [[3]], [[4]].
    3. The id corresponds to the number in the provided context.
    """

    context = dspy.InputField(desc="Numbered web search snippets in the format: 1. [content] \n 2. [content]")
    question = dspy.InputField(desc="The user's query that determines the output language and topic.")
    answer = dspy.OutputField(desc="Summarized response matching the input language with mandatory [[id]] citations.")


class GoogleRAG(dspy.Module):
    """Generate cited answers from already-retrieved web snippets."""

    def __init__(self, callbacks: Any | None = None):
        super().__init__(callbacks)
        self.generate_answer = dspy.ChainOfThought(WebRAGWithCitations)

    def forward(self, question: str, context: str) -> dspy.Prediction:
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# Backward-compatible alias for the old misspelled class name.
GoogleRAg = GoogleRAG


class AssessQuality(dspy.Signature):
    """
    Evaluate generated answer quality.

    Criteria:
    1. Groundedness: the answer must be derived solely from the provided context.
    2. Language consistency: language and punctuation must match the user's question.
    3. Citation accuracy: every claim must include correct [[id]] markers.
    """

    context = dspy.InputField(desc="Retrieved background snippets in numbered format")
    question = dspy.InputField(desc="The original query that defines the required language and topic")
    answer = dspy.InputField(desc="The generated response to be assessed")

    is_grounded = dspy.OutputField(
        desc="Rate the answer on a scale from 0 to 1 according to its degree of alignment with the query",
        type_=float,
    )
    language_match = dspy.OutputField(desc="Boolean indicating if the output language aligns with the question", type_=bool)
    citation_correct = dspy.OutputField(desc="Boolean indicating if [[id]] tags are present and accurate", type_=bool)
    rationale = dspy.OutputField(desc="Textual explanation for the given assessment scores")


def configure_lm(main_model: str = "openai/gpt-4.1-mini") -> dspy.LM:
    """Configure the default DSPy LM for this app and return it."""
    main_lm = dspy.LM(main_model)
    dspy.configure(lm=main_lm)
    return main_lm


def make_judge(judge_model: str = "openai/gpt-4o") -> tuple[dspy.LM, dspy.Module]:
    judge_lm = dspy.LM(judge_model)
    return judge_lm, dspy.ChainOfThought(AssessQuality)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def build_llm_judge_metric(judge_lm: dspy.LM | None = None, judge: dspy.Module | None = None):
    """Create a metric callable suitable for DSPy optimizers."""
    if judge_lm is None or judge is None:
        judge_lm, judge = make_judge()

    def llm_judge_metric(example, pred, trace=None):
        with dspy.context(lm=judge_lm):
            assessment = judge(context=example.context, question=example.question, answer=pred.answer)

        total_score = (
            float(assessment.is_grounded)
            + float(_as_bool(assessment.language_match))
            + float(_as_bool(assessment.citation_correct))
        ) / 3.0

        return dspy.Prediction(score=total_score, feedback=assessment.rationale)

    return llm_judge_metric


def build_trainset(queries: Iterable[str], min_results: int = 1, max_results: int = 10) -> list[dspy.Example]:
    """Build a DSPy trainset by retrieving snippets from Google CSE."""
    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    search_engine_id = os.environ.get("GOOGLE_CSE_CX")
    if not api_key or not search_engine_id:
        msg = "GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX must be set to build a search trainset."
        raise RuntimeError(msg)

    google_client = GoogleCSE(api_key=api_key, search_engine_id=search_engine_id)
    trainset = []

    for question in queries:
        num_results = random.randint(min_results, max_results)
        search_results = google_client.web_search(question, num_results=num_results)
        context = "\n".join(f"{idx + 1}. {result.snippet}" for idx, result in enumerate(search_results))
        trainset.append(dspy.Example(question=question, context=context).with_inputs("question", "context"))

    return trainset


def answer(question: str, context: str, model: str = "openai/gpt-4.1-mini") -> dspy.Prediction:
    """Convenience entry point for direct, unoptimized inference."""
    configure_lm(model)
    return GoogleRAG()(question=question, context=context)


if __name__ == "__main__":
    sample_question = os.environ.get("PROMPT_MENDER_SAMPLE_QUESTION", "What is DSPy?")
    sample_context = os.environ.get("PROMPT_MENDER_SAMPLE_CONTEXT", "1. DSPy is a framework for programming LM pipelines.")
    print(answer(sample_question, sample_context).answer)
