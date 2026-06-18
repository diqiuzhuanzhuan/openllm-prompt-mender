from __future__ import annotations

import json
import os
from pathlib import Path

import chainlit as cl
from ollama import Client

from openllm_prompt_mender.apps.audio_assistant import generate_template

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "Qwen3-4B-2507-RL-global-285-0123-fp16"
DEFAULT_COMPILED_PROGRAM = "audio_assistant.json"
DEFAULT_PROMPT_PATH = "20260123.prompt.json"
START_MARKER = "[[ ## template ## ]]"
END_MARKER = "[[ ## completed ## ]]"


class TokenStreamExtractor:
    """Yield content between DSPy template markers without leaking marker fragments."""

    def __init__(self, start_marker: str = START_MARKER, end_marker: str = END_MARKER):
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.state = "WAIT_START"
        self.partial = ""

    def feed(self, chunk: str) -> list[str]:
        if not chunk:
            return []

        outputs = []
        self.partial += chunk

        while self.partial:
            if self.state == "WAIT_START":
                start_index = self.partial.find(self.start_marker)
                if start_index == -1:
                    _, self.partial = self._split_safe_suffix(self.partial, len(self.start_marker))
                    break
                self.partial = self.partial[start_index + len(self.start_marker) :]
                self.state = "COLLECTING"
                continue

            end_index = self.partial.find(self.end_marker)
            if end_index == -1:
                safe, self.partial = self._split_safe_suffix(self.partial, len(self.end_marker))
                if safe:
                    outputs.append(safe)
                break

            if end_index > 0:
                outputs.append(self.partial[:end_index])
            self.partial = self.partial[end_index + len(self.end_marker) :]
            self.state = "WAIT_START"

        return outputs

    @staticmethod
    def _split_safe_suffix(pending: str, marker_len: int) -> tuple[str, str]:
        keep = max(marker_len - 1, 0)
        if len(pending) <= keep:
            return "", pending
        split_index = len(pending) - keep
        return pending[:split_index], pending[split_index:]


def load_prompt_messages(prompt_path: str | Path) -> list[dict[str, str]] | None:
    path = Path(prompt_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_prompt_messages(requirements: str, prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> list[dict[str, str]] | None:
    messages = load_prompt_messages(prompt_path)
    if not messages:
        return None

    messages = [dict(message) for message in messages]
    messages[-1]["content"] = (
        "[[ ## requirements ## ]]\n"
        f"{requirements}\n\n"
        "Respond with the corresponding output field"
    )
    return messages


async def stream_ollama_template(requirements: str, stream_msg: cl.Message) -> bool:
    messages = build_prompt_messages(requirements, os.environ.get("AUDIO_ASSISTANT_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    if not messages:
        return False

    client = Client(host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    model_name = os.environ.get("AUDIO_ASSISTANT_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    extractor = TokenStreamExtractor()

    response = client.chat(model=model_name, messages=messages, stream=True)
    for chunk in response:
        chunk_text = chunk.message.get("content", "") if chunk.message else ""
        for segment in extractor.feed(chunk_text):
            if segment:
                await stream_msg.stream_token(segment)

    return True


@cl.on_message
async def render_ui(message: cl.Message):
    requirements = message.content.strip()
    if not requirements:
        return

    stream_msg = cl.Message(content="")
    await stream_msg.send()

    streamed = await stream_ollama_template(requirements, stream_msg)
    if streamed:
        await stream_msg.update()
        return

    compiled_path = os.environ.get("AUDIO_ASSISTANT_COMPILED_PATH", DEFAULT_COMPILED_PROGRAM)
    template = generate_template(requirements, compiled_path=compiled_path if Path(compiled_path).exists() else None)
    stream_msg.content = template
    await stream_msg.update()
