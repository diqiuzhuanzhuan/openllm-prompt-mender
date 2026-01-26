# MIT License
#
# Copyright (c) 2025 LoongMa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import dspy
import os
from openllm_prompt_mender.utils.data_utils import load_trainset
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

class AnalyzeRequirement(dspy.Signature):
    """Extract structured constraints from a free-form requirement."""

    requirement = dspy.InputField(desc="User requirement text.")
    language = dspy.OutputField(desc="Target language, e.g. zh, en, ja, de.")
    style = dspy.OutputField(desc="Speaking style, e.g. academic, business, popular science.")
    tone = dspy.OutputField(desc="Tone or attitude, e.g. formal, friendly, neutral.")
    audience = dspy.OutputField(desc="Target audience, e.g. executives, engineers, beginners.")
    verbosity = dspy.OutputField(desc="Expected level of detail: short / medium / detailed.")


class TemplateGenerator(dspy.Signature):
    """
    Generate a summary template based on user requirements.

    If the user does not specify formatting requirements,
    generate a structured outline using Markdown headers (#, ##)
    and bullet points.
    """
    requirements = dspy.InputField(desc="User input containing either keywords or a descriptive paragraph")
    template = dspy.OutputField(desc="By default, generate a structured outline using Markdown headers (#, ##) and bullet points unless the user explicitly specifies alternative formatting requirements.")



class VoiceMemoApp(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_template = dspy.Predict(TemplateGenerator)

    def forward(self, requirements):
        return self.generate_template(requirements=requirements)


class AssessTemplateQuality(dspy.Signature):
    """
    Evaluate the quality of a generated template based on user requirements.

    Assessment criteria:
    1. Clarity and conciseness of the template.
    2. Ease of understanding and usability.
    3. Flexibility for modification and extension.
    4. Alignment with the user's requirements, intended scenario, audience, and style.
    5. Match the output language to the requirements unless a specific language is requested.
    """

    # -------------------- Input Fields --------------------
    requirements: str = dspy.InputField(
        desc="The user's requirements or instructions for the template."
    )
    template: str = dspy.InputField(
        desc="The generated template that needs to be evaluated."
    )

    # -------------------- Output Fields --------------------
    general_score: float = dspy.OutputField(
        desc="Overall quality score of the template, from 0.0 to 1.0.",
        type_=float
    )

    tone_score: float = dspy.OutputField(
        desc="Score (0.0–1.0) evaluating how well the template's tone matches the intended style or voice.",
        type_=float
    )

    scenario_alignment_score: float = dspy.OutputField(
        desc="Score (0.0–1.0) measuring how well the template fits the specified scenario.",
        type_=float
    )

    audience_match_score: float = dspy.OutputField(
        desc="Score (0.0–1.0) assessing whether the template is appropriate for the target audience.",
        type_=float
    )

    language_consistency_score: float = dspy.OutputField(desc=(
        "A score between 0.0 and 1.0 indicating how consistent the language of the template is "
        "with the language used in the requirements. "
        "Score 1.0 means fully consistent, and 0.0 means completely inconsistent. "
        "If the requirements explicitly request a specific language, evaluate consistency "
        "against that requested language instead."),
        type_=float
    )
    language_appropriateness_score: float = dspy.OutputField(desc=(
        "Score (0.0–1.0) evaluating whether the use of multiple languages in the template "
        "is appropriate and contextually justified. "
        "A high score means that any foreign-language terms (e.g., English in a Chinese template) "
        "are limited to widely accepted abbreviations or technical terms (e.g., ROI, KPI), "
        "while structural elements, placeholders, and labels are properly localized. "
        "A low score indicates unnecessary or inappropriate mixing, such as untranslated "
        "placeholders like '[Action]' or '[Date]' in a non-English template."
    ),
    type_=float
    )

    rationale: str = dspy.OutputField(
        desc="A concise explanation justifying the scores assigned for each evaluation dimension.",
        type_=str
    )

main_lm = dspy.LM(model="ollama/Qwen3-4B-2507-RL-global-20-0114-fp16", max_tokens=10240)
dspy.configure(lm=main_lm)
judge_lm = dspy.LM(model="openai/gpt-5-mini")
judge = dspy.ChainOfThought(AssessTemplateQuality)


def llm_judge_metric(example, pred, trace=None):
    with dspy.context(lm=judge_lm):
        # Execute the judge module with the conrequirements, question, and predicted answer
        assessment = judge(
            requirements=example.requirements,
            template=pred.template
        )
        if assessment.language_consistency_score < 0.5:
            print("language_consistency_score: ", assessment.language_consistency_score)
            print("requirements: ", example.requirements)
            print("template: ", pred.template)
        
    # Calculate the average score from the float criterias
    # Each True counts as 1.0, False counts as 0.0
    total_score = (
        assessment.general_score +
        assessment.tone_score +
        assessment.scenario_alignment_score +
        assessment.audience_match_score +
        assessment.language_consistency_score + 
        assessment.language_appropriateness_score
    ) / 6.0
    
    # Return a Prediction object which can include both the numeric score and feedback [4]
    # Feedback is crucial for optimizers like GEPA to understand how to improve the program [6, 7]
    return dspy.Prediction(
        score=total_score, 
        feedback=assessment.rationale
    )


def main():
    from dspy.teleprompt import MIPROv2

    tp = MIPROv2(
        metric=llm_judge_metric, 
        auto="light",  # auto can be set as light, medium, heavy
        prompt_model=judge_lm, 
        teacher_settings=dict(lm=judge_lm)
    )

    if os.path.exists("data/requirements.jsonl"):
        trainset = load_trainset("data/requirements.jsonl", input_keys=tuple(['requirements', ]))

    compiled_program = tp.compile(VoiceMemoApp(), trainset=trainset)

    import pprint
    pprint.pprint("prompt: " + "#"*300)
    for message in dspy.clients.base_lm.GLOBAL_HISTORY[-1]["messages"]:
        pprint.pprint(message)
    pprint.pprint("#"*300)
    compiled_program.save("audio_assistant.json")

    while True:
        requirements = input("Enter your requirements: ")
        if requirements == "exit":
            break

        pred = compiled_program(requirements=requirements)

        import pprint
        pprint.pprint(pred.template)

if __name__ == "__main__":
    main()

import chainlit as cl

def dump_prompt():
    import pprint
    pprint.pprint("prompt: " + "#"*300)
    messages = dspy.clients.base_lm.GLOBAL_HISTORY[-1]["messages"]
    pprint(messages)
    pprint.pprint("#"*300)
    return messages

@cl.on_message
async def render_ui(message: cl.Message):
    if message.content == "<dump_prompt>":
        await cl.Message(
            content=dump_prompt()).send
        return
    voice_memo_app = VoiceMemoApp()
    voice_memo_app.load("audio_assistant.json")
    pred = voice_memo_app(requirements=message.content)
    await cl.Message(
        content=pred.template,
        actions=[
            cl.Action(name="copy", label="copy", payload={"text": "pred.template"}, type="copy")
        ]
        ).send()

#chainlit run src/openllm_prompt_mender/apps/audio_assistant.py -w --host 0.0.0.0 --port 8538 
"""
import gradio as gr

def staged_template(requirements):
    yield "### Understanding requirements...\n"
    yield f"{requirements}\n\n"

    # 模拟计算 / 调用 DSPy
    pred = compiled_program(requirements=requirements)

    yield "### Generated Template\n"
    for para in pred.template.split("\n\n"):
        yield para + "\n\n"

gr.Interface(
    fn=staged_template,
    inputs=gr.Textbox(placeholder="请输入模板要求"),
    outputs=gr.Markdown()
).launch()
"""