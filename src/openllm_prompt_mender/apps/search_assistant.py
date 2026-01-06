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
from typing import List
from dotenv import load_dotenv
import os
from datasets import load_dataset
from openllm_prompt_mender.utils.data_utils import load_trainset, save_trainset


load_dotenv()

class WebRAGWithCitations(dspy.Signature):
    """
    Answer the question based on the provided web snippets. 
    Requirements:
    1. The answer's language and punctuation must strictly match the language used in the user's question.
    2. Cite the source using [[id]] format for every claim (e.g., [[3]], [[4]]). 
    3. The id corresponds to the number in the provided context.
    """

    context = dspy.InputField(desc="Numbered web search snippets in the format: 1. [content] \n 2. [content]")
    question = dspy.InputField(desc="The user's query that determines the output language and topic.")
    answer = dspy.OutputField(desc="Summarized response matching the input language with mandatory [[id]] citations.")



class GoogleRAg(dspy.Module):
    def __init__(self, callbacks=None):
        """Initialize the SearchAssistant class.

        Args:
            callbacks (optional): Callback functions to be used during the execution.
                Defaults to None.

        Sets up:
            retrieve: A dspy.Retrieve component configured to retrieve 10 items.
            generate_answer: A dspy.ChainOfThought component using WebRAGWithCitations
                for generating answers with citations.
        """
        super().__init__(callbacks)
        self.retrieve = dspy.Retrieve(k=10)
        self.generate_answer = dspy.ChainOfThought(WebRAGWithCitations)


    def forward(self, question):
        """Process a question by retrieving relevant context and generating an answer.

        Args:
            question (str): The input question to be processed.

        Returns:
            dspy.Prediction: A prediction object containing:
                - context (list): The retrieved context passages relevant to the question.
                - answer (str): The generated answer to the question based on the context.
        """
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
   
  

class AssessQuality(dspy.Signature):
    """
    Evaluate the quality of a generated answer based on three key requirements:
    1. Groundedness: The answer must be derived solely from the provided context.
    2. Language Consistency: The language and punctuation must strictly match the user's question.
    3. Citation Accuracy: Every claim must include correct [[id]] markers corresponding to the context.
    """
    context = dspy.InputField(desc="Retrieved background snippets in numbered format")
    question = dspy.InputField(desc="The original query that defines the required language and topic")
    answer = dspy.InputField(desc="The generated response to be assessed")
    
    # Evaluation outcomes
    is_grounded = dspy.OutputField(desc="Boolean indicating if the answer is supported by the context", type_=bool)
    language_match = dspy.OutputField(desc="Boolean indicating if the output language aligns with the question", type_=bool)
    citation_correct = dspy.OutputField(desc="Boolean indicating if [[id]] tags are present and accurate", type_=bool)
    rationale = dspy.OutputField(desc="Textual explanation for the given assessment scores") 

judge = dspy.ChainOfThought(AssessQuality)

main_lm = dspy.LM('openai/gpt-4.1-mini')
judge_lm = dspy.LM('openai/gpt-4o')
dspy.configure(lm=main_lm)


def llm_judge_metric(example, pred, trace=None):
    with dspy.context(lm=judge_lm):
        # Execute the judge module with the context, question, and predicted answer
        assessment = judge(
            context=example.context, 
            question=example.question, 
            answer=pred.answer
        )
        
    # Calculate the average score from the boolean criteria [4, 5]
    # Each True counts as 1.0, False counts as 0.0
    total_score = (
        float(assessment.is_grounded) + 
        float(assessment.language_match) + 
        float(assessment.citation_correct)
    ) / 3.0
    
    # Return a Prediction object which can include both the numeric score and feedback [4]
    # Feedback is crucial for optimizers like GEPA to understand how to improve the program [6, 7]
    return dspy.Prediction(
        score=total_score, 
        feedback=assessment.rationale
    )

from dspy.teleprompt import MIPROv2

tp = MIPROv2(
    metric=llm_judge_metric, 
    auto="light", 
    prompt_model=judge_lm, 
    teacher_settings=dict(lm=judge_lm)
    ) # auto can be set as light, medium, heavy

    
def build_trainset(queries: List[str]):
    trainset = []
    import random
    from google_cse import GoogleCSE
    print(os.environ["GOOGLE_CSE_API_KEY"])
    google_client = GoogleCSE(
        api_key=os.environ["GOOGLE_CSE_API_KEY"],
        search_engine_id=os.environ["GOOGLE_CSE_CX"],
    )
    
    for i in range(len(queries)):
        n = random.randint(1, 10)
        question = queries[i]
        search_results = google_client.web_search(question, num_results=n)
        context = "\n".join([f"{i + 1}. {result.snippet}" for i, result in enumerate(search_results)])
        print(context)
        trainset.append(dspy.Example(question=question, context=context).with_inputs("question", "context"))
        import time
        time.sleep(5)
    return trainset


def load_queries(file_path: str):
    with open(file_path, "r") as f:
        queryies = f.readlines()
    return queryies


dataset = load_dataset(
    "json",
    data_files="data/queries.jsonl",
    split="train"
)
print(dataset["query"])
if os.path.exists("data/trainset.jsonl"):
    trainset = load_trainset("data/trainset.jsonl")
else: 
    trainset = build_trainset(list(dataset["query"]))
    save_trainset(trainset, "data/trainset.jsonl")


#compiled_program = tp.compile(GoogleRAg, trainset=dspy.Example(
#
#))

