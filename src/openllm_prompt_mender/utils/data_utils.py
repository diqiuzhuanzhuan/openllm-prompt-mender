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

import json
import dspy
from typing import List

# 1. Save the trainset to a local JSONL file

def save_trainset(trainset: List[dspy.Example], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for ex in trainset:
            # Convert dspy.Example to a dictionary using .toDict()
            # This captures all fields like question, context, answer, etc.
            f.write(json.dumps(ex.toDict(), ensure_ascii=False) + '\n')

# 2. Load the trainset back from the JSONL file
def load_trainset(file_path: str, input_keys: tuple=("question", "context")) -> List[dspy.Example]:
    loaded_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Reconstruct the dspy.Example object
            # Use .with_inputs() to specify which fields are inputs [4, 5]
            example = dspy.Example(**data).with_inputs(*input_keys)
            loaded_data.append(example)
    return loaded_data
