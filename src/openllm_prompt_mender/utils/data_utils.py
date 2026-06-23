# Copyright (c) 2025 Loong Ma
# SPDX-License-Identifier: MIT

import json
import random
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
            random.shuffle(loaded_data)

    return loaded_data
