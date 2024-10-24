import os 
from datasets import load_dataset

def get_dataset_path():
    return os.path.expanduser("~/f24-nvidia/ta-benchmarking/benchmark_generation/questions_200_harness_formatting.json")

def process_docs(dataset):
    def _process_doc(doc):
        return {
            "question": "The following are multiple choice questions (with answers)\n\n" + doc["question"] + f"\n{doc['choices'][0]}\n{doc['choices'][1]}\n{doc['choices'][2]}\n{doc['choices'][3]}",  # question text
            "choices": ["A", "B", "C", "D"],  # list of answer choice texts
            "answer": ["A", "B", "C", "D"].index(doc["answer"]),  # index of correct answer
        }

    return dataset.map(_process_doc)
