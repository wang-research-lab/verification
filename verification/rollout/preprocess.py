from datasets import load_dataset, concatenate_datasets, Value

def load_verification_training_problems():
    dataset = load_dataset("WangResearchLab/verification-training-problems")["train"]
    return dataset

def load_aime2024():
    dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    dataset = dataset.rename_column("ID", "problem_id")
    dataset = dataset.map(lambda x: {"source": "aime2024"})
    dataset = dataset.rename_column("Problem", "prompt")
    dataset = dataset.map(lambda x: {"prompt": f"{x['prompt']} Please reason step by step, and put your final answer within \\boxed{{}}."})
    dataset = dataset.rename_column("Answer", "gold_standard_solution")
    dataset = dataset.cast_column("gold_standard_solution", Value("string"))
    dataset = dataset.remove_columns(["Solution"])
    return dataset

def load_aime2025():
    dataset_1 = load_dataset("opencompass/AIME2025", "AIME2025-I")["test"]
    dataset_1 = dataset_1.map(lambda x, idx: {"problem_id": f"2025-I-{idx+1}"}, with_indices=True)
    dataset_2 = load_dataset("opencompass/AIME2025", "AIME2025-II")["test"]
    dataset_2 = dataset_2.map(lambda x, idx: {"problem_id": f"2025-II-{idx+1}"}, with_indices=True)
    dataset = concatenate_datasets([dataset_1, dataset_2])
    dataset = dataset.map(lambda x: {"source": "aime2025"})
    dataset = dataset.rename_column("question", "prompt")
    dataset = dataset.map(lambda x: {"prompt": f"{x['prompt']} Please reason step by step, and put your final answer within \\boxed{{}}."})
    dataset = dataset.rename_column("answer", "gold_standard_solution")
    return dataset

def load_livebench_math():
    dataset = load_dataset("livebench/math")["test"]
    dataset = dataset.rename_column("question_id", "problem_id")
    dataset = dataset.map(lambda x: {"source": "livebench-math"})
    dataset = dataset.map(lambda x: {"prompt": x["turns"][0]})
    dataset = dataset.rename_column("ground_truth", "gold_standard_solution")
    dataset = dataset.remove_columns(["category", "turns", "task", "subtask", "livebench_release_date", "livebench_removal_date", "expressions", "release_date", "year", "hardness"])
    return dataset

def load_gpqa():
    dataset = load_dataset("jeggers/gpqa_formatted", "diamond")["train"]
    dataset = dataset.map(lambda x, idx: {"problem_id": idx}, with_indices=True)
    def construct_prompt(x):
        prompt = f"{x['Question']}\n"
        for i, option in enumerate(x["options"]):
            prompt += f"{chr(ord('A') + i)}. {option}\n"
        prompt += "Please reason step by step, and put your final answer within \\boxed{{}}."
        return {"prompt": prompt}
    dataset = dataset.map(construct_prompt)
    dataset = dataset.map(lambda x: {"gold_standard_solution": ["A", "B", "C", "D"][x["answer"]]})
    dataset = dataset.map(lambda x: {"source": "gpqa"})
    dataset = dataset.remove_columns(["Question", "options", "answer", "Canary String"])
    return dataset

def load_verification_training_data():
    dataset = load_dataset("WangResearchLab/verification-training-data")["train"]
    return dataset

def load_verification_evaluation_data(dataset_split):
    dataset = load_dataset("WangResearchLab/verification-evaluation-data", split=dataset_split)
    return dataset

def load_preprocessed_dataset(dataset_name, dataset_split=""):
    if dataset_name == "verification-training-problems":
        return load_verification_training_problems()
    elif dataset_name == "aime2024":
        return load_aime2024()
    elif dataset_name == "aime2025":
        return load_aime2025()
    elif dataset_name == "gpqa":
        return load_gpqa()
    elif dataset_name == "livebench-math":
        return load_livebench_math()
    elif dataset_name == "verification-training-data":
        return load_verification_training_data()
    elif dataset_name == "verification-evaluation-data":
        return load_verification_evaluation_data(dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")