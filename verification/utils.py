import json
import string
import random

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def generate_short_id(length=8):
    """Generate a short random ID."""
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))