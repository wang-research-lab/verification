import argparse
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from tqdm import tqdm
from verification.utils import load_jsonl, save_jsonl
from verification.metrics import compute_equiv_matrix

@contextmanager
def ignore_output():
    """
    A context manager that redirects sys.stdout and sys.stderr to os.devnull,
    effectively ignoring any output.
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    data = load_jsonl(args.input_path)
    
    for item in tqdm(data):
        with ignore_output():
            item["equiv_matrix"] = compute_equiv_matrix(item)
    
    save_jsonl(args.output_path, data)