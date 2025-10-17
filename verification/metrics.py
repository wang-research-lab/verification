import math
import random

import numpy as np
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

def compute_equiv_matrix(instance):
    responses = instance["llm_responses"]
    m = len(responses)
    
    preds = [
        parse(
            response["response"], extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        ) for response in responses
    ]
    
    equiv_matrix = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            equiv_matrix[i, j] = verify(preds[i], preds[j])
            
    return (equiv_matrix | equiv_matrix.T).tolist()

def pass_at_n(instance, indices):
    responses = [instance["llm_responses"][i] for i in indices]
    return any(x["correct"] for x in responses)

def cons_at_n(instance, indices):
    equiv_matrix = instance.get("equiv_matrix", None)
    if equiv_matrix is None:
        equiv_matrix = compute_equiv_matrix(instance) 
        
    equiv_matrix = np.array(equiv_matrix)
    indices_array = np.array(list(indices))
    
    sub_matrix = equiv_matrix[np.ix_(indices_array, indices_array)]
    row_sums_in_subset = sub_matrix.sum(axis=1)
    local_argmax_idx = np.argmax(row_sums_in_subset)
    original_idx_for_argmax = indices_array[local_argmax_idx]
    is_correct = instance["llm_responses"][original_idx_for_argmax]["correct"]
        
    return is_correct

def verify_at_n(instance, indices):
    responses = [instance["llm_responses"][i] for i in indices]
    chosen = max(responses, key=lambda x: x["score"][0])
    return chosen["correct"]

def weighted_cons_at_n(instance, indices):
    equiv = instance.get("equiv_matrix")
    if equiv is None:
        equiv = compute_equiv_matrix(instance)
    equiv = np.array(equiv)

    idxs = list(indices)
    # Count multiplicities from sampling with replacement
    counts = {}
    for i in idxs:
        counts[i] = counts.get(i, 0) + 1
    visited, clusters = set(), []
    for i in idxs:
        if i in visited:
            continue
        stack, group = [i], []
        visited.add(i)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for j in idxs:
                if j not in visited and equiv[cur, j]:
                    visited.add(j)
                    stack.append(j)
        clusters.append(group)

    best_val, best_corr = -float("inf"), False
    for group in clusters:
        # Weighted sum of verification scores with multiplicity weights
        weighted_sum = 0.0
        for i in group:
            w = counts.get(i, 0)
            if w == 0:
                continue
            scores = instance["llm_responses"][i]["score"]
            weighted_sum += w * sum(scores)

        if weighted_sum > best_val:
            best_val = weighted_sum
            best_corr = any(instance["llm_responses"][i]["correct"] for i in group)

    return best_corr

def pess_at_n(instance, indices, alpha):
    # 1) Get or compute the equivalence matrix
    equiv = instance.get("equiv_matrix")
    if equiv is None:
        equiv = compute_equiv_matrix(instance)
    equiv = np.array(equiv)

    # 2) Build clusters of equivalent indices
    idxs = list(indices)
    # Count multiplicities from sampling with replacement
    counts = {}
    for i in idxs:
        counts[i] = counts.get(i, 0) + 1
    visited, clusters = set(), []
    for i in idxs:
        if i in visited:
            continue
        stack, group = [i], []
        visited.add(i)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for j in idxs:
                if j not in visited and equiv[cur, j]:
                    visited.add(j)
                    stack.append(j)
        clusters.append(group)

    # Use total draws for N to reflect multiplicities
    N = len(indices)
    logN = math.log(N)
    best_val, best_corr = -float("inf"), False

    # 3) Evaluate each cluster
    for group in clusters:
        # Weighted empirical mean over verification scores using multiplicities
        numerator = 0.0
        denom = 0
        for i in group:
            w = counts.get(i, 0)
            if w == 0:
                continue
            scores = instance["llm_responses"][i]["score"]
            numerator += w * sum(scores)
            denom += w * len(scores)

        r = (numerator / denom) if denom > 0 else 0.0

        # Effective cluster weight accounts for multiplicity
        Na = sum(counts.get(i, 0) for i in group) + 1
        penalty = alpha * (logN / Na)
        val = r - penalty

        if val > best_val:
            best_val = val
            best_corr = any(
                instance["llm_responses"][i]["correct"] for i in group
            )

    return best_corr

def monte_carlo_at_n(instances, metric_fn, N, R, seed=42):
    """
    instances (List[Dict[str, Any]]): each with key "llm_responses"
    metric_fn (Callable): takes (instance, tuple_of_indices) → bool
    N (int): number of responses to sample per trial
    R (int): number of random trials to perform
    """
    random.seed(seed)
    np.random.seed(seed)

    per_instance_rates = []
    for instance in instances:
        k = len(instance["llm_responses"])
        pass_count = 0

        # keep drawing until we have R unique samples
        for _ in range(R):
            indices = tuple(random.choices(range(k), k=N))
            if metric_fn(instance, indices):
                pass_count += 1

        per_instance_rates.append(pass_count / R)

    # average across all instances
    return sum(per_instance_rates) / len(per_instance_rates)