# evaluation.py

import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset, DatasetDict, Dataset
from tqdm.auto import tqdm
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple, Any

# Evaluation Metrics Libraries
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

# Project Modules
import config
from system import MemoryAugmentedLLM
from utils import (
    load_longmemeval_dataset,
    preprocess_longmemeval,
    seed_everything,
    get_embedding_model, # Need embedding model for heuristic eval
    ensure_nltk_data
)

# --- Configuration ---
DEFAULT_SIMILARITY_THRESHOLD = 0.6 # Threshold for heuristic retrieval relevance
DEFAULT_EVAL_SPLIT = 'validation' # Or 'test', depending on dataset structure
DEFAULT_MAX_EVAL_SAMPLES = 100 # Limit evaluation samples for speed, set to None for full eval

# --- Helper Functions for Metrics ---

def calculate_answer_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculates ROUGE and BLEU scores."""
    print("Calculating Answer Quality Metrics (ROUGE, BLEU)...")
    if not predictions or not references or len(predictions) != len(references):
        print("Warning: Invalid input for metric calculation.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu4": 0.0}

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        # Handle empty strings gracefully
        pred = pred if pred and pred.strip() else "empty"
        ref = ref if ref and ref.strip() else "empty"
        score = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(score['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(score['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(score['rougeL'].fmeasure)

    avg_rouge = {k: np.mean(v) * 100 for k, v in rouge_scores.items()} # Report as percentage

    # BLEU-4
    # NLTK expects tokenized input
    smoothie = SmoothingFunction().method4 # Smoothing for short sentences
    bleu_scores = []
    for pred, ref in zip(predictions, references):
         # Handle empty strings gracefully
        pred_tokens = nltk.word_tokenize(pred) if pred and pred.strip() else []
        ref_tokens = nltk.word_tokenize(ref) if ref and ref.strip() else []
        if not pred_tokens or not ref_tokens:
            bleu_scores.append(0.0) # Assign 0 if prediction or reference is empty
            continue
        # NLTK expects reference to be list of lists of tokens
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    avg_bleu4 = np.mean(bleu_scores) * 100 # Report as percentage

    print("Answer Metrics Calculated.")
    return {**avg_rouge, "bleu4": avg_bleu4}


def is_retrieval_relevant_heuristic(retrieved_item_content: str, ground_truth_answer: str, embedding_model, similarity_threshold: float) -> bool:
    """Determines if a retrieved item is heuristically relevant to the ground truth answer."""
    if not retrieved_item_content or not ground_truth_answer:
        return False

    try:
        with torch.no_grad():
            # Get embeddings (ensure they are on CPU for sklearn similarity)
            item_embedding = embedding_model.encode(retrieved_item_content, convert_to_tensor=True).cpu().unsqueeze(0)
            answer_embedding = embedding_model.encode(ground_truth_answer, convert_to_tensor=True).cpu().unsqueeze(0)

            # Calculate cosine similarity
            similarity = cosine_similarity(item_embedding.numpy(), answer_embedding.numpy())[0, 0]

        return similarity >= similarity_threshold
    except Exception as e:
        # Handle potential errors during encoding or similarity calculation
        # print(f"Warning: Error calculating heuristic relevance: {e}") # Can be verbose
        return False


def calculate_retrieval_metrics(
    memory_chains: List[List[Dict[str, Any]]],
    ground_truth_answers: List[str],
    embedding_model,
    similarity_threshold: float
) -> Dict[str, float]:
    """Calculates heuristic Precision and Hit Rate for memory retrieval."""
    print(f"Calculating Heuristic Retrieval Metrics (Threshold: {similarity_threshold})...")
    total_precision = 0.0
    total_items_retrieved = 0
    relevant_items_retrieved_count = 0
    hit_count = 0 # Number of queries where at least one relevant item was retrieved
    valid_chains_count = 0 # Count chains that actually retrieved something

    if not memory_chains or not ground_truth_answers or len(memory_chains) != len(ground_truth_answers):
         print("Warning: Invalid input for retrieval metric calculation.")
         return {"retrieval_precision": 0.0, "retrieval_hit_rate": 0.0}


    for chain, answer in zip(memory_chains, ground_truth_answers):
        if not chain: # Skip if no memory items were retrieved for this query
            continue

        valid_chains_count += 1
        num_retrieved = len(chain)
        num_relevant_in_chain = 0
        chain_had_hit = False

        for item in chain:
            content = item.get('content', '')
            is_relevant = is_retrieval_relevant_heuristic(content, answer, embedding_model, similarity_threshold)
            if is_relevant:
                num_relevant_in_chain += 1
                chain_had_hit = True

        precision = num_relevant_in_chain / num_retrieved if num_retrieved > 0 else 0.0
        total_precision += precision
        relevant_items_retrieved_count += num_relevant_in_chain
        total_items_retrieved += num_retrieved
        if chain_had_hit:
            hit_count += 1

    # Calculate overall averages
    # Average precision over queries that had retrievals
    avg_precision = (total_precision / valid_chains_count) * 100 if valid_chains_count > 0 else 0.0
    # Hit rate over all queries (including those with no retrievals)
    avg_hit_rate = (hit_count / len(memory_chains)) * 100 if memory_chains else 0.0

    print("Retrieval Metrics Calculated.")
    return {
        "retrieval_precision": avg_precision, # Avg precision per query chain
        "retrieval_hit_rate": avg_hit_rate    # % of queries retrieving >=1 relevant item
    }


# --- Main Evaluation Function ---

@torch.no_grad() # Disable gradients for evaluation
def evaluate_system(
    system: MemoryAugmentedLLM,
    eval_dataset: Dataset,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_samples: Optional[int] = DEFAULT_MAX_EVAL_SAMPLES
) -> Dict[str, float]:
    """Runs the evaluation loop and computes all metrics."""

    print("\n--- Starting Evaluation ---")
    system.actor.eval()
    system.reward_model.eval()
    if system.llm:
        system.llm.eval()
    else:
        print("Warning: LLM (Gemma) not loaded in the system. Cannot evaluate answer quality.")
        # Decide whether to proceed with only retrieval eval or stop
        # For now, let's proceed but answer quality will be 0

    # Prepare dataset subset if needed
    if max_samples is not None and max_samples < len(eval_dataset):
        print(f"Evaluating on a subset of {max_samples} samples.")
        eval_subset = Subset(eval_dataset, range(max_samples))
        eval_loader = DataLoader(eval_subset, batch_size=config.BATCH_SIZE) # Use config batch size
    else:
        print(f"Evaluating on the full dataset ({len(eval_dataset)} samples).")
        eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE) # Use config batch size


    all_predictions = []
    all_references = []
    all_memory_chains = []

    progress_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        queries = batch['query']
        ground_truth_answers = batch['ground_truth_answer']

        # Process batch item by item (as run_inference handles one query)
        for i in range(len(queries)):
            query = queries[i]
            reference = ground_truth_answers[i]

            if not query or not reference:
                 print(f"Warning: Skipping evaluation for empty query or reference.")
                 continue # Skip this sample

            # Run inference for the single query
            # Note: run_inference clears memory each time by default
            final_response, memory_chain = system.run_inference(query)

            all_predictions.append(final_response)
            all_references.append(reference)
            all_memory_chains.append(memory_chain)

    # --- Calculate Metrics ---
    results = {}

    # Answer Quality Metrics (if LLM was loaded)
    if system.llm:
        answer_metrics = calculate_answer_metrics(all_predictions, all_references)
        results.update(answer_metrics)
    else:
         results.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu4": 0.0})


    # Retrieval Quality Metrics (Heuristic)
    # Requires the embedding model used during retrieval eval
    embedding_model = get_embedding_model() # Get the shared embedding model
    retrieval_metrics = calculate_retrieval_metrics(
        all_memory_chains,
        all_references, # Use references (ground truth answers) for heuristic
        embedding_model,
        similarity_threshold
    )
    results.update(retrieval_metrics)

    print("--- Evaluation Finished ---")
    return results


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Memory-Augmented LLM")
    parser.add_argument(
        "--actor_ckpt",
        type=str,
        required=True,
        help="Filename of the trained Actor checkpoint (e.g., actor_epoch_3.pth) located in MODEL_SAVE_DIR."
    )
    parser.add_argument(
        "--reward_ckpt",
        type=str,
        required=True,
        help="Filename of the trained Reward Model checkpoint (e.g., reward_epoch_3.pth) located in MODEL_SAVE_DIR."
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=DEFAULT_EVAL_SPLIT,
        help=f"Dataset split to use for evaluation (default: {DEFAULT_EVAL_SPLIT})."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for heuristic retrieval relevance (default: {DEFAULT_SIMILARITY_THRESHOLD})."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_EVAL_SAMPLES,
        help=f"Maximum number of samples to evaluate (default: {DEFAULT_MAX_EVAL_SAMPLES}, use -1 for all)."
    )

    args = parser.parse_args()

    # Ensure NLTK data is available
    ensure_nltk_data()
    seed_everything(config.SEED)

    # --- Initialize System ---
    print("Initializing system for evaluation...")
    system = MemoryAugmentedLLM(n_iterations=config.N_ITERATIONS)

    # --- Load Trained Models ---
    actor_path = os.path.join(config.MODEL_SAVE_DIR, args.actor_ckpt)
    reward_path = os.path.join(config.MODEL_SAVE_DIR, args.reward_ckpt)

    if not os.path.exists(actor_path) or not os.path.exists(reward_path):
        print(f"Error: Checkpoint files not found at:")
        print(f"  Actor: {actor_path}")
        print(f"  Reward: {reward_path}")
        exit(1)

    try:
        system.load_models(actor_path, reward_path)
    except Exception as e:
        print(f"Failed to load models: {e}")
        exit(1)


    # --- Load Evaluation Dataset ---
    print(f"Loading evaluation dataset split: '{args.eval_split}'")
    try:
        eval_dataset_full = load_longmemeval_dataset(split=args.eval_split)
        # Apply preprocessing
        eval_dataset_processed = eval_dataset_full.map(
            preprocess_longmemeval,
            remove_columns=eval_dataset_full.column_names
        )
    except ValueError as e:
        print(f"Error loading or processing dataset split '{args.eval_split}': {e}")
        exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during dataset loading: {e}")
         exit(1)


    # --- Run Evaluation ---
    max_eval_samples = None if args.max_samples == -1 else args.max_samples
    evaluation_results = evaluate_system(
        system=system,
        eval_dataset=eval_dataset_processed,
        similarity_threshold=args.threshold,
        max_samples=max_eval_samples
    )

    # --- Print Results ---
    print("\n" + "="*30)
    print("      Evaluation Results")
    print("="*30)
    print(f"Dataset Split:         {args.eval_split}")
    print(f"Max Samples:           {'All' if max_eval_samples is None else max_eval_samples}")
    print(f"Actor Checkpoint:      {args.actor_ckpt}")
    print(f"Reward Checkpoint:     {args.reward_ckpt}")
    print(f"Retrieval Threshold:   {args.threshold:.2f}")
    print("-"*30)
    print("Answer Quality:")
    print(f"  ROUGE-1 F1:          {evaluation_results.get('rouge1', 0.0):.2f}")
    print(f"  ROUGE-2 F1:          {evaluation_results.get('rouge2', 0.0):.2f}")
    print(f"  ROUGE-L F1:          {evaluation_results.get('rougeL', 0.0):.2f}")
    print(f"  BLEU-4:              {evaluation_results.get('bleu4', 0.0):.2f}")
    print("-"*30)
    print("Heuristic Retrieval Quality:")
    print(f"  Precision (%):       {evaluation_results.get('retrieval_precision', 0.0):.2f}")
    print(f"  Hit Rate (%):        {evaluation_results.get('retrieval_hit_rate', 0.0):.2f}")
    print("="*30)

    # Example command:
    # python evaluation.py --actor_ckpt actor_epoch_3.pth --reward_ckpt reward_epoch_3.pth --eval_split validation --max_samples 100 --threshold 0.6