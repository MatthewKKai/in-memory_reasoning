import torch
import argparse
import os

import config
from system import MemoryAugmentedLLM
from utils import seed_everything

def run_single_inference(query: str, actor_checkpoint: str, reward_checkpoint: str, individual_mem: list = None, shared_mem: list = None):
    """Runs inference for a single query using saved models."""
    print("--- Running Single Inference ---")
    seed_everything(config.SEED)

    # --- Initialize System ---
    # Load system architecture (will load Gemma again)
    system = MemoryAugmentedLLM(n_iterations=config.N_ITERATIONS)

    # --- Load Trained Actor and Reward Model Weights ---
    actor_path = os.path.join(config.MODEL_SAVE_DIR, actor_checkpoint)
    reward_path = os.path.join(config.MODEL_SAVE_DIR, reward_checkpoint)

    if not os.path.exists(actor_path) or not os.path.exists(reward_path):
        print(f"Error: Checkpoint files not found at:")
        print(f"  Actor: {actor_path}")
        print(f"  Reward: {reward_path}")
        print("Please train the model first or provide correct checkpoint filenames.")
        return

    system.load_models(actor_path, reward_path)
    system.actor.eval()
    system.reward_model.eval()
    if system.llm:
        system.llm.eval()

    # --- Prepare Initial Memory (Optional) ---
    initial_memories = {}
    if individual_mem:
        initial_memories['individual'] = individual_mem
    if shared_mem:
        initial_memories['shared'] = shared_mem

    # --- Run Inference ---
    final_response, memory_chain = system.run_inference(query, initial_memories=initial_memories)

    print("\n--- Inference Complete ---")
    print(f"Query: {query}")
    print(f"Retrieved Memory Chain IDs: {[mem['id'] for mem in memory_chain]}")
    print(f"Final Response:\n{final_response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Memory-Augmented LLM")
    parser.add_argument("query", type=str, help="The input query for the LLM.")
    parser.add_argument(
        "--actor_ckpt",
        type=str,
        default="actor_epoch_3.pth", # Default to last epoch if trained for 3
        help="Filename of the trained Actor checkpoint (e.g., actor_epoch_3.pth) located in MODEL_SAVE_DIR."
    )
    parser.add_argument(
        "--reward_ckpt",
        type=str,
        default="reward_epoch_3.pth", # Default to last epoch if trained for 3
        help="Filename of the trained Reward Model checkpoint (e.g., reward_epoch_3.pth) located in MODEL_SAVE_DIR."
    )
    parser.add_argument(
        "--ind_mem",
        nargs='+', # Allows multiple space-separated strings
        default=None,
        help="Optional initial individual memories (strings)."
    )
    parser.add_argument(
        "--shared_mem",
        nargs='+', # Allows multiple space-separated strings
        default=None,
        help="Optional initial shared memories (strings)."
    )

    args = parser.parse_args()

    run_single_inference(
        query=args.query,
        actor_checkpoint=args.actor_ckpt,
        reward_checkpoint=args.reward_ckpt,
        individual_mem=args.ind_mem,
        shared_mem=args.shared_mem
    )

    # Example command:
    # python inference.py "What is my favorite color and what is the capital of France?" --actor_ckpt actor_epoch_1.pth --reward_ckpt reward_epoch_1.pth --ind_mem "My favorite color is green." --shared_mem "Paris is the capital of France."