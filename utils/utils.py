import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, DatasetDict, Dataset
from typing import Dict, Any, Optional
import numpy as np
import random
import nltk
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

import config # Import config variables

# --- Seeding ---
def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Embedding Helper ---
# Cache the model loading
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
    return _embedding_model

def get_embedding(text: str, model: SentenceTransformer) -> torch.Tensor:
    """Generates embedding for a given text using the provided SentenceTransformer model."""
    # Ensure embedding is on the correct device and detached
    with torch.no_grad():
        embedding = model.encode(text, convert_to_tensor=True, device=config.DEVICE)
    return embedding.unsqueeze(0) # Add batch dimension (1, embedding_dim)

def get_embedding_dim(model: SentenceTransformer) -> int:
    """Gets the embedding dimension of the SentenceTransformer model."""
    return model.get_sentence_embedding_dimension()

# --- Dataset Loading & Preprocessing ---
def load_longmemeval_dataset(split: Optional[str] = None) -> DatasetDict | Dataset:
    """Loads the LongMemEval dataset from Hugging Face Hub."""
    print(f"Loading dataset: {config.DATASET_NAME}")
    try:
        # Adjust 'name' if the dataset requires a specific configuration
        dataset = load_dataset(config.DATASET_NAME, name=config.DATASET_CONFIG)
        print("Dataset loaded successfully.")
        if split and isinstance(dataset, DatasetDict):
            if split in dataset:
                return dataset[split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
        elif split and isinstance(dataset, Dataset):
             print(f"Warning: Dataset is not a DatasetDict, returning the whole dataset for split '{split}'.")
             return dataset
        return dataset # Return full DatasetDict if no split specified
    except Exception as e:
        print(f"Error loading dataset '{config.DATASET_NAME}': {e}")
        print("Please ensure the dataset name is correct and you have internet access.")
        print("You might need to authenticate with Hugging Face CLI: `huggingface-cli login`")
        raise

def preprocess_longmemeval(example: Dict[str, Any]) -> Dict[str, Any]:
    """Basic preprocessing: Extracts query and answer. Adapt if needed."""
    # Check if expected fields exist
    if config.DATASET_QUERY_FIELD not in example or config.DATASET_ANSWER_FIELD not in example:
         raise ValueError(f"Dataset example missing required fields: '{config.DATASET_QUERY_FIELD}' or '{config.DATASET_ANSWER_FIELD}'. Found keys: {list(example.keys())}")

    return {
        "query": example[config.DATASET_QUERY_FIELD],
        "ground_truth_answer": example[config.DATASET_ANSWER_FIELD],
        # 'memory_evidence' is NOT directly available in LongMemEval
        # We'll calculate reward based on chain quality vs. ground_truth_answer later
    }

# --- Reward Calculation Heuristic (Example) ---
# This function calculates a simple reward based on semantic similarity
# between retrieved memory content and the ground truth answer.
# This is a heuristic as LongMemEval doesn't provide ground truth memory chains.
def calculate_heuristic_reward(memory_chain: list[Dict[str, Any]], ground_truth_answer: str, embedding_model) -> torch.Tensor:
    """
    Calculates a heuristic reward based on the relevance of the memory chain
    to the ground truth answer using embedding similarity.
    """
    if not memory_chain:
        return torch.tensor([0.0], device=config.DEVICE) # No memory retrieved, zero reward

    # Combine content of the memory chain
    chain_content = " ".join([mem['content'] for mem in memory_chain])

    if not chain_content.strip() or not ground_truth_answer.strip():
        return torch.tensor([0.0], device=config.DEVICE) # Avoid errors with empty strings

    try:
        # Get embeddings (use the utility function's model)
        with torch.no_grad():
            chain_embedding = embedding_model.encode(chain_content, convert_to_tensor=True, device=config.DEVICE)
            answer_embedding = embedding_model.encode(ground_truth_answer, convert_to_tensor=True, device=config.DEVICE)

        # Calculate cosine similarity
        # Use sklearn's implementation for potentially better numerical stability on CPU/GPU tensor inputs
        similarity = sklearn_cosine_similarity(
            chain_embedding.unsqueeze(0).cpu().numpy(), # Move to CPU and add batch dim
            answer_embedding.unsqueeze(0).cpu().numpy()  # Move to CPU and add batch dim
        )[0, 0] # Get the scalar value

        # Normalize similarity to [0, 1] range (cosine similarity is [-1, 1])
        reward_val = (similarity + 1.0) / 2.0
        # Clamp reward to avoid potential numerical issues
        reward_val = max(0.0, min(1.0, reward_val))

    except Exception as e:
        print(f"Warning: Error calculating heuristic reward: {e}")
        reward_val = 0.0 # Default to 0 reward on error

    return torch.tensor([reward_val], dtype=torch.float32, device=config.DEVICE)

# --- NLTK Data Download (for ROUGE) ---
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
    except LookupError:
         print("NLTK 'punkt' tokenizer not found. Downloading...")
         nltk.download('punkt', quiet=True)