import torch

# --- Model Names ---
# Using a smaller Gemma variant for feasibility, adjust as needed
# Make sure you have accepted the Gemma license on Hugging Face Hub
GEMMA_MODEL_NAME = "model/gemma-3-1b-it" # Using Gemma3-ib-it. Adjust if specific checkpoint exists.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Sentence Transformer for retrieval embeddings
REWARD_MODEL_NAME = 'bert-base-uncased'    # Base for reward model (can be fine-tuned)

# --- Dataset ---
DATASET_NAME = "dataset/longmemeval_data" # Using the specified dataset hub name
# You might need to specify a subset/config if the dataset has multiple ones
DATASET_CONFIG = None # e.g., 'config_name' if applicable
DATASET_QUERY_FIELD = 'query' # Adjust if field names are different
DATASET_ANSWER_FIELD = 'answer' # Adjust if field names are different

# --- Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITERATIONS = 3           # Memory retrieval iterations per query
EPOCHS = 3
BATCH_SIZE = 4             # Adjust based on GPU memory
ACTOR_LR = 1e-5
REWARD_LR = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2 # Accumulate gradients for larger effective batch size
MAX_GRAD_NORM = 1.0        # Gradient clipping

# --- Model Dimensions ---
ACTOR_HIDDEN_DIM = 256     # Hidden layer size for the Actor MLP

# --- Paths ---
MODEL_SAVE_DIR = "./trained_models"
CHECKPOINT_FILENAME_ACTOR = "actor_checkpoint.pth"
CHECKPOINT_FILENAME_REWARD = "reward_checkpoint.pth"

# --- Generation Parameters (for Gemma) ---
GEMMA_MAX_NEW_TOKENS = 150
GEMMA_TEMPERATURE = 0.7
GEMMA_TOP_K = 50

# --- Evaluation ---
EVAL_STEPS = 100 # Evaluate every N steps

# --- Misc ---
SEED = 42