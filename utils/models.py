import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional, List, Dict, Any

import config
from utils import get_embedding # Use utility function

class Actor(nn.Module):
    """Decides whether to query Individual or Shared Memory."""
    def __init__(self, embedding_model: SentenceTransformer, hidden_dim: int = config.ACTOR_HIDDEN_DIM):
        super().__init__()
        self.embedding_model = embedding_model
        input_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), # 2 actions: 0=Individual, 1=Shared
            # Softmax is implicitly handled by Categorical distribution later
        ).to(config.DEVICE)
        print(f"Initialized Actor with input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, query: str, context_embedding: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """
        Takes a query string, gets its embedding, and predicts action probabilities.

        Args:
            query (str): The input query text.
            context_embedding (Optional[torch.Tensor]): Optional context embedding (not used in this version).

        Returns:
            Tuple[int, torch.Tensor]: action index (0 or 1) and its log probability.
        """
        # Get query embedding using the passed model instance
        query_embedding = get_embedding(query, self.embedding_model).to(config.DEVICE)

        # TODO: Potentially incorporate context_embedding if needed
        input_state = query_embedding

        logits = self.network(input_state)
        probs = torch.softmax(logits, dim=-1)

        # Sample action based on probabilities
        dist = Categorical(probs=probs) # Use probs directly for Categorical
        action = dist.sample()         # action = 0 (Individual) or 1 (Shared)
        log_prob = dist.log_prob(action) # Needed for REINFORCE training

        return action.item(), log_prob


class RewardModel(nn.Module):
    """
    Assigns a score potentially based on query, memory chain, and maybe a generated response.
    In this version, it's trained to predict the heuristic reward (chain relevance to ground truth answer).
    """
    def __init__(self, model_name: str = config.REWARD_MODEL_NAME, num_labels: int = 1):
        super().__init__()
        try:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) # num_labels=1 for regression
            print(f"Initialized Reward Model based on {model_name}")
        except Exception as e:
            print(f"Error loading reward model '{model_name}': {e}")
            raise

    def prepare_input(self, query: str, memory_chain: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """ Formats the input for the reward model. """
        chain_content = " | ".join([mem['content'] for mem in memory_chain])
        # Simple format: Query [SEP] Memory Chain Content
        input_text = f"Query: {query} [SEP] Chain: {chain_content}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512 # Standard max length for BERT-like models
        )
        return inputs

    def forward(self, query: str, memory_chain: List[Dict[str, Any]]) -> torch.Tensor:
        """ Predicts the reward score for the given query and memory chain. """
        inputs = self.prepare_input(query, memory_chain)
        # Move inputs to the correct device
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        with torch.no_grad(): # Usually run reward model in eval mode during RL training step
             outputs = self.model(**inputs)

        # Output is typically logits. Squeeze if num_labels=1 for regression.
        reward_prediction = outputs.logits.squeeze()
        return reward_prediction