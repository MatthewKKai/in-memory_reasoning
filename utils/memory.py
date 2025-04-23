import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional
import random

import config
from utils import get_embedding # Use the utility function

class MemoryModule:
    """Base class for memory storage."""
    def __init__(self, embedding_model: SentenceTransformer):
        self.memory: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model # Use passed model

    @torch.no_grad()
    def add(self, content: str, item_id: Optional[str] = None):
        if not content or not content.strip():
             print(f"Warning: Attempted to add empty content to {self.__class__.__name__}. Skipping.")
             return
        if item_id is None:
            item_id = f"mem_{len(self.memory)}_{random.randint(1000, 9999)}"
        if any(item['id'] == item_id for item in self.memory):
            print(f"Warning: Memory item with ID {item_id} already exists in {self.__class__.__name__}. Skipping.")
            return

        # Use the passed embedding model instance
        embedding = get_embedding(content, self.embedding_model)
        self.memory.append({'id': item_id, 'content': content, 'embedding': embedding.cpu()}) # Store embeddings on CPU to save GPU memory
        # print(f"Added memory item {item_id} to {self.__class__.__name__}") # Reduce verbosity

    @torch.no_grad()
    def retrieve(self, query_embedding: torch.Tensor, k: int = 1) -> List[Dict[str, Any]]:
        if not self.memory:
            return []

        # Move memory embeddings to device for calculation
        memory_embeddings = torch.cat([item['embedding'].to(config.DEVICE) for item in self.memory], dim=0)
        query_embedding_dev = query_embedding.to(config.DEVICE) # Ensure query embedding is on device

        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding_dev, memory_embeddings, dim=1)

        # Get top-k indices
        # Ensure k is not larger than the number of items in memory
        actual_k = min(k, len(self.memory))
        if actual_k == 0:
             return []
        top_k_indices = torch.topk(similarities, k=actual_k, dim=0).indices

        # Return the corresponding memory items
        return [self.memory[i] for i in top_k_indices.cpu().tolist()] # Indices back to cpu

    def delete(self, item_id: str):
        initial_len = len(self.memory)
        self.memory = [item for item in self.memory if item['id'] != item_id]
        if len(self.memory) < initial_len:
            print(f"Deleted memory item {item_id} from {self.__class__.__name__}")

    def clear(self):
        print(f"Clearing all memory from {self.__class__.__name__}")
        self.memory = []

    def summarize(self):
        print(f"Summarization not implemented for {self.__class__.__name__}")
        pass

    def merge(self):
        print(f"Merging not implemented for {self.__class__.__name__}")
        pass

    def get_all_content(self) -> List[str]:
         return [item['content'] for item in self.memory]

    def __len__(self) -> int:
        return len(self.memory)


class IndividualMemory(MemoryModule):
    def __init__(self, embedding_model: SentenceTransformer):
        super().__init__(embedding_model)
        print("Initialized Individual Memory")

class SharedMemory(MemoryModule):
    def __init__(self, embedding_model: SentenceTransformer):
        super().__init__(embedding_model)
        print("Initialized Shared Memory")
        self.reflection_guidance_data = {} # Placeholder

    def update_reflection_guidance(self, data: Dict):
        self.reflection_guidance_data.update(data)
        print("Updated Shared Memory reflection/guidance data.")