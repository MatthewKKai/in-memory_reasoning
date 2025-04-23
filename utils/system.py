import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple, Any, Optional

import config
from memory_modules import IndividualMemory, SharedMemory
from models import Actor, RewardModel
from retriever import CoTEnhancedRetriever
from utils import get_embedding_model, calculate_heuristic_reward # Import necessary utils

# --- Aggregator Function ---
def aggregate_context(current_query: str, retrieved_mems: List[Dict[str, Any]]) -> str:
    """Combines current query/context with retrieved memory for next iteration."""
    if not retrieved_mems:
        return current_query

    # Simple concatenation of last retrieved item's content for context
    new_context = retrieved_mems[-1]['content'] # Use only the last retrieved memory
    # Example update logic: Frame it as continuing the thought process
    updated_query = f"Based on '{new_context[:100]}...', continue thinking about: '{current_query[:100]}...'"
    # Limit length to avoid excessive growth
    max_len = 512
    if len(updated_query) > max_len:
        updated_query = updated_query[:max_len] + "..."

    return updated_query


# --- Orchestrator Class ---
class MemoryAugmentedLLM:
    def __init__(self, n_iterations: int = config.N_ITERATIONS):
        print("Initializing MemoryAugmentedLLM System...")
        self.embedding_model = get_embedding_model() # Load/get shared embedding model instance

        self.individual_memory = IndividualMemory(self.embedding_model)
        self.shared_memory = SharedMemory(self.embedding_model)
        self.memories = [self.individual_memory, self.shared_memory]

        self.actor = Actor(embedding_model=self.embedding_model, hidden_dim=config.ACTOR_HIDDEN_DIM).to(config.DEVICE)
        self.reward_model = RewardModel().to(config.DEVICE) # Uses default from config

        self.retrievers = [
            CoTEnhancedRetriever(self.individual_memory, self.embedding_model),
            CoTEnhancedRetriever(self.shared_memory, self.embedding_model)
        ]
        self.n_iterations = n_iterations

        # Load the final LLM (Gemma)
        print(f"Loading final LLM: {config.GEMMA_MODEL_NAME}")
        # Optional: Quantization for lower memory usage (requires bitsandbytes)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 # Or float16
        )
        try:
             # Trust remote code if needed for Gemma models
            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_MODEL_NAME)
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.GEMMA_MODEL_NAME,
                # quantization_config=quantization_config, # Enable quantization if desired
                device_map="auto", # Automatically distribute across GPUs if available
                torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
                trust_remote_code=True, # Sometimes needed
            )
            print(f"{config.GEMMA_MODEL_NAME} loaded successfully.")
            # Ensure tokenizer has a padding token if it doesn't (like some Gemma versions)
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                self.llm.config.pad_token_id = self.llm.config.eos_token_id

        except Exception as e:
             print(f"Error loading Gemma model '{config.GEMMA_MODEL_NAME}': {e}")
             print("Ensure you have accepted the license and have enough memory/disk space.")
             print("Try running `huggingface-cli login`.")
             # Fallback or raise error
             self.llm = None
             self.llm_tokenizer = None
             raise RuntimeError(f"Failed to load Gemma model: {e}")


    def run_inference(self, query: str, initial_memories: Optional[Dict[str, List[str]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs the iterative retrieval process and generates a final response using Gemma.

        Args:
            query (str): The user's input query.
            initial_memories (Optional[Dict[str, List[str]]]): Pre-populate memories,
                e.g., {'individual': ['fact 1', 'fact 2'], 'shared': ['shared fact']}

        Returns:
            Tuple[str, List[Dict[str, Any]]]: The generated LLM response and the memory chain.
        """
        print(f"\n--- Running Inference for Query: '{query}' ---")
        self.actor.eval() # Set actor to evaluation mode
        self.reward_model.eval() # Set reward model to evaluation mode

        # Clear existing session memory and optionally populate
        self.individual_memory.clear()
        self.shared_memory.clear()
        if initial_memories:
            for content in initial_memories.get('individual', []):
                self.individual_memory.add(content)
            for content in initial_memories.get('shared', []):
                self.shared_memory.add(content)
            print(f"Populated memories: {len(self.individual_memory)} individual, {len(self.shared_memory)} shared.")


        memory_chain = []
        current_context_query = query

        for i in range(self.n_iterations):
            print(f"\nIteration {i+1}/{self.n_iterations}")

            # 1. Actor decides memory type (no gradients needed for inference)
            with torch.no_grad():
                action_idx, _ = self.actor(current_context_query) # Log prob not needed for inference
            selected_memory_type = "Individual" if action_idx == 0 else "Shared"
            print(f"Actor chose: {selected_memory_type} Memory")

            # 2. Retriever fetches memory
            retriever = self.retrievers[action_idx]
            retrieved_mems = retriever.retrieve(current_context_query, k=1)

            if retrieved_mems:
                memory_item = retrieved_mems[0] # Assuming k=1
                memory_chain.append(memory_item)
                print(f"Retrieved: {memory_item['id']} - '{memory_item['content'][:80]}...'")
                # 3. Aggregator updates context for next iteration
                current_context_query = aggregate_context(current_context_query, retrieved_mems)
                # print(f"Updated context/query for next step: '{current_context_query[:100]}...'") # Less verbose
            else:
                print(f"No relevant memory found in {selected_memory_type} module for this iteration.")
                # Optional: Break or try the other memory? For now, just continue.
                # current_context_query = query # Reset context? Or keep it? Keep for now.

        # 4. Final LLM Generation using Gemma
        if self.llm and self.llm_tokenizer:
            final_response = self.generate_llm_response(query, memory_chain)
        else:
            print("LLM not loaded. Returning placeholder response.")
            final_response = "[LLM Response Placeholder - Model Not Loaded]"

        print(f"\nFinal Memory Chain ({len(memory_chain)} items): {[m['id'] for m in memory_chain]}")
        print(f"Final LLM Response: {final_response}")

        return final_response, memory_chain

    def generate_llm_response(self, original_query: str, memory_chain: List[Dict[str, Any]]) -> str:
        """ Generates the final response using the loaded Gemma LLM. """
        if not self.llm or not self.llm_tokenizer:
            return "[LLM Response Placeholder - Model Not Loaded]"

        self.llm.eval() # Ensure LLM is in eval mode

        # Format the prompt for Gemma (specific formatting might be needed based on fine-tuning)
        # Using a generic instruction format here
        chain_bullets = "\n".join([f"- {mem['content']}" for mem in memory_chain])
        prompt = f"""<start_of_turn>user
Use the following retrieved information to answer the query. If the information is not relevant, rely on your general knowledge.

Retrieved Information:
{chain_bullets if chain_bullets else "- None"}

Query: {original_query}<end_of_turn>
<start_of_turn>model
""" # Gemma instruction format

        print("\n--- Generating Final Response with Gemma ---")
        # print(f"Prompt:\n{prompt}") # Can be very long

        inputs = self.llm_tokenizer(prompt, return_tensors="pt", padding=False).to(config.DEVICE)

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": config.GEMMA_MAX_NEW_TOKENS,
            "temperature": config.GEMMA_TEMPERATURE,
            "top_k": config.GEMMA_TOP_K,
            "do_sample": True, # Sample for potentially more creative answers
            "pad_token_id": self.llm_tokenizer.eos_token_id # Use EOS token for padding during generation
        }

        try:
            with torch.no_grad():
                outputs = self.llm.generate(**inputs, **generation_kwargs)

            # Decode the response, skipping the prompt part
            response_text = self.llm_tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response_text.strip()

        except Exception as e:
            print(f"Error during Gemma generation: {e}")
            return f"[LLM Generation Error: {e}]"


    def train_step(self, query: str, ground_truth_answer: str, actor_optimizer: optim.Optimizer, reward_optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Performs one training step: runs retrieval, calculates rewards, updates Actor and Reward Model.

        Args:
            query (str): Input query.
            ground_truth_answer (str): The expected final answer from the dataset.
            actor_optimizer: Optimizer for the Actor model.
            reward_optimizer: Optimizer for the Reward model.

        Returns:
            Tuple[float, float]: Actor loss and Reward Model loss for this step.
        """
        # --- Forward Pass (Inference-like to collect trajectory) ---
        self.actor.train() # Set actor to train mode for gradient calculation
        # Reward model is often kept in eval during actor update phase if using predicted rewards
        self.reward_model.eval()

        memory_chain = []
        actor_log_probs = []
        current_context_query = query
        rewards = [] # Store rewards predicted at each step (or just final)

        for i in range(self.n_iterations):
            # 1. Actor action
            action_idx, log_prob = self.actor(current_context_query)
            actor_log_probs.append(log_prob)
            selected_memory_type = "Individual" if action_idx == 0 else "Shared"

            # 2. Retriever fetches memory (no gradients through retrieval itself)
            with torch.no_grad():
                 retriever = self.retrievers[action_idx]
                 retrieved_mems = retriever.retrieve(current_context_query, k=1)

            if retrieved_mems:
                memory_chain.append(retrieved_mems[0])
                # 3. Aggregator updates context
                current_context_query = aggregate_context(current_context_query, retrieved_mems)
            else:
                # Optional: Penalize if no memory found? For now, just continue.
                pass

            # 4. (Optional) Predict step-wise reward (or use final reward)
            # If predicting step-wise:
            # with torch.no_grad():
            #     step_reward = self.reward_model(query, memory_chain[:i+1])
            #     rewards.append(step_reward)


        # --- Reward Calculation ---
        # Calculate the overall reward for the generated chain based on the ground truth answer
        # using our heuristic (or a trained reward model's prediction)
        final_reward_heuristic = calculate_heuristic_reward(memory_chain, ground_truth_answer, self.embedding_model)
        final_reward_heuristic = final_reward_heuristic.detach() # Detach for actor update

        # --- Actor Training (REINFORCE) ---
        actor_optimizer.zero_grad()
        policy_loss = []
        # Assign the final heuristic reward to all steps (simple REINFORCE)
        # TODO: Consider discounted rewards or advantage (A2C) for more stable training
        for log_prob in actor_log_probs:
            policy_loss.append(-log_prob * final_reward_heuristic) # Maximize reward

        actor_loss = torch.tensor(0.0, device=config.DEVICE)
        if policy_loss:
            actor_loss = torch.cat(policy_loss).mean() # Use mean instead of sum for stability
            actor_loss.backward() # Calculate gradients for actor
            # nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM) # Optional gradient clipping
            # Defer optimizer step until after accumulation (if used)
            # actor_optimizer.step() # Step done in train loop after accumulation
        else:
            print("Warning: No actions taken in trajectory, skipping actor update.")


        # --- Reward Model Training (Supervised) ---
        # Train the reward model to predict the heuristic reward we calculated
        self.reward_model.train() # Set reward model to train mode
        reward_optimizer.zero_grad()

        # Get the reward model's prediction for the final chain
        reward_prediction = self.reward_model(query, memory_chain)

        # Use MSE loss to train the reward model to predict the heuristic score
        loss_fn_reward = nn.MSELoss()
        reward_loss = loss_fn_reward(reward_prediction.squeeze(), final_reward_heuristic.squeeze())
        # print(f"  Reward Prediction: {reward_prediction.item():.4f}, Target (Heuristic): {final_reward_heuristic.item():.4f}, Loss: {reward_loss.item():.4f}")

        reward_loss.backward() # Calculate gradients for reward model
        # nn.utils.clip_grad_norm_(self.reward_model.parameters(), config.MAX_GRAD_NORM) # Optional gradient clipping
        # Defer optimizer step until after accumulation (if used)
        # reward_optimizer.step() # Step done in train loop after accumulation


        actor_loss_val = actor_loss.item() if torch.is_tensor(actor_loss) else actor_loss
        reward_loss_val = reward_loss.item() if torch.is_tensor(reward_loss) else reward_loss

        # Return losses needed for accumulation outside this function
        return actor_loss, reward_loss


    def save_models(self, epoch: int):
        """Saves checkpoints for Actor and Reward Model."""
        import os
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        actor_path = os.path.join(config.MODEL_SAVE_DIR, f"actor_epoch_{epoch}.pth")
        reward_path = os.path.join(config.MODEL_SAVE_DIR, f"reward_epoch_{epoch}.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.reward_model.state_dict(), reward_path)
        print(f"Saved models at epoch {epoch} to {config.MODEL_SAVE_DIR}")

    def load_models(self, actor_checkpoint_path: str, reward_checkpoint_path: str):
        """Loads Actor and Reward Model checkpoints."""
        try:
            self.actor.load_state_dict(torch.load(actor_checkpoint_path, map_location=config.DEVICE))
            self.reward_model.load_state_dict(torch.load(reward_checkpoint_path, map_location=config.DEVICE))
            print(f"Loaded Actor from {actor_checkpoint_path}")
            print(f"Loaded Reward Model from {reward_checkpoint_path}")
            self.actor.to(config.DEVICE)
            self.reward_model.to(config.DEVICE)
        except FileNotFoundError as e:
            print(f"Error loading models: {e}. Ensure checkpoints exist.")
        except Exception as e:
             print(f"An error occurred during model loading: {e}")