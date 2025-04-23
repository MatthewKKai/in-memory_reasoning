import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm # Progress bars
import os
import math

import config
from system import MemoryAugmentedLLM
from utils import load_longmemeval_dataset, preprocess_longmemeval, seed_everything
# from evaluation import evaluate_system # TODO: Implement evaluation function


def train():
    print("--- Starting Training Process ---")
    seed_everything(config.SEED)

    # --- Initialize System ---
    # Handles loading embedding model, Gemma, Actor, Reward Model etc.
    system = MemoryAugmentedLLM(n_iterations=config.N_ITERATIONS)

    # --- Load and Prepare Dataset ---
    print("Loading and preprocessing dataset...")
    # Load full dataset or specific split
    dataset = load_longmemeval_dataset() # Returns DatasetDict or Dataset

    # Assuming 'train' split exists, adjust if needed
    if isinstance(dataset, dict) and 'train' in dataset:
        train_dataset = dataset['train']
    elif isinstance(dataset, Dataset):
         train_dataset = dataset # Use the whole dataset if not dict
    else:
         raise ValueError("Could not find 'train' split or suitable dataset format.")

    # Select a smaller subset for faster debugging/iteration if needed
    # train_dataset = Subset(train_dataset, range(100)) # Uncomment for quick test
    # print(f"Using subset of {len(train_dataset)} examples for training.")


    # Apply preprocessing
    processed_train_dataset = train_dataset.map(
        preprocess_longmemeval,
        remove_columns=train_dataset.column_names # Keep only query, ground_truth_answer
    )
    print(f"Dataset preprocessed. Training examples: {len(processed_train_dataset)}")

    # --- DataLoader ---
    # Collate function isn't strictly needed if just passing strings
    train_dataloader = DataLoader(
        processed_train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # collate_fn=collate_batch # Define if custom batching needed
    )

    # --- Optimizers and Schedulers ---
    actor_optimizer = optim.AdamW(system.actor.parameters(), lr=config.ACTOR_LR)
    reward_optimizer = optim.AdamW(system.reward_model.parameters(), lr=config.REWARD_LR)

    num_training_steps = math.ceil(len(train_dataloader) / config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup

    actor_scheduler = get_linear_schedule_with_warmup(actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    reward_scheduler = get_linear_schedule_with_warmup(reward_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # --- Training Loop ---
    print(f"Starting training for {config.EPOCHS} epochs on {config.DEVICE}...")
    global_step = 0
    total_actor_loss = 0.0
    total_reward_loss = 0.0
    log_interval = 20 # Log every N steps

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        system.actor.train()
        system.reward_model.train() # Ensure reward model is trainable

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)

        for step, batch in enumerate(progress_bar):
            # Ensure batch items are strings (DataLoader might wrap them)
            queries = batch['query']
            ground_truth_answers = batch['ground_truth_answer']

            # Accumulate loss over micro-batches
            accumulated_actor_loss = 0.0
            accumulated_reward_loss = 0.0

            # Loop through items in the batch (as train_step processes one item)
            for i in range(len(queries)):
                query = queries[i]
                answer = ground_truth_answers[i]

                # Check for empty strings which can cause issues
                if not query or not answer:
                    print(f"Warning: Skipping empty query or answer at step {global_step}, index {i}")
                    continue

                # Perform one training step for a single query-answer pair
                # This returns the loss tensors before backward() is called on them
                actor_loss_tensor, reward_loss_tensor = system.train_step(
                    query,
                    answer,
                    actor_optimizer,
                    reward_optimizer
                )

                # Scale loss for gradient accumulation
                actor_loss_scaled = actor_loss_tensor / config.GRADIENT_ACCUMULATION_STEPS
                reward_loss_scaled = reward_loss_tensor / config.GRADIENT_ACCUMULATION_STEPS

                # Accumulate gradients (backward is called inside train_step now)
                # actor_loss_scaled.backward() # Should be called inside train_step for REINFORCE
                # reward_loss_scaled.backward() # Should be called inside train_step

                accumulated_actor_loss += actor_loss_scaled.item() * config.GRADIENT_ACCUMULATION_STEPS # Log unscaled loss
                accumulated_reward_loss += reward_loss_scaled.item() * config.GRADIENT_ACCUMULATION_STEPS # Log unscaled loss


            # --- Optimizer Step (after accumulation) ---
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                 # Clip gradients before optimizer step
                 nn.utils.clip_grad_norm_(system.actor.parameters(), config.MAX_GRAD_NORM)
                 nn.utils.clip_grad_norm_(system.reward_model.parameters(), config.MAX_GRAD_NORM)

                 actor_optimizer.step()
                 reward_optimizer.step()

                 actor_scheduler.step()
                 reward_scheduler.step()

                 # Reset gradients for the next accumulation cycle
                 actor_optimizer.zero_grad()
                 reward_optimizer.zero_grad()

                 global_step += 1

                 # Logging
                 total_actor_loss += accumulated_actor_loss / len(queries) # Average loss for the batch
                 total_reward_loss += accumulated_reward_loss / len(queries)

                 if global_step % log_interval == 0:
                     avg_actor_loss = total_actor_loss / log_interval
                     avg_reward_loss = total_reward_loss / log_interval
                     progress_bar.set_postfix({
                         "Actor Loss": f"{avg_actor_loss:.4f}",
                         "Reward Loss": f"{avg_reward_loss:.4f}",
                         "LR": f"{actor_scheduler.get_last_lr()[0]:.2e}"
                     })
                     total_actor_loss = 0.0
                     total_reward_loss = 0.0

                 # --- Evaluation (Optional) ---
                 # if global_step % config.EVAL_STEPS == 0:
                 #     print(f"\n--- Evaluating at Step {global_step} ---")
                 #     # evaluation_results = evaluate_system(system, eval_dataset) # Implement evaluation
                 #     # print(f"Evaluation Results: {evaluation_results}")
                 #     system.actor.train() # Switch back to train mode after eval
                 #     system.reward_model.train()


        # --- Save Model Checkpoint at End of Epoch ---
        system.save_models(epoch + 1)

    print("--- Training Finished ---")

if __name__ == "__main__":
    # Ensure NLTK data is available for potential evaluation metrics
    from utils import ensure_nltk_data
    ensure_nltk_data()
    train()