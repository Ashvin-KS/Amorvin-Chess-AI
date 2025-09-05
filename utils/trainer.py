import torch
import torch.optim as optim
import numpy as np
import os # Added for os.path operations
from .move_encoder import UCI_MOVES
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm # Import tqdm
from torch.optim import lr_scheduler # Import lr_scheduler
from config import GlobalConfig # Import GlobalConfig

class Trainer:
    def __init__(self, model, device, 
                 learning_rate=GlobalConfig.LEARNING_RATE_SUPERVISED, 
                 value_loss_weight=GlobalConfig.VALUE_LOSS_WEIGHT):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Scheduler parameters can be moved to GlobalConfig if needed, for now using fixed values
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5) # Decay LR by 0.5 every 20 epochs
        self.policy_loss_fn = torch.nn.CrossEntropyLoss()
        self.value_loss_fn = torch.nn.MSELoss()
        self.value_loss_weight = value_loss_weight

    def train(self, batch_data):
        self.model.train()

        # Unpack the single batch
        board_states, move_probabilities, outcomes = batch_data

        board_states = torch.FloatTensor(np.array(board_states)).to(self.device)
        outcomes = torch.FloatTensor(outcomes).unsqueeze(1).to(self.device)

        # Create target policy tensor
        target_policies = torch.zeros(len(board_states), len(UCI_MOVES)).to(self.device)
        for i, move_probs in enumerate(move_probabilities):
            for move, prob in move_probs.items():
                target_policies[i, move] = prob

        # Forward pass
        predicted_policies, predicted_values = self.model(board_states)

        # Calculate loss
        policy_loss = self.policy_loss_fn(predicted_policies, torch.argmax(target_policies, dim=1))
        value_loss = self.value_loss_fn(predicted_values, outcomes)
        loss = policy_loss + (self.value_loss_weight * value_loss) # Apply weight to value loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() # Return loss for monitoring

    def supervised_train(self, processed_data_dir, pgn_file_path, epochs=20, batch_size=32):
        self.model.train()

        print(f"Starting supervised pre-training for {epochs} epochs...")

        base_filename = os.path.basename(pgn_file_path).replace(".pgn.zst", "")

        for epoch in tqdm(range(epochs), desc="Supervised Pre-training"):
            chunk_idx = 0
            epoch_total_loss = 0.0
            epoch_batch_count = 0

            while True:
                board_states_chunk_path = os.path.join(processed_data_dir, f"{base_filename}_board_states_chunk_{chunk_idx}.npy")
                move_targets_chunk_path = os.path.join(processed_data_dir, f"{base_filename}_move_targets_chunk_{chunk_idx}.npy")
                outcomes_chunk_path = os.path.join(processed_data_dir, f"{base_filename}_outcomes_chunk_{chunk_idx}.npy")

                if not os.path.exists(board_states_chunk_path):
                    break # No more chunks

                board_states_np = np.load(board_states_chunk_path)
                move_targets_np = np.load(move_targets_chunk_path)
                outcomes_np = np.load(outcomes_chunk_path)

                # Convert to PyTorch tensors (on CPU, then move to device)
                board_states_tensor = torch.FloatTensor(board_states_np)
                move_targets_tensor = torch.FloatTensor(move_targets_np)
                outcomes_tensor = torch.FloatTensor(outcomes_np).unsqueeze(1)

                # Create DataLoader
                dataset = TensorDataset(board_states_tensor, move_targets_tensor, outcomes_tensor)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for batch_idx, (boards, targets, outcomes) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} Chunk {chunk_idx}", leave=False)):
                    boards = boards.to(self.device)
                    targets = targets.to(self.device)
                    outcomes = outcomes.to(self.device)

                    # Forward pass
                    predicted_policies, predicted_values = self.model(boards)

                    # Calculate loss
                    policy_loss = self.policy_loss_fn(predicted_policies, torch.argmax(targets, dim=1))
                    value_loss = self.value_loss_fn(predicted_values, outcomes)
                    loss = policy_loss + (self.value_loss_weight * value_loss) # Apply weight to value loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_total_loss += loss.item()
                    epoch_batch_count += 1
                
                chunk_idx += 1
            
            if epoch_batch_count > 0: # Only step scheduler if some training occurred
                self.scheduler.step() # Step the learning rate scheduler
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_total_loss / epoch_batch_count:.4f}, Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: No data chunks found or processed.")

        print("Supervised pre-training finished.")
