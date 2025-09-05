import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from utils.model import ChessModel
from utils.self_play import SelfPlay
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.replay_buffer import ReplayBuffer
from tqdm import tqdm
from config import GlobalConfig # Import GlobalConfig

def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning for chess AI.")
    # Parse arguments (can override config values)
    parser.add_argument("--initial_model_path", type=str, default=GlobalConfig.INITIAL_RL_MODEL_PATH, help="Path to the initial model.")
    parser.add_argument("--model_save_path", type=str, default=GlobalConfig.FINAL_RL_MODEL_SAVE_PATH, help="Path to save the final model.")
    parser.add_argument("--num_iterations", type=int, default=GlobalConfig.RL_NUM_ITERATIONS, help="Total RL iterations.")
    parser.add_argument("--num_games_per_iteration", type=int, default=GlobalConfig.RL_NUM_GAMES_PER_ITERATION, help="Self-play games per iteration.")
    parser.add_argument("--replay_buffer_size", type=int, default=GlobalConfig.RL_REPLAY_BUFFER_SIZE, help="Max samples to store in replay buffer.")
    parser.add_argument("--rl_batch_size", type=int, default=GlobalConfig.RL_BATCH_SIZE, help="Batch size for RL training.")
    parser.add_argument("--num_training_epochs_per_iteration", type=int, default=GlobalConfig.RL_NUM_TRAINING_EPOCHS_PER_ITERATION, help="How many epochs to train on collected data.")
    parser.add_argument("--mcts_num_simulations_self_play", type=int, default=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_SELF_PLAY, help="MCTS simulations for self-play.")
    parser.add_argument("--mcts_num_simulations_eval", type=int, default=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_EVAL, help="MCTS simulations for evaluation.")
    parser.add_argument("--num_evaluation_games", type=int, default=GlobalConfig.RL_NUM_EVALUATION_GAMES, help="Number of evaluation games.")
    parser.add_argument("--challenger_win_rate_threshold", type=float, default=GlobalConfig.RL_CHALLENGER_WIN_RATE_THRESHOLD, help="Challenger win rate threshold.")
    parser.add_argument("--max_moves_per_game", type=int, default=GlobalConfig.RL_MAX_MOVES_PER_GAME, help="Max moves in a self-play game.")
    parser.add_argument("--model_save_interval", type=int, default=GlobalConfig.RL_MODEL_SAVE_INTERVAL, help="Save best model every X iterations.")
    parser.add_argument("--learning_rate", type=float, default=GlobalConfig.LEARNING_RATE_RL, help="Learning rate.")
    args = parser.parse_args()

    device = torch.device(GlobalConfig.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    current_model = ChessModel().to(device)
    try:
        current_model.load_state_dict(torch.load(args.initial_model_path, map_location=device), strict=False)
        print(f"Successfully loaded initial model from {args.initial_model_path} with strict=False.")
    except Exception as e:
        print(f"Error loading initial model from {args.initial_model_path}: {e}")
        print("Initializing model from scratch.")
    
    best_model = ChessModel().to(device)
    best_model.load_state_dict(current_model.state_dict())

    # Set up training components
    trainer = Trainer(current_model, device, learning_rate=args.learning_rate)
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    self_play = SelfPlay(current_model, device, mcts_num_simulations=args.mcts_num_simulations_self_play)
    evaluator = Evaluator(current_model, best_model, device, n_simulations=args.mcts_num_simulations_eval, num_games=args.num_evaluation_games)

    print("\n--- Starting Reinforcement Learning (Self-Play) ---")
    highest_win_rate = 0.0

    # Main RL training loop
    for i in tqdm(range(args.num_iterations), desc="RL Iterations"):
        # Decay learning rate at specified iteration
        if (i + 1) == GlobalConfig.RL_LR_DECAY_ITERATION:
            new_lr = GlobalConfig.RL_LR_DECAY_VALUE
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate decayed to {new_lr} after {i+1} iterations.")

        # Generate self-play games and add to replay buffer
        for _ in tqdm(range(args.num_games_per_iteration), desc=f"Iteration {i+1} Self-Play Games", leave=False):
            game_history = self_play.play_game()
            if game_history:
                replay_buffer.add_experience(game_history)
        # Train model if enough samples in buffer
        if len(replay_buffer) >= args.rl_batch_size:
            for epoch in tqdm(range(args.num_training_epochs_per_iteration), desc=f"Iteration {i+1} RL Training Epochs", leave=False):
                batch_board_states, batch_move_probs, batch_outcomes = replay_buffer.sample_batch(args.rl_batch_size)
                trainer.train((batch_board_states, batch_move_probs, batch_outcomes))
        # Evaluate and save model at intervals
        if (i + 1) % args.model_save_interval == 0:
            current_model_win_rate = evaluator.evaluate()
            if current_model_win_rate > highest_win_rate:
                highest_win_rate = current_model_win_rate
                print(f"New best model found with win rate: {highest_win_rate:.2f}. Saving as {GlobalConfig.BEST_RL_MODEL_SAVE_PATH}")
                best_model.load_state_dict(current_model.state_dict())
                torch.save(best_model.state_dict(), GlobalConfig.BEST_RL_MODEL_SAVE_PATH)
            if current_model_win_rate > args.challenger_win_rate_threshold:
                print(f"Current model (win rate: {current_model_win_rate:.2f}) is good enough to become the new challenger. Continuing training with current model.")
            else:
                print(f"Current model (win rate: {current_model_win_rate:.2f}) is not significantly better than the threshold. Keeping current model for further training, but not updating best model.")
    # Save final model
    torch.save(current_model.state_dict(), args.model_save_path)
    print(f"Final model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()
