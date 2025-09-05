import torch
import chess
from .board import ChessBoard
from .mcts import MCTS
from . import move_encoder
import numpy as np
from config import GlobalConfig # Import GlobalConfig

class Evaluator:
    def __init__(self, current_model, best_model, device, 
                 n_simulations=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_EVAL, 
                 num_games=GlobalConfig.RL_NUM_EVALUATION_GAMES):
        self.current_model = current_model
        self.best_model = best_model
        self.device = device
        self.n_simulations = n_simulations
        self.num_games = num_games

    def evaluate(self):
        """
        Play games between current and best model, return win rate for current model.
        """
        current_wins = 0
        best_wins = 0
        draws = 0

        print(f"\n--- Starting evaluation of Current Model vs Best Model ({self.num_games} games) ---")

        for i in range(self.num_games):
            board = ChessBoard()  # Start a new game
            # Alternate which model plays white
            if i % 2 == 0:
                white_mcts = MCTS(self.current_model, self.device, n_simulations=self.n_simulations)
                black_mcts = MCTS(self.best_model, self.device, n_simulations=self.n_simulations)
                current_is_white = True
            else:
                white_mcts = MCTS(self.best_model, self.device, n_simulations=self.n_simulations)
                black_mcts = MCTS(self.current_model, self.device, n_simulations=self.n_simulations)
                current_is_white = False

            while not board.board.is_game_over():
                # Pick which MCTS to use based on turn
                mcts_player = white_mcts if board.board.turn == chess.WHITE else black_mcts
                with torch.no_grad():
                    move_probs = mcts_player.get_move_probabilities(board)
                if not move_probs:
                    break
                # Play move with highest probability
                best_encoded_move = max(move_probs, key=move_probs.get)
                move_uci = move_encoder.decode_move(best_encoded_move)
                board.make_move(move_uci)
                white_mcts.update_with_move(best_encoded_move)
                black_mcts.update_with_move(best_encoded_move)
            # Get winner
            outcome = board.board.outcome()
            if outcome is None:
                winner = "Unknown"
            elif outcome.winner == chess.WHITE:
                winner = "White"
            elif outcome.winner == chess.BLACK:
                winner = "Black"
            else:
                winner = "Draw"
            # Track win counts
            if winner == "White":
                if current_is_white:
                    current_wins += 1
                else:
                    best_wins += 1
            elif winner == "Black":
                if current_is_white:
                    best_wins += 1
                else:
                    current_wins += 1
            else:
                draws += 1
            print(f"    Game {i+1} finished. Winner: {winner}")
        # Print summary
        total_games = self.num_games
        current_win_rate = current_wins / total_games
        best_win_rate = best_wins / total_games
        draw_rate = draws / total_games
        print(f"  Evaluation Results:")
        print(f"    Current Model Wins: {current_wins} ({current_win_rate:.2f})")
        print(f"    Best Model Wins:    {best_wins} ({best_win_rate:.2f})")
        print(f"    Draws:              {draws} ({draw_rate:.2f})")
        print("--- Evaluation finished ---")

        return current_win_rate

if __name__ == "__main__":
    # Example usage (requires trained models)
    device = torch.device(GlobalConfig.DEVICE if torch.cuda.is_available() else "cpu")
    
    # For demonstration, let's create dummy models
    from .model import ChessModel
    current_model = ChessModel().to(device)
    best_model = ChessModel().to(device)

    # Load actual weights if available
    try:
        current_model.load_state_dict(torch.load(GlobalConfig.FINAL_RL_MODEL_SAVE_PATH, map_location=device, weights_only=True))
        best_model.load_state_dict(torch.load(GlobalConfig.BEST_RL_MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("Loaded models for evaluation example.")
    except FileNotFoundError:
        print("Model files not found. Using untrained models for example. Please ensure models are trained and saved in the 'models/rl' directory.")

    evaluator = Evaluator(current_model, best_model, device, 
                          n_simulations=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_EVAL, 
                          num_games=GlobalConfig.RL_NUM_EVALUATION_GAMES)
    win_rate = evaluator.evaluate()
    print(f"Current Model Win Rate against Best Model: {win_rate:.2f}")
