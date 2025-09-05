from .board import ChessBoard
from .mcts import MCTS
import numpy as np
from . import move_encoder
import chess # Import chess for game status checks
from config import GlobalConfig # Import GlobalConfig

class SelfPlay:
    def __init__(self, model, device, mcts_num_simulations=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_SELF_PLAY):
        self.model = model
        self.device = device
        self.board = ChessBoard()
        self.mcts = MCTS(model, device, n_simulations=mcts_num_simulations)

    def play_game(self):
        game_history = []
        move_count = 0
        while True:
            # Add Dirichlet noise to the root node's policy for the first move
            add_noise = (move_count == 0)

            # Use temperature for the first ~20 moves
            temperature = 1.0 if move_count < 20 else 0.1 # Decay temperature after 20 moves

            move_probs = self.mcts.get_move_probabilities(self.board, add_noise=add_noise, temperature=temperature)
            
            if not move_probs:
                break # No legal moves, game ends

            legal_moves = list(move_probs.keys())
            probs = np.array(list(move_probs.values()))
            
            # Normalize probabilities to ensure they sum to 1
            if probs.sum() == 0:
                probs = np.ones_like(probs) / len(probs) # Assign uniform probability if all are zero
            else:
                probs /= probs.sum() 
            
            game_history.append([self.board.get_board_state(), move_probs, None])

            # Choose a move
            encoded_move = np.random.choice(legal_moves, p=probs)
            move_uci = move_encoder.decode_move(encoded_move)

            self.board.make_move(move_uci)
            self.mcts.update_with_move(encoded_move)
            move_count += 1

            if self.board.board.is_game_over():
                # Game is over, assign the outcome to all states in the history
                outcome = self.board.board.outcome()
                if outcome.winner == chess.WHITE:
                    final_outcome_white_perspective = 1.0
                elif outcome.winner == chess.BLACK:
                    final_outcome_white_perspective = -1.0
                else:
                    final_outcome_white_perspective = 0.0
                
                for state_idx in range(len(game_history)):
                    is_white_turn_at_state = (state_idx % 2 == 0)
                    
                    if is_white_turn_at_state:
                        game_history[state_idx][2] = final_outcome_white_perspective
                    else:
                        game_history[state_idx][2] = -final_outcome_white_perspective
                
                return game_history
