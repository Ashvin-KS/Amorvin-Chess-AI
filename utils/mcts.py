import numpy as np
import math
import torch
from . import move_encoder
import chess # Import chess for piece colors
import numpy.random as nr # Import for Dirichlet noise
from config import GlobalConfig # Import GlobalConfig

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.N = 0  # visit count
        self.W = 0  # total action value
        self.Q = 0  # mean action value
        self.P = prior_p  # prior probability

    def expand(self, action_priors):
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_p=prob)

    def select(self, c_puct):
        # Ensure children is not empty before calling max
        if not self.children:
            return None, None # Or raise an error, depending on desired behavior

        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        # Avoid division by zero if parent.N is 0 (shouldn't happen if root is properly initialized)
        if self.parent and self.parent.N == 0:
            return self.Q + c_puct * self.P # Simplified UCB1 if parent not visited
        
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

    def update(self, leaf_value):
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, model, device, 
                 n_simulations=GlobalConfig.RL_MCTS_NUM_SIMULATIONS_SELF_PLAY, 
                 c_puct=1.0): # c_puct is often kept as a hyperparameter not in global config
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root = Node()

    def _playout(self, state):
        node = self.root
        moves_made = []
        # Traverse the tree
        while True:
            if node.is_leaf():
                break
            
            encoded_move, node = node.select(self.c_puct)
            if encoded_move is None: # Handle case where node has no children (shouldn't happen if expand is correct)
                break
            
            move_uci = move_encoder.decode_move(encoded_move)
            state.make_move(move_uci)
            moves_made.append(move_uci)

        # Determine the value of the leaf node
        game_status = state.get_game_status()
        if game_status != "in_progress":
            if game_status == "checkmate":
                # The current player is checkmated, so the value is -1 from their perspective
                value = -1.0
            elif game_status == "draw":
                # Penalize draws slightly
                value = -0.1 # Instead of 0.0
            else:
                # Other draw conditions (stalemate, insufficient material, etc.)
                value = 0.0
        else:
            # Use the neural network to predict the value and policy
            board_state = torch.FloatTensor(state.get_board_state()).unsqueeze(0).to(self.device)
            with torch.no_grad(): # Ensure no gradients are computed during inference
                policy_logits, value_tensor = self.model(board_state)
            
            value = value_tensor.item()
            policy = torch.exp(policy_logits).squeeze(0).cpu().numpy() # Convert logits to probabilities

            # Expand the leaf node
            legal_moves = state.get_legal_moves_encoded()
            if legal_moves:
                # Filter policy to only legal moves and normalize
                legal_policy_values = {m: policy[m] for m in legal_moves}
                
                # Ensure probabilities sum to 1 for legal moves
                sum_legal_policy = sum(legal_policy_values.values())
                if sum_legal_policy > 0:
                    legal_policy = {m: p / sum_legal_policy for m, p in legal_policy_values.items()}
                else:
                    # If all legal moves have zero policy probability, assign uniform
                    legal_policy = {m: 1.0 / len(legal_moves) for m in legal_moves}
                
                node.expand(legal_policy)

        # Backpropagate the value up the tree
        while node is not None:
            node.update(value)
            # The value is from the perspective of the player at the node.
            # The parent node is the other player, so we negate the value.
            value = -value
            node = node.parent
            
        # Undo the moves made during the playout
        for _ in range(len(moves_made)):
            state.board.pop()

    def get_move_probabilities(self, state, add_noise=False, temperature=1.0):
        # Reset root for a new search
        self.root = Node() 
        
        # If the root is not expanded, expand it with model's policy
        if self.root.is_leaf():
            board_state = torch.FloatTensor(state.get_board_state()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, _ = self.model(board_state)
            policy = torch.exp(policy_logits).squeeze(0).cpu().numpy()
            
            legal_moves = state.get_legal_moves_encoded()
            if legal_moves:
                legal_policy_values = {m: policy[m] for m in legal_moves}
                sum_legal_policy = sum(legal_policy_values.values())
                if sum_legal_policy > 0:
                    legal_policy = {m: p / sum_legal_policy for m, p in legal_policy_values.items()}
                else:
                    legal_policy = {m: 1.0 / len(legal_moves) for m in legal_moves}
                
                # Add Dirichlet noise to the root node's policy
                if add_noise:
                    noise = nr.dirichlet([0.3] * len(legal_moves)) # Alpha value 0.3 as in AlphaZero
                    for i, move in enumerate(legal_moves):
                        legal_policy[move] = legal_policy[move] * 0.75 + noise[i] * 0.25 # Mix with 25% noise
                    
                    # Re-normalize after adding noise
                    sum_noisy_policy = sum(legal_policy.values())
                    legal_policy = {m: p / sum_noisy_policy for m, p in legal_policy.items()}

                self.root.expand(legal_policy)
            else:
                return {} # No legal moves from initial state

        for _ in range(self.n_simulations):
            self._playout(state)

        # If root has no children after simulations (e.g., game ended immediately)
        if not self.root.children:
            return {}

        # Apply temperature to move probabilities
        if temperature < 1e-6:
            best_move = max(self.root.children.items(), key=lambda item: item[1].N)[0]
            move_probs = {move: 1.0 if move == best_move else 0.0 for move in self.root.children.keys()}
        elif temperature == 1.0:
            move_probs = {move: node.N / self.root.N for move, node in self.root.children.items()}
        else:
            # Apply temperature: raise visit counts to power of 1/temperature
            # Then normalize
            sum_visits_temp = sum([node.N**(1/temperature) for node in self.root.children.values()])
            move_probs = {move: (node.N**(1/temperature)) / sum_visits_temp for move, node in self.root.children.items()}

        return move_probs

    def update_with_move(self, last_move_encoded):
        # If the last_move_encoded is in the current root's children,
        # make that child the new root. Otherwise, start a new tree.
        if last_move_encoded in self.root.children:
            self.root = self.root.children[last_move_encoded]
            self.root.parent = None # Detach from old tree
        else:
            self.root = Node() # Start a new tree if move not found
