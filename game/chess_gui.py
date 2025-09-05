import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from PIL import Image, ImageTk
import chess
import torch

from utils.model import ChessModel
from utils.board import ChessBoard
from utils.mcts import MCTS
from utils import move_encoder
from config import GlobalConfig # Import GlobalConfig

class ChessGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Chess AI")
        self.canvas = tk.Canvas(self.master, width=400, height=400, bg="light gray")
        self.canvas.pack()

        self.board_size = 8
        self.square_size = 50
        self.board_colors = ["#DDB88C", "#A66D4F"] # Board colors

        self.selected_square = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load trained model (RL first, fallback to supervised)
        self.model = ChessModel().to(self.device)
        model_loaded = False
        rl_model_path = GlobalConfig.FINAL_RL_MODEL_SAVE_PATH
        if os.path.exists(rl_model_path):
            try:
                self.model.load_state_dict(torch.load(rl_model_path, map_location=self.device, weights_only=True), strict=False)
                print(f"RL Model loaded successfully from {rl_model_path} with strict=False.")
                model_loaded = True
            except Exception as e:
                print(f"Error loading RL model from {rl_model_path} with strict=False: {e}")
        
        if not model_loaded:
            supervised_model_path = GlobalConfig.SUPERVISED_MODEL_SAVE_PATH
            if os.path.exists(supervised_model_path):
                try:
                    self.model.load_state_dict(torch.load(supervised_model_path, map_location=self.device, weights_only=True), strict=False)
                    print(f"Supervised Model loaded successfully from {supervised_model_path} with strict=False.")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading supervised model from {supervised_model_path} with strict=False: {e}")
        if not model_loaded:
            print("Error: No suitable model file found. Please ensure a model is trained and saved in the 'models' directory.")
            self.master.destroy()
            return
        self.model.eval()

        # Set up board and MCTS
        self.game_board = ChessBoard()
        self.mcts = MCTS(self.model, self.device, n_simulations=500)

        self.draw_board()   # Draw chessboard squares
        self.draw_pieces()  # Draw chess pieces

        self.canvas.bind("<Button-1>", self.on_square_click)  # Handle mouse clicks

    def draw_board(self):
        # Draw the chessboard squares
        for row in range(self.board_size):
            for col in range(self.board_size):
                color_index = (row + col) % 2
                fill_color = self.board_colors[color_index]
                self.canvas.create_rectangle(
                    col * self.square_size,
                    row * self.square_size,
                    (col + 1) * self.square_size,
                    (row + 1) * self.square_size,
                    fill=fill_color,
                    tags="square"
                )

    def draw_pieces(self):
        # Draw chess pieces as text
        self.canvas.delete("piece")
        for square_index in range(64):
            piece = self.game_board.board.piece_at(square_index)
            if piece:
                row, col = 7 - (square_index // 8), square_index % 8
                piece_char = piece.symbol()
                x = col * self.square_size + self.square_size // 2
                y = row * self.square_size + self.square_size // 2
                self.canvas.create_text(
                    x,
                    y,
                    text=piece_char,
                    font=("Arial", 24, "bold"),
                    fill="black" if piece.color == chess.BLACK else "white", # Adjust text color for visibility
                    tags="piece"
                )

    def on_square_click(self, event):
        col = event.x // self.square_size
        row = event.y // self.square_size
        square_index = (7 - row) * 8 + col # Convert GUI row to chess.Board square index

        if self.game_board.board.turn == chess.WHITE: # Only human (White) can click
            if self.selected_square is None:
                # Select a piece
                piece = self.game_board.board.piece_at(square_index)
                if piece and piece.color == chess.WHITE:
                    self.selected_square = square_index
                    self.highlight_square(row, col, "blue")
            else:
                # Make a move
                from_square = self.selected_square
                to_square = square_index
                
                move = chess.Move(from_square, to_square)
                
                # Check for promotion
                if self.game_board.board.piece_at(from_square).piece_type == chess.PAWN and \
                   (chess.square_rank(to_square) == 7 or chess.square_rank(to_square) == 0):
                    # For simplicity, always promote to Queen for now
                    move.promotion = chess.QUEEN

                if move in self.game_board.board.legal_moves:
                    self.game_board.make_move(move.uci())
                    self.mcts.update_with_move(move_encoder.encode_move(move))
                    self.selected_square = None
                    self.draw_board()
                    self.draw_pieces()
                    self.master.after(100, self.ai_turn) # AI makes a move after human
                else:
                    print("Illegal move. Try again.")
                    self.clear_highlights()
                    self.selected_square = None
        
        self.check_game_status()

    def highlight_square(self, row, col, color):
        self.canvas.create_rectangle(
            col * self.square_size,
            row * self.square_size,
            (col + 1) * self.square_size,
            (row + 1) * self.square_size,
            fill=color,
            tags="highlight"
        )
        self.draw_pieces() # Redraw pieces on top of highlight

    def clear_highlights(self):
        self.canvas.delete("highlight")
        self.draw_board()
        self.draw_pieces()

    def ai_turn(self):
        if self.game_board.board.turn == chess.BLACK:
            print("AI is thinking...")
            move_probs = self.mcts.get_move_probabilities(self.game_board, temperature=0)

            if not move_probs:
                print("AI has no legal moves. Game ends.")
                self.check_game_status()
                return

            # Select the move with the highest probability
            best_encoded_move = max(move_probs, key=move_probs.get)
            ai_move_uci = move_encoder.decode_move(best_encoded_move)
            
            print(f"AI moves: {ai_move_uci}")
            self.game_board.make_move(ai_move_uci)
            self.mcts.update_with_move(best_encoded_move)
            
            self.draw_board()
            self.draw_pieces()
            self.check_game_status()

    def check_game_status(self):
        game_status = self.game_board.get_game_status()
        if game_status != "in_progress":
            if game_status == "checkmate":
                message = "Game Over: Checkmate!"
                if self.game_board.board.turn: # If it's White's turn, Black delivered checkmate
                    message += " AI wins!"
                else:
                    message += " Human wins!"
            elif game_status == "stalemate":
                message = "Game Over: Stalemate!"
            else:
                message = f"Game Over: {game_status.replace('_', ' ').title()}!"
            
            print(message)
            # Display message on GUI (e.g., using a label or message box)
            # For simplicity, we'll just print to console for now.
            # You might want to add a tk.Label or tk.messagebox.showinfo here.

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()
