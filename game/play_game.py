import torch
import chess
from model import ChessModel
from board import ChessBoard
from mcts import MCTS
import move_encoder


def play_game_vs_ai(model_path="../../model_20.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = ChessModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the model is trained and saved.")
        return
    model.eval() # Set model to evaluation mode

    board = ChessBoard()
    mcts = MCTS(model, device, n_simulations=400)

    print("Starting AI vs Human Chess Game!")
    print("You are playing as White. Enter moves in UCI format (e.g., 'e2e4').")

    while True:
        # Print the current board
        print("\n" + "="*30)
        print(board.board)
        print("="*30 + "\n")

        game_status = board.get_game_status()
        if game_status != "in_progress":
            # Handle game over conditions
            if game_status == "checkmate":
                print("Game Over: Checkmate!")
                if board.board.turn:
                    print("AI wins!")
                else:
                    print("Human wins!")
            elif game_status == "stalemate":
                print("Game Over: Stalemate!")
            else:
                print(f"Game Over: {game_status.replace('_', ' ').title()}!")
            break

        if board.board.turn == chess.WHITE:
            # Human's turn: prompt for move
            while True:
                try:
                    human_move_uci = input("Your move (UCI): ")
                    move = chess.Move.from_uci(human_move_uci)
                    if move in board.board.legal_moves:
                        board.make_move(human_move_uci)
                        mcts.update_with_move(move_encoder.encode_move(move))
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid UCI format. Try again.")
        else:
            # AI's turn: select and play move
            print("AI is thinking...")
            move_probs = mcts.get_move_probabilities(board)
            if not move_probs:
                print("AI has no legal moves. Game ends.")
                break
            best_encoded_move = max(move_probs, key=move_probs.get)
            ai_move_uci = move_encoder.decode_move(best_encoded_move)
            print(f"AI moves: {ai_move_uci}")
            board.make_move(ai_move_uci)
            mcts.update_with_move(best_encoded_move)

if __name__ == "__main__":
    play_game_vs_ai()
