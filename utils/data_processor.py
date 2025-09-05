import chess.pgn
import zstandard
import os
import torch
import numpy as np
from .board import ChessBoard # Assuming ChessBoard has get_board_state
from .move_encoder import encode_move, UCI_MOVES # Assuming move_encoder has encode_move and UCI_MOVES
import io # Import io
from config import GlobalConfig # Import GlobalConfig

def process_pgn_file(pgn_filepath=GlobalConfig.PGN_FILE_PATH, output_dir=GlobalConfig.PROCESSED_DATA_DIR, max_games=GlobalConfig.MAX_GAMES_PROCESS):
    """
    Decompress PGN file, parse games, and convert to training data for neural network.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    board_states_list = []   # Stores board state tensors
    move_targets_list = []   # Stores move target vectors
    outcomes_list = []       # Stores game outcomes
    chunk_idx = 0

    print(f"Processing {pgn_filepath}...")
    game_count = 0
    base_filename = os.path.basename(pgn_filepath).replace(".pgn.zst", "")

    try:
        with open(pgn_filepath, 'rb') as compressed_file:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as decompressed_stream:
                text_stream = io.TextIOWrapper(decompressed_stream, encoding='utf-8', errors='ignore')
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    game_count += 1
                    if max_games and game_count > max_games:
                        break

                    board = ChessBoard()
                    # Get game outcome
                    result = game.headers.get("Result", "*")
                    if result == "1-0":
                        outcome = 1.0
                    elif result == "0-1":
                        outcome = -1.0
                    else:
                        outcome = 0.0

                    # Collect board states and move targets for each move
                    for i, move in enumerate(game.mainline_moves()):
                        board_states_list.append(board.get_board_state())
                        target_policy = np.zeros(len(UCI_MOVES), dtype=np.float32)
                        encoded_move = encode_move(move)
                        if encoded_move is not None:
                            target_policy[encoded_move] = 1.0
                        move_targets_list.append(target_policy)
                        outcomes_list.append(outcome if board.board.turn == chess.WHITE else -outcome)
                        board.make_move(move.uci())
                    
                    # Save chunk to disk if needed
                    if len(board_states_list) >= 50000: # Save every 50,000 game states
                        save_chunk(board_states_list, move_targets_list, outcomes_list, output_dir, base_filename, chunk_idx)
                        board_states_list = []
                        move_targets_list = []
                        outcomes_list = []
                        chunk_idx += 1


    except Exception as e:
        print(f"Error processing {pgn_filepath}: {e}")
        return None

    # Save any remaining data
    if board_states_list:
        save_chunk(board_states_list, move_targets_list, outcomes_list, output_dir, base_filename, chunk_idx, final_chunk=True)

    print(f"Finished processing {game_count} games from {pgn_filepath}.")
    print(f"Saved processed data to {output_dir}/")
    return output_dir # Return the directory where chunks are saved

def save_chunk(board_states_list, move_targets_list, outcomes_list, output_dir, base_filename, chunk_idx, final_chunk=False):
    if final_chunk and not board_states_list: # Don't save empty final chunk
        return
    
    board_states_np = np.array(board_states_list, dtype=np.float32)
    move_targets_np = np.array(move_targets_list, dtype=np.float32)
    outcomes_np = np.array(outcomes_list, dtype=np.float32)

    np.save(os.path.join(output_dir, f"{base_filename}_board_states_chunk_{chunk_idx}.npy"), board_states_np)
    np.save(os.path.join(output_dir, f"{base_filename}_move_targets_chunk_{chunk_idx}.npy"), move_targets_np)
    np.save(os.path.join(output_dir, f"{base_filename}_outcomes_chunk_{chunk_idx}.npy"), outcomes_np)
    print(f"Saved chunk {chunk_idx} with {len(board_states_list)} states.")


if __name__ == "__main__":
    # Example usage:
    # You need to download a .pgn.zst file first, e.g., from Lichess database
    # For example, place 'lichess_db_standard_rated_2013-01.pgn.zst' in c:/nonclgstuffs
    
    # IMPORTANT: Replace this with the actual path to your downloaded .pgn.zst file
    pgn_file = GlobalConfig.PGN_FILE_PATH
    
    # Check if the file exists
    if not os.path.exists(pgn_file):
        print(f"Error: PGN file '{pgn_file}' not found.")
        print("Please ensure the PGN file is in the 'database' folder.")
    else:
        # Process a small number of games for testing
        processed_dir = process_pgn_file(pgn_file, output_dir=GlobalConfig.PROCESSED_DATA_DIR, max_games=GlobalConfig.MAX_GAMES_PROCESS) # Process more games to test chunking
        if processed_dir:
            print(f"Successfully processed data. Output directory: {processed_dir}")
            # To load:
            # all_board_states = []
            # for f in os.listdir(processed_dir):
            #     if "board_states_chunk" in f:
            #         all_board_states.append(np.load(os.path.join(processed_dir, f)))
            # final_board_states = np.concatenate(all_board_states)
