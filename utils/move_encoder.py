import chess # Ensure this is at the top of your file


# --- Comprehensive UCI Move Generation ---
def generate_all_uci_moves():
    """
    Generates a comprehensive list of all theoretically possible UCI move strings,
    including promotions, for use as a neural network's output space.
    This list typically contains 4416 unique strings.
    """
    files = [chr(ord('a') + i) for i in range(8)]
    ranks = [str(i) for i in range(1, 9)]
    promotions = ['q', 'r', 'b', 'n']

    all_uci_moves_generated = []

    for from_file in files:
        for from_rank in ranks:
            from_square = from_file + from_rank
            
            for to_file in files:
                for to_rank in ranks:
                    to_square = to_file + to_rank
                    
                    if from_square == to_square:
                        continue
                    
                    # Determine if this from_square-to_square path could be a pawn promotion
                    # This is purely based on ranks, not actual piece type, to cover all string possibilities
                    is_white_pawn_promotion_path = (from_rank == '7' and to_rank == '8')
                    is_black_pawn_promotion_path = (from_rank == '2' and to_rank == '1')
                    
                    is_promotion_move = is_white_pawn_promotion_path or is_black_pawn_promotion_path
                    
                    if is_promotion_move:
                        # For potential promotion moves, add all 4 promotion types
                        for p_piece in promotions:
                            all_uci_moves_generated.append(from_square + to_square + p_piece)
                    else:
                        # For non-promotion moves, just the from_square to to_square string
                        all_uci_moves_generated.append(from_square + to_square)

    all_uci_moves_generated.sort() # Sort for consistent indexing
    
    return all_uci_moves_generated

# --- Initialize the Global Move List and Mappings ---
UCI_MOVES = generate_all_uci_moves()
MOVE_TO_INT = {move_uci: i for i, move_uci in enumerate(UCI_MOVES)}
INT_TO_MOVE = {i: move_uci for i, move_uci in enumerate(UCI_MOVES)}

# You can print this to confirm the number of moves:
# print(f"Initialized {len(UCI_MOVES)} unique UCI moves for encoding.")


# --- Encoding and Decoding Functions ---
def encode_move(move_obj):
    """
    Encodes a python-chess Move object into an integer index.
    Returns None if the move's UCI string is not in the predefined list (should not happen for legal moves).
    """
    return MOVE_TO_INT.get(move_obj.uci())

def decode_move(encoded_int):
    """
    Decodes an integer index back into its UCI string representation.
    Returns None if the integer is out of bounds.
    """
    return INT_TO_MOVE.get(encoded_int)