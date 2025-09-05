import chess
import numpy as np
from . import move_encoder

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()

    def make_move(self, move):
        self.board.push_uci(move)

    def get_game_status(self):
        if self.board.is_checkmate():
            return "checkmate"
        elif self.board.is_stalemate():
            return "stalemate"
        elif self.board.is_insufficient_material():
            return "insufficient_material"
        elif self.board.is_seventyfive_moves():
            return "seventyfive_moves"
        elif self.board.is_fivefold_repetition():
            return "fivefold_repetition"
        else:
            return "in_progress"

    def get_board_state(self):
        # Build a tensor representing the board state for neural network input
        board_state = np.zeros((18, 8, 8), dtype=np.uint8)

        # Encode piece positions
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_type_offset = piece.piece_type - 1
                board_state[color_offset + piece_type_offset, i // 8, i % 8] = 1

        # Encode side to move
        if self.board.turn == chess.WHITE:
            board_state[12, :, :] = 1

        # Encode castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            board_state[13, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            board_state[14, :, :] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            board_state[15, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            board_state[16, :, :] = 1

        # Encode fifty-move rule counter
        board_state[17, :, :] = self.board.halfmove_clock

        return board_state

    def get_legal_moves(self):
        return self.board.legal_moves

    def get_legal_moves_encoded(self):
        encoded_moves = []
        for move in self.board.legal_moves:
            encoded = move_encoder.encode_move(move)
            if encoded is not None:
                encoded_moves.append(encoded)
        return encoded_moves

    def copy(self):
        new_board = ChessBoard()
        new_board.board = self.board.copy()
        return new_board
