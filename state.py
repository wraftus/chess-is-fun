#!/usr/bin/env python3
import chess
import numpy as np

piece_to_state = {'P': 1, 'R': 2, 'N': 4, 'B': 5, 'Q': 6, 'K': 7, \
                  'p': 9, 'r': 10, 'n': 12, 'b': 13, 'q': 14, 'k': 15 }

# holds current state of board, and serializes it according to README
class State(object):
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    # serialize current board state
    def serialize(self):
        # make sure the board is valid
        assert self.board.is_valid()

        # construct bit state
        bstate = np.zeros(64, np.uint8)
        for square in range(64):
            piece = self.board.piece_at(square)
            if piece != None:
                bstate[square] = piece_to_state[piece.symbol()]

        # check for castling
        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert bstate[0] == piece_to_state['R']
            bstate[0] += 1
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert bstate[7] == piece_to_state['R']
            bstate[7] += 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert bstate[56] == piece_to_state['r']
            bstate[56] += 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert bstate[63] == piece_to_state['r']
            bstate[63] += 1

        # check for en passant
        if self.board.ep_square != None:
            assert bstate[self.board.ep_square] == 0
            bstate[self.board.ep_square] = 8

        # reshape to 8x8 grid
        bstate = bstate.reshape(8,8)

        # turn into 8x8x5 binary state map
        state = np.zeros((8,8,5), np.uint8)
        state[:,:,0] = (bstate >> 3) & 1
        state[:,:,1] = (bstate >> 2) & 1
        state[:,:,2] = (bstate >> 1) & 1
        state[:,:,3] = (bstate >> 0) & 1
        state[:,:,4] = self.board.turn * 1.0

        return state
