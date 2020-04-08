#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import chess
from board import BoardWrapper
from state import State

class HumanPlayer(object):
    def __init__(self, colour):
        self.colour = colour

    def get_next_move(self):
        while(True):
            raw_move = input("Please enter your next move " + ("White: " if self.colour else "Black: "))
            raw_move = raw_move.split(' ')
            if len(raw_move) != 2:
                print("Moves should be in the form 'A3 A4'")
                continue

            from_col = ord(raw_move[0][0]) - ord('A')
            from_row = ord(raw_move[0][1]) - ord('1')
            to_col = ord(raw_move[1][0]) - ord('A')
            to_row = ord(raw_move[1][1]) - ord('1')

            if (not 0 <= from_col < 8) or (not 0 <= from_row < 8):
                print("First coordinate is not valid")
                continue
            if (not 0 <= to_col < 8) or (not 0 <= to_row < 8):
                print("Second coordinate is not valid")
                continue

            from_coord = from_col + 8 * from_row
            to_coord = to_col + 8 * to_row
            return chess.Move(from_coord, to_coord)

class ComputerPlayer(object):
        def __init__(self, colour, board):
            self.colour = colour
            self.board = board
            self.model = tf.keras.models.load_model('state_model')

        def get_next_move(self):
            return self.minimax(1, True)[1]

        def minimax(self, depth, is_max):
            if self.board.is_game_over():
                return { '1-0':1, '0-1':-1, '1/2-1/2':0 }[self.board.result()]
            if depth <= 0:
                state = np.expand_dims(State(self.board).serialize(), 0)
                return self.model.predict(state)

            val = -2 if is_max else 2
            next_move = None
            for move in self.board.legal_moves:
                self.board.push(move)
                if is_max:
                    new_val = self.minimax(depth -1, not is_max)[0]
                    if new_val > val:
                        val = new_val
                        next_move = move
                else:
                    new_val = self.minimax(depth -1, not is_max)[0]
                    if new_val < val:
                        val = new_val
                        next_move = move
                self.board.pop()

            return val, next_move

if __name__ == '__main__':
    board = BoardWrapper()
    white_player = ComputerPlayer(True, board.board)
    black_player = HumanPlayer(False)
    cur_player = white_player

    while(board.playing):
        os.system('clear')
        board.draw()
        while(True):
            if board.move(cur_player.get_next_move()):
                break
            print("Not a valid move")
        cur_player = black_player if cur_player.colour else white_player
        
        
