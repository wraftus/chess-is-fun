#!/usr/bin/env python3
import os
import chess.pgn
import numpy as np
from state import State

def create_dataset(num_samples=None):
    result_values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    X, y, = [], []
    games = 0

    # read in each pgn file
    for fn in os.listdir("pgns"):
        pgn = open(os.path.join("pgns", fn))
        while True:
            # load in next game from pgn
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break

            # get value for result from game
            result = game.headers['Result']
            if result not in result_values:
                continue
            value = result_values[result]
            
            # go through and serialize each move, and assign it a value
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                serialized = State(board).serialize()
                X.append(serialized)
                y.append(value)
            games += 1
            print("Parsed game %d, dataset now has %d moves" % (games, len(X)))
            
            # check if we have enough data
            if num_samples is not None and len(X) > num_samples:
                return np.array(X), np.array(y)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = create_dataset(10000)
    np.savez("data/dataset.npz", X, y)
