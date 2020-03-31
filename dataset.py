import os
import chess.pgn
import numpy as np
from state import State

class ChessDataset(object):
    def __init__(self, create, train, num_samples=None):
        # load in or create new dataset
        self.X, self.y = self.create_dataset(train, num_samples) if create else self.load_dataset(train)
        
        # if we created the dataset, save it to disk so we can load it later
        if create:
            file_name = 'datasets/train.npz' if train else 'datasets/val.npz'
            np.savez(file_name, self.X, self.y)

    # create a numpy array of training examples based on pgns
    def create_dataset(self, train, num_samples):
        result_values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
        X, y, = [], []
        games = 0

        # read in each pgn file
        pgn_dir = 'train_pgns' if train else 'val_pgns' 
        for fn in os.listdir(pgn_dir):
            pgn = open(os.path.join(pgn_dir, fn))
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
                num_moves = sum(1 for i in game.mainline_moves())
                if num_moves < 2:
                    continue
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    serialized = State(board).serialize()
                    X.append(serialized)
                    y.append(value * (i/(num_moves-1)))
                games += 1
                print("Parsed game %d, dataset now has %d moves" % (games, len(X)))
                
                # check if we have enough data
                if num_samples is not None and len(X) > num_samples:
                    return np.array(X), np.array(y)
        
        return np.array(X), np.array(y)

    # loads a premade data set of numpy arrays
    def load_dataset(self, train):
        # load in dataset
        file_name = 'datasets/train.npz' if train else 'datasets/val.npz'
        dataset = np.load(file_name)
        return dataset['arr_0'], dataset['arr_1']
