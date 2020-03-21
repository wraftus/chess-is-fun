#!/usr/bin/env python3
import os
import chess.pgn
import numpy as np
from state import State
import argparse
import tensorflow as tf

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
    # check if we want to recreate dataset or not
    parser = argparse.ArgumentParser()
    parser.add_argument('--recreate_dataset', dest='recreate_dataset', action='store_true',
                        help="flag to recreate the data set from pgns")
    parser.set_defaults(recreate_dataset = False)
    args = parser.parse_args()

    if args.recreate_dataset:
        # recreate and save dataset
        X, y = create_dataset(50000)
        np.savez("data/dataset.npz", X, y)
    else:
        # load in dataset
        dataset = np.load("data/dataset.npz")
        X = dataset['arr_0']
        y = dataset['arr_1']
    print("Loaded in X ", X.shape, " and y ", y.shape)

    # create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(100).batch(64)

    for item in dataset:
        print(item)
        break
    
    # create CNN model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(8,8,5)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    
    model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same',  strides=2))

    model.add(tf.keras.layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (1, 1), activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='tanh'))

    model.summary()
   
    # compile and train model
    model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy'])
    history = model.fit(X, y, epochs=10)
    print(history)
