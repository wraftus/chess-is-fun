#!/usr/bin/env python3
import numpy as np
import argparse
from dataset import ChessDataset
import tensorflow as tf

if __name__ == "__main__":
    # check if we want to recreate dataset or not
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_train', dest='create_train', action='store_true',
                        help="flag to recreate the train dataset from pgns")
    parser.set_defaults(create_train = False)
    parser.add_argument('--create_val', dest='create_val', action='store_true',
                        help="flag to recreate the validation dataset from pgns")
    parser.set_defaults(create_val = False)
    args = parser.parse_args()

    train_dataset = ChessDataset(args.create_train, True, 1000000)
    val_dataset = ChessDataset(args.create_val, False, 100000)
    
    print("Training Set: X ", train_dataset.X.shape, " and y ", train_dataset.y.shape)
    print("Validation Set: X ", val_dataset.X.shape, " and y ", val_dataset.y.shape)

    # create tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset.X, train_dataset.y))
    train_dataset = train_dataset.shuffle(1000000).batch(256)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset.X, val_dataset.y))
    val_dataset = val_dataset.batch(256)

    # create CNN model
    model = tf.keras.models.Sequential()
   
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(8,8,5)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2))
    
    model.add(tf.keras.layers.ZeroPadding2D((1,1)))
    model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
    
    model.add(tf.keras.layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (1, 1), activation='relu'))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='tanh'))

    model.summary()

    # compile, train and evaluate model
    model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy'])
    model.fit(train_dataset, epochs=50)
    
    print("Evaluating Model ...")
    model.evaluate(val_dataset)

    # save model
    print("Saving model ...")
    model.save('state_model')
