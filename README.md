# Chess is fun
A simple chess playing AI for fun  
Using a neural network to caluclate value of board states and minimax to look ahead

## How to play
Simply run `game.py` to start a game  
Enter moves in the form `A5 A6` (first select the piece then the position to move it to)  
Currently, the AI always starts  
You will need to have these python packages installed to run it:
- `python-chess`
- `numpy`
- `tensor-flow`

## How to train
Chess matches are stored in `pgn` files, and the ones I used to train can be found [here](https://archive.org/download/KingBase2018), but you can really use any ones that you want  
Put the train pgns into a `train_pgns` folder, validation pgns into `val_pgns`, and create a `dataset` folder to hold the parsed pgns  
Run `train.py` with `--create_train` and/or `--create_val` options to parse the train and validation pgns according to 'state/py', and then train the model again, which will be saved in `state_model`
Currently, the model will fit the train dataset pretty well, but preforms kinda poorly on the validation set
