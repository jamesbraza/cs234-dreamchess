# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:42:54 2023

@author: h_min
"""

import chess.pgn
from glob import glob
import numpy as np
from pandas import DataFrame
import random
import string

# Load the PGN file

# Set the file path
#file_path = "C:/Users/h_min/repos/cs234-dreamchess/decision-transformer/dataset/"
file_path = "C:/Users/h_min/repos/cs234-dreamchess/decision-transformer/dataset/ficsgamesdb_2021_standard2000_nomovetimes_280587.pgn/"
fname = "ficsgamesdb_2021_standard2000_nomovetimes_280587.pgn"
# Load the data

#full_data = open(file_path + 'all_with_filtered_anotations_since1998.txt').readlines()

pgn = open(file_path + fname)

def alter_result (_result: str) -> int:
	if _result == "1-0":
		return +1
	elif _result == "1/2-1/2":
		return 0
	elif _result == "0-1":
		return -1
	raise ValueError()


def board_as_matrix (_board: chess.Board):
	empty_space = ord('.')
	matrix = [[ord(y) - empty_space for y in x.split(" ")] for x in str(_board).split("\n")]
	return np.asarray(matrix)


data_frame = DataFrame()

counter = 0
max_counter = 1e5
while True:
	pgn_game = chess.pgn.read_game(pgn)
	if pgn_game is None or counter > int(max_counter):
		break

	counter += 1
	# print(round(counter/max_counter*100, 2), "%", sep="")

	result = alter_result(pgn_game.headers["Result"])

	board = pgn_game.board()

	boards = []
	for move in pgn_game.mainline_moves():
		board.push(move)
		boards.append(board_as_matrix(board))

	game_data_frame = DataFrame(data=np.array(boards).reshape((len(boards), 64,)),
	                            columns=["p" + str(x + 1) for x in range(64)])
	game_data_frame["result"] = result

	data_frame = data_frame.append(game_data_frame, ignore_index=True)

s = "".join([random.choice(string.ascii_letters) for _ in range(10)])
data_frame.to_csv(file_path + "board_converted_{}.csv".format(s.lower()), index=False)






"""
# Read games in the PGN file and store them in games list 
games = []
while True:
    # Read the next game
    pgn_game = chess.pgn.read_game(pgn_file)
    
    if pgn_game is not None:
        games.append(pgn_game)
    else: 
        break
    
# Create observation data for state embedding     
game_boards = {}
game_idx = 0 
for game in games[]:
    game_board[game_idx] 
    board = game.board()

# Test
pgn_game = games[0]
board = pgn_game.board()
# Iterate through each move and update the board
for move in pgn_game.mainline_moves():
    board.push(move)
    print('\n')
    print(board)

# Print the final board state
print(board)


# Iterate through each game in the PGN file
while True:
    # Read the next game
    pgn_game = chess.pgn.read_game(pgn_file)

    # Check if there are no more games left
    if pgn_game is None:
        break

    # Get the starting position of the game
    board = pgn_game.board()

    # Iterate through each move and update the board
    for move in pgn_game.mainline_moves():
        board.push(move)

    # Print the final board state
    print(board)
    
    
"""