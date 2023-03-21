# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:42:54 2023

@author: h_min
"""

import chess.pgn

# Load the PGN file
pgn_file = open("example_game.pgn")


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