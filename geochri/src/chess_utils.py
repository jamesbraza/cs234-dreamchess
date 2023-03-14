import chess.pgn
import io
import string
from chess_board import board as c_board
import encoder_decoder as ed
import copy
import geochri.src.chess_board as gb
import numpy as np
import chess


def make_string_geochri(current_board):
    s = ""
    for i in range(8):
        for j in range(8):
            c = current_board[i][j]
            if c == " ":
                c = "*"
            s += c
    return s

def make_string_chess(board):
    s = ""
    for i in range(8):
        for j in range(8):
            c = board.piece_at(chess.square(j, 7 - i))
            if c == None:
                c = "*"
            else:
                c = c.symbol()
            s += c
    return s

"""
def make_FEN_geochri(current_board): 
    s = ""
    #1 add piece positions
    empty_counter = 0
    for i in range(8):
        for j in range(8):
            c = current_board[i][j]
            if c == " ":
                empty_counter += 1
            else:
                if empty_counter != 0:
                    s += str(empty_counter)
                s += c
                empty_counter = 0

        if empty_counter != 0:
            s += str(empty_counter)
            empty_counter = 0
        s += '/'
    #2 'w' or 'b', indicates whos move is now

    return s
"""

def load_chessboard_to_Geochri(board:chess.Board):
    FEN_parts = board.fen().split(" ")
    geochri_board = gb.board()

    #1 load positions
    temp_board = []
    for row in FEN_parts[0].split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend([' '] * int(c))
            else:
                brow.append(c)

        temp_board.append(brow)

    for i in range(8):
        for j in range(8):
            geochri_board.current_board[i][j] = temp_board[i][j]

    #2 load player who moves next
    if FEN_parts[1] == 'w':
        geochri_board.player = 0
    else:
        geochri_board.player = 1

    #3 load castling rights
    geochri_board.r1_move_count = 1  # black's Queenside rook
    geochri_board.r2_move_count = 1  # black's Kingside rook
    geochri_board.R1_move_count = 1  # white's Queenside rook
    geochri_board.R2_move_count = 1  # white's Kingside rook

    if 'K' in FEN_parts[2]:
        geochri_board.R2_move_count = 0
    if 'Q' in FEN_parts[2]:
        geochri_board.R1_move_count = 0
    if 'k' in FEN_parts[2]:
        geochri_board.r2_move_count = 0
    if 'q' in FEN_parts[2]:
        geochri_board.r1_move_count = 0

    #4 load castling rights
    geochri_board.en_passant = -999
    geochri_board.en_passant_move = 0
    if FEN_parts[3] != '-':
        geochri_board.en_passant = string.ascii_lowercase.index(FEN_parts[3][0])
        geochri_board.en_passant_move = string.ascii_lowercase.index(FEN_parts[3][0])


    #5 halfmove count
    geochri_board.no_progress_count = int(FEN_parts[4])

    #6 full move count
    geochri_board.no_progress_count = int(FEN_parts[4])

    #7 check repetitions
    geochri_board.repetitions_w = 0
    geochri_board.repetitions_b = 0

    if (board.is_repetition(1)):
        geochri_board.repetitions_w = 1
        geochri_board.repetitions_b = 1


    if (board.is_repetition(2)):
        geochri_board.repetitions_w = 2
        geochri_board.repetitions_b = 2

    if (board.can_claim_threefold_repetition):
        geochri_board.repetitions_w = 3
        geochri_board.repetitions_b = 3



    return geochri_board







def one_hot_policy(policy_idx):
    policy = np.zeros([4672], dtype=np.float32)
    policy[policy_idx] = 1
    return policy


