"""
Script used to create a random player
and play against the decision transformer
Decision transformer plays as white.
White (random player) starts the game. 
"""

import argparse
import torch

import chess
import random

import time
from IPython.display import display, HTML, clear_output

from chessformers.configuration import get_configuration
from chessformers.model import Transformer
from chessformers.tokenizer import Tokenizer


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Chessformers inference parser')

    parser.add_argument('--load_model', type=str, default="model/chessformer_epoch_1.pth",
                        help='model to load and do inference')
    parser.add_argument('--config', type=str, default="configs/default.yaml",
                        help='location of the configuration file (a yaml)')
    parser.add_argument('--tokenizer', type=str, default="vocabs/kaggle2_vocab.txt",
                        help='location of the tokenizer file')

    args = parser.parse_args()
    return args


def main(args) -> None:
    config = get_configuration(args.config)
    tokenizer = Tokenizer(args.tokenizer)
    model = Transformer(tokenizer,
                        num_tokens=tokenizer.vocab_size(),
                        dim_model=config["model"]["dim_model"],
                        d_hid=config["model"]["d_hid"],
                        num_heads=config["model"]["num_heads"],
                        num_layers=config["model"]["num_layers"],
                        dropout_p=config["model"]["dropout_p"],
                        n_positions=config["model"]["n_positions"],
                        )
    model.load_state_dict(torch.load(args.load_model))

    print(
        "===== Play against a random player =====\n"
     )
    
    # Create board instance
    board = chess.Board()
    
    # Define an initial state
    #input_string = "<bos>"
    board_history = ["h2h4"]
    
    # Define a random player
    def random_player(board, board_history): 
        move = random.choice(list(board.legal_moves))
        board_history.append(move.uci())
        return move.uci(), board_history
    
    def dt_player(board, board_history):
        if board_history is None: 
            board_history = "<bos>"
        else:
            board_history = board_history
            
        legal_uci_moves = [move.uci() for move in board.legal_moves]
        
        move = model.predict(
            " ".join(board_history), 
            stop_at_next_move=True, 
            temperature=0.2,)
        
        if move in legal_uci_moves: 
            board_history.append(move)
            return move, board_history
        else: 
            move = model.predict(
                " ".join(board_history), 
                stop_at_next_move=True, 
                temperature=0.2,)
            board_history.append(move)
            return move, board_history
    
    def who(player): 
        return "White" if player == chess.WHITE else "Black"
    
    def display_board(board, use_svg):
        if use_svg:
            return board._repr_svg_()
        else:
            return "<pre>" + str(board) + "</pre>"
    
    def play_game(player1, player2, visual="svg", pause=0.1):
        """
        playerN1, player2: functions that takes board, board_history and return uci move and updated board history
        visual: "simple" | "svg" | None
    
        """
        use_svg = (visual == "svg")
        board = chess.Board()
        board_history = []
        try:
            while not board.is_game_over(claim_draw=True):
                if board.turn == chess.WHITE:
                    uci,board_history = player1(board, board_history)
                else:
                    uci, board_history = player2(board, board_history)
                name = who(board.turn)
                board.push_uci(uci)
                board_stop = display_board(board, use_svg)
                html = "<b>Move %s %s, Play '%s':</b><br/>%s" % (
                    len(board.move_stack), name, uci, board_stop)
                if visual is not None:
                    if visual == "svg":
                        clear_output(wait=True)
                    display(HTML(html))
                    if visual == "svg":
                        time.sleep(pause)
        except KeyboardInterrupt:
            msg = "Game interrupted!"
            return (None, msg, board, board_history)
        
        result = None
        if board.is_checkmate():
            msg = "checkmate: " + who(not board.turn) + " wins!"
            result = not board.turn
        elif board.is_stalemate():
            msg = "draw: stalemate"
        elif board.is_fivefold_repetition():
            msg = "draw: 5-fold repetition"
        elif board.is_insufficient_material():
            msg = "draw: insufficient material"
        elif board.can_claim_draw():
            msg = "draw: claim"
        if visual is not None:
            print(msg)
        return (result, msg, board, board_history)
    
    counts = {None: 0, True: 0, False: 0}
    for i in range(10):
        result, msg, board, board_history = play_game(random_player, dt_player, visual=None)
        counts[result] += 1
        print(counts)

    counts
    
if __name__ == "__main__":
    args = _parse_args()
    main(args)
