from azg.Coach import Coach

from azg_chess.game import ChessGame
from azg_chess.nn import NNetWrapper, UnsignedNNetWrapper  # noqa: F401
from azg_chess.test import CoachArgs

chess_game = ChessGame()
nnet_wrapper = UnsignedNNetWrapper(chess_game)
coach = Coach(chess_game, nnet_wrapper, CoachArgs())
coach.learn()
