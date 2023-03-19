import resource

from azg.Coach import Coach

from azg_chess.game import ChessGame
from azg_chess.nn import NNetWrapper, UnsignedNNetWrapper  # noqa: F401
from azg_chess.test import CoachArgs


def memory_limit(soft_limit: int) -> None:
    """
    Limit max memory usage to a lower soft limit.

    Source: https://stackoverflow.com/a/41125461
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if not 0 < soft_limit <= soft:
        raise ValueError(f"Soft limit {soft_limit} should be in (0, {soft}].")
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard))


memory_limit(soft_limit=int(6.2e8))
chess_game = ChessGame()
nnet_wrapper = UnsignedNNetWrapper(chess_game)
coach = Coach(chess_game, nnet_wrapper, CoachArgs())
coach.learn()
