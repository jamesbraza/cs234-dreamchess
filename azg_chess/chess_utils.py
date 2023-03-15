import os
from typing import TYPE_CHECKING, Literal

import chess

if TYPE_CHECKING:
    from collections.abc import Callable


def to_display(board: chess.Board, *, verbosity: int = 0, pretty: bool = False) -> str:
    """
    Convert the input board to a displayable representation.

    SEE: https://github.com/niklasf/python-chess/issues/971#issuecomment-1464138409

    Args:
        board: Board to display.
        verbosity: Verbosity level (0: no labels, 1: compressed labels,
            2: full labels).
        pretty: Prettify the print (use unicode characters).

    Returns:
        String representation to pass to print.
    """
    rows = board.unicode(empty_square=".") if pretty else str(board).split("\n")
    if verbosity == 0:
        return os.linesep.join(rows)
    if verbosity == 1:
        board_string = os.linesep.join(f"{8-i} {row}" for i, row in enumerate(rows))
        return f"  a b c d e f g h{os.linesep}{board_string}"
    if verbosity == 2:
        top_corner = "*" if board.turn == chess.BLACK else "."
        bottom_corner = "*" if board.turn == chess.WHITE else "."
        board_string = os.linesep.join(
            f"{8-i} | {row} | {8-i}" for i, row in enumerate(rows)
        )
        return (
            f"    a b c d e f g h{os.linesep}"
            f"  {top_corner} --------------- {top_corner}{os.linesep}"
            f"{board_string}{os.linesep}"
            f"  {bottom_corner} --------------- {bottom_corner}{os.linesep}"
            f"    a b c d e f g h"
        )
    raise NotImplementedError(f"Unimplemented verbosity {verbosity}.")


def make_display_func(
    print_func: "Callable[[str], None]" = print, **to_display_kwargs
) -> "Callable[[chess.Board], None]":
    """Get a function to display a board."""
    return lambda b: print_func(to_display(b, **to_display_kwargs))


pprint = make_display_func(verbosity=2)  # For convenience

ICC_K_FACTOR = 32


def update_elo(
    p1_elo: int, p2_elo: int, winner: Literal[-1, 0, 1], k: int = ICC_K_FACTOR
) -> tuple[int, int]:
    """
    Calculate the new Elo for P1 and P2 after the match.

    SEE: https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/

    Args:
        p1_elo: Player 1's original Elo.
        p2_elo: Player 2's original Elo.
        winner: Match outcome, 1 if P1 won, 0 if a tie, -1 if P2 won.
        k: K-factor for match weight.

    Returns:
        Two tuple of new Elo for player 1 and player 2.
    """
    r1, r2 = 10 ** (p1_elo / 400), 10 ** (p2_elo / 400)
    sum_r = r1 + r2
    e1, e2 = r1 / sum_r, r2 / sum_r
    s1, s2 = 0.5 * winner + 0.5, -0.5 * winner + 0.5
    return round(p1_elo + k * (s1 - e1)), round(p2_elo + k * (s2 - e2))
