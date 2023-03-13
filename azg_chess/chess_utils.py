import os
from typing import TYPE_CHECKING

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
