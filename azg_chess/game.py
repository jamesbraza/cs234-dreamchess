from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias

import chess
import numpy as np
from azg.Game import Game

if TYPE_CHECKING:
    import numpy.typing as npt

    from azg_chess.players import Player

BOARD_DIMENSIONS = (8, 8)
NUM_SQUARES = len(chess.SQUARES)
ACTION_INDICES = np.arange(NUM_SQUARES**2, dtype=int).reshape(NUM_SQUARES, -1)

# Just FYI
AZ_PAPER_ACTION_SPACE_SIZE = 64 * 73
# SEE: https://ai.stackexchange.com/a/34733
NORMAL_CASE_ACTION_SPACE_SIZE = 1924

Board: TypeAlias = chess.Board
ActionIndex: TypeAlias = int
PlayerID: TypeAlias = int
PLAYER_1 = 1
PLAYER_2 = -1


def to_action(uci: str | chess.Move) -> ActionIndex:
    """Convert a Universal Chess Interface (UCI) string or Move to an action."""
    if isinstance(uci, chess.Move):
        uci: str = uci.uci()
    origin_index = chess.SQUARE_NAMES.index(uci[0:2])
    destination_index = chess.SQUARE_NAMES.index(uci[2:4])
    return ACTION_INDICES[origin_index, destination_index]


class ChessGame(Game):
    """Game adaptation for the chess library."""

    def __init__(self, player_1: Player, player_2: Player):
        super().__init__()
        self.players: dict[PlayerID, Player] = {PLAYER_1: player_1, PLAYER_2: player_2}

    def getInitBoard(self) -> Board:
        """Get a board representation."""
        return chess.Board()

    def getBoardSize(self) -> tuple[int, int]:
        """Get the board's dimensions."""
        return BOARD_DIMENSIONS

    def getActionSize(self) -> int:
        """Get the size of the action space |A|."""
        return math.prod(ACTION_INDICES.shape)

    def getNextState(
        self, board: Board, player: PlayerID, action: ActionIndex
    ) -> tuple[Board, PlayerID]:
        b = board.copy()  # TODO: is this necessary?

        origin_index = np.where(ACTION_INDICES == action)[0][0]
        destination_index = np.where(ACTION_INDICES == action)[1][0]

        uci = f"{chess.square_name(origin_index)}{chess.square_name(destination_index)}"
        # TODO: verify this, and need expanding?
        if b.piece_at(origin_index).piece_type == chess.PAWN and chess.square_rank(
            destination_index
        ) in [0, 7]:
            uci += "q"  # Promotion

        b.push(move=chess.Move.from_uci(uci))
        return b.mirror(), -1 * player  # TODO: why mirror?

    def getValidMoves(self, board: Board, player: PlayerID) -> npt.NDArray[bool]:
        """
        Get a vector that identifies moves as invalid False or valid True.

        Args:
            board: Current board.
            player: Current player ID.

        Returns:
            Vector of size self.getActionSize() where each element is a
                False (invalid move) or True (valid move).
        """
        return self.players[player].get_valid_moves(board)

    UNFINISHED_REWARD = 0
    WON_REWARD = 1
    LOST_REWARD = -1
    DRAW_REWARD = 1e-5  # Small non-zero value

    def getGameEnded(self, board: Board, player: PlayerID) -> float:
        """Get the current reward associated with the board and player."""
        # TODO: use player?
        result: str = board.result()
        if result == "*":
            return self.UNFINISHED_REWARD
        player_outcome = result.split("-")[0]
        if player_outcome == "1":
            return self.WON_REWARD
        if player_outcome == "0":
            return self.LOST_REWARD
        assert player_outcome == "1/2"  # Confirm no other possibilities
        return self.DRAW_REWARD

    def getCanonicalForm(self, board: Board, player: PlayerID):
        raise NotImplementedError

    def getSymmetries(self, board: Board, pi):
        raise NotImplementedError

    def stringRepresentation(self, board: Board) -> str:
        """Represent the board as a string, for MCTS hashing."""
        # NOTE: fen = Forsyth-Edwards Notation (FEN)
        # FEN is a single line of text that captures a board's state, enabling
        # lightweight sharing of a complete game state
        return board.fen()
