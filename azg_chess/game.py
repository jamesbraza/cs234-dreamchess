from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import chess
import numpy as np
from azg.Game import Game

if TYPE_CHECKING:
    import numpy.typing as npt


BOARD_DIMENSIONS = (8, 8)
NUM_SQUARES = len(chess.SQUARES)  # 64
ACTION_INDICES = np.arange(NUM_SQUARES**2, dtype=int).reshape(NUM_SQUARES, -1)
_EIGHTH_RANK = range(chess.A8, chess.H8 + 1)

Board: TypeAlias = chess.Board
ActionIndex: TypeAlias = int
PlayerID: TypeAlias = int
WHITE_PLAYER: PlayerID = 1  # White
BLACK_PLAYER: PlayerID = -1  # Black
State: TypeAlias = tuple[Board, PlayerID]
Policy: TypeAlias = Sequence[float]


def move_to_action(move: str | chess.Move) -> ActionIndex:
    """Convert a Universal Chess Interface (UCI) string or Move to an action."""
    if isinstance(move, str):
        move = chess.Move.from_uci(move)
    return move.from_square * NUM_SQUARES + move.to_square


def action_to_move(action: ActionIndex) -> chess.Move:
    return chess.Move(
        from_square=int(action / NUM_SQUARES), to_square=action % NUM_SQUARES
    )


INVALID_MOVE = False
VALID_MOVE = True


def get_moves(game: Game, board: Board) -> npt.NDArray[bool]:
    """
    Get a vector that identifies moves as invalid False or valid True.

    Args:
        game: Current game.
        board: Current board.

    Returns:
        Vector of size game.getActionSize() where each element is a
            False (invalid move) or True (valid move).
    """
    valids = np.zeros(game.getActionSize(), dtype=bool)
    for move in board.legal_moves:
        valids[move_to_action(move)] = VALID_MOVE
    return valids


class ChessGame(Game):
    """Game adaptation for the chess library."""

    def getInitBoard(self) -> Board:
        """Get a board representation at the start of a match."""
        return chess.Board()

    def getBoardSize(self) -> tuple[int, int]:
        """Get the board's dimensions."""
        return BOARD_DIMENSIONS

    def getActionSize(self) -> int:
        """Get the size of the action space |A|."""
        # AlphaGo Zero used (8x8)x73
        # Best case is 1924, SEE: https://ai.stackexchange.com/a/34733
        return math.prod(ACTION_INDICES.shape)

    def getNextState(
        self, board: Board, player: PlayerID, action: ActionIndex
    ) -> State:
        move = action_to_move(action)
        if (
            board.piece_at(move.from_square).piece_type == chess.PAWN
            and chess.square_rank(move.to_square) in _EIGHTH_RANK
        ):
            move.promotion = chess.QUEEN  # Assume always queening
        board.push(move=move)  # NOTE: this flips board.turn
        return board, -1 * player

    INVALID_MOVE = False
    VALID_MOVE = True

    def getValidMoves(self, board: Board, player: PlayerID) -> Sequence[bool]:
        """
        Get a vector that identifies moves as invalid False or valid True.

        Args:
            board: Current board.
            player: ID of the player who needs to move.

        Returns:
            Vector of size self.getActionSize() where each element is a
                False (invalid move) or True (valid move).
        """
        return get_moves(game=self, board=board)

    UNFINISHED_REWARD = 0
    WON_REWARD = 1
    LOST_REWARD = -1
    DRAW_REWARD = 1e-5  # Small non-zero value

    def getGameEnded(self, board: Board, player: PlayerID) -> float:
        """Get the current reward associated with the board and player."""
        result: str = board.result()
        if result == "*":
            return self.UNFINISHED_REWARD
        white_player_outcome = result.split("-")[0]
        match player == WHITE_PLAYER, white_player_outcome == "1", white_player_outcome == "0":
            case (True, True, _) | (False, _, True):
                return self.WON_REWARD
            case (True, _, True) | (False, True, _):
                return self.LOST_REWARD
        assert white_player_outcome == "1/2"  # Confirm no other possibilities
        return self.DRAW_REWARD

    def getCanonicalForm(self, board: Board, player: PlayerID) -> Board:
        """
        Get a player-independent ("canonical") representation of the board.

        When you play chess, you view the board from one angle (player's point
        of view).  The canonical form is the board, as viewed by the player.
        For chess, the canonical form is from white player's point of view.
        If the black player is moving, invert the colors and flip vertically.

        Args:
            board: Current board.
            player: ID of the player who needs to move.

        Returns:
            Canonical form of the board.
        """
        if board.turn == chess.BLACK:
            # Mirror vertically, swap piece colors, flip board.turn, etc.
            board.apply_mirror()
        return board  # NOTE: this is not a copy

    def getSymmetries(self, board: Board, pi: Policy) -> list[tuple[Board, Policy]]:
        """
        Get symmetrical board representations to expand training data.

        Args:
            board: Current board.
            pi: Policy vector of size self.getActionSize().

        Returns:
            List of tuples of symmetrical board, corresponding policy.
        """
        if not isinstance(pi, np.ndarray):
            pi = np.array(pi, dtype=float)
        action_size = self.getActionSize()
        assert pi.shape == (action_size,)
        # 1. No flip
        symmetries = [(board, pi)]
        pi = pi.reshape(NUM_SQUARES, NUM_SQUARES)
        # 2. Horizontal flip
        symmetries.append(
            (board.transform(chess.flip_horizontal), np.flip(pi, axis=1).flatten())
        )
        # TODO: other flips?
        return symmetries

    def stringRepresentation(self, board: Board) -> str:
        """Represent the board as a string, for MCTS hashing."""
        # NOTE: fen = Forsyth-Edwards Notation (FEN)
        # FEN is a single line of text that captures a board's state, enabling
        # lightweight sharing of a complete game state
        return board.fen()
