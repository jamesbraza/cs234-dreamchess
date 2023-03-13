import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import chess
import numpy as np
from azg.Game import Game

from azg_chess.chess_utils import to_display

if TYPE_CHECKING:
    from collections.abc import Callable

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


def get_moves(game: Game, board: Board) -> "npt.NDArray[bool]":
    """
    Get a vector that identifies moves as invalid False or valid True.

    Args:
        game: Game.
        board: Board, don't mutate.

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
        """
        Apply the action to the board in-place, returning it and next player.

        Args:
            board: Un-canonicalized board, to be mutated.
            player: ID of the player taking the action.
            action: Action index of a move on a canonicalized board.

        Returns:
            Tuple of next board, next turn's player ID.
        """
        # Canonicalize the board since the action passed in corresponds with
        # coordinates on a canonicalized board
        board = self.getCanonicalForm(board, player)
        move = action_to_move(action)
        if (
            board.piece_type_at(move.from_square) == chess.PAWN
            and chess.square_rank(move.to_square) in _EIGHTH_RANK
        ):
            move.promotion = chess.QUEEN  # Assume always queening
        board.push(move=move)  # NOTE: this in-place flips board.turn
        if player == BLACK_PLAYER:
            # This synchronizes player concept with board's turn concept, so if:
            # - Next player is white: board's turn is white
            # - Next player is black: board's turn is black
            board.apply_mirror()  # NOTE: in-place mirror, not a copy
        return board, -1 * player

    INVALID_MOVE = False
    VALID_MOVE = True

    def getValidMoves(self, board: Board, player: PlayerID) -> Sequence[bool]:
        """
        Get a vector that identifies moves as invalid False or valid True.

        Args:
            board: Canonicalized board, don't mutate.
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
        """
        Get the reward associated with a game state.

        Args:
            board: Un-canonicalized board, don't mutate.
            player: ID of the player to check if won/lost.

        Returns:
            Reward associated with the game state.
        """
        white_result: str = board.result()
        if white_result == "*":
            return self.UNFINISHED_REWARD
        white_player_outcome = white_result.split("-")[0]
        match (
            player == WHITE_PLAYER,
            white_player_outcome == "1",
            white_player_outcome == "0",
        ):
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
            board: Board to be canonicalized, can mutate the board.
            player: ID of the player who needs to move.

        Returns:
            Canonical form of the board.
        """
        match board.turn == chess.BLACK, player == BLACK_PLAYER:
            case True, True:
                # In-place mirror so future canonical calls are faster
                board.apply_mirror()
            case True, False:
                raise NotImplementedError("Unreachable, by design.")
        return board  # NOTE: this is not a copy

    def getSymmetries(self, board: Board, pi: Policy) -> list[tuple[Board, Policy]]:
        """
        Get symmetrical board representations to expand training data.

        Args:
            board: Canonicalized board, don't mutate.
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

    @staticmethod
    def display(
        board, print_func: "Callable[[str], None]" = print, **to_display_kwargs
    ) -> None:
        """Display the board using the input printing function."""
        print_func(to_display(board, **to_display_kwargs))
