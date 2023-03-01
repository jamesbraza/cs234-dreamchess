from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import chess
import numpy as np
import numpy.typing as npt

from azg_chess.game import PLAYER_1, to_action

if TYPE_CHECKING:
    from azg_chess.game import ActionIndex, Board, ChessGame, PlayerID


class Player(ABC):
    """Base class for a chess player."""

    def __init__(self, game: ChessGame, player_id: PlayerID = PLAYER_1):
        self.game = game
        self._player = player_id

    @property
    def id(self) -> PlayerID:
        return self._player

    INVALID_MOVE = False
    VALID_MOVE = True

    def get_valid_moves(self, board: Board) -> npt.NDArray[bool]:
        """
        Get a vector that identifies moves as invalid False or valid True.

        Args:
            board: Current board.

        Returns:
            Vector of size self.getActionSize() where each element is a
                False (invalid move) or True (valid move).
        """
        valids = np.zeros(self.game.getActionSize(), dtype=bool)
        for move in board.legal_moves:
            valids[to_action(move)] = self.VALID_MOVE
        return valids

    @abstractmethod
    def act(self, board: Board) -> ActionIndex:
        """Choose an action given the board."""


class RandomPlayer(Player):
    """A random player randomly selects a valid action."""

    DEFAULT_SEED = 42

    def __init__(
        self, game: ChessGame, player_id: PlayerID = PLAYER_1, seed: int = DEFAULT_SEED
    ):
        super().__init__(game, player_id)
        self._rng = np.random.default_rng(seed)

    def act(self, board: Board) -> ActionIndex:
        valid_moves = self.get_valid_moves(board)
        return self._rng.choice(valid_moves.nonzero()[0])


class HumanChessPlayer(Player):
    """Human players choose actions based on a user inputting a UCI string."""

    def act(self, board: Board) -> ActionIndex:
        while True:
            uci_input = input("Please input a valid UCI string: ")
            try:
                move = chess.Move.from_uci(uci_input)
                break
            except chess.InvalidMoveError:
                print(f"Invalid UCI {uci_input}.")
        return to_action(move)
