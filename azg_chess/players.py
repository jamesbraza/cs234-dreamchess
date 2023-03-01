from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import chess
import chess.engine
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
    """Player that randomly selects a valid action."""

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
    """Player that chooses an action based on user input of a UCI string."""

    def act(self, board: Board) -> ActionIndex:
        while True:
            uci_input = input("Please input a valid UCI string: ")
            try:
                move = chess.Move.from_uci(uci_input)
                break
            except chess.InvalidMoveError:
                print(f"Invalid UCI {uci_input}.")
        return to_action(move)


class StockfishPlayer(Player):
    """
    Player whose decisions are made by the Stockfish chess engine.

    NOTE: development was done with Stockfish 15.1.
    SEE: https://stockfishchess.org/
    """

    # NOTE: this is the path for macOS after `brew install stockfish`
    DEFAULT_STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

    def __init__(
        self,
        game: ChessGame,
        player_id: PlayerID = PLAYER_1,
        stockfish_path: str = DEFAULT_STOCKFISH_PATH,
    ):
        super().__init__(game, player_id)
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def act(self, board: Board) -> ActionIndex:
        # SEE: https://python-chess.readthedocs.io/en/latest/engine.html#playing
        result = self._engine.play(board, limit=chess.engine.Limit(time=0.1))
        assert result.move is not None, "Stockfish didn't pick a best move."
        return to_action(result.move)

    def __del__(self) -> None:
        self._engine.close()
