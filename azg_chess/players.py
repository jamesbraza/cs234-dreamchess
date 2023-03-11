from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import chess
import chess.engine
import numpy as np

from azg_chess.game import WHITE_PLAYER, move_to_action

if TYPE_CHECKING:
    from azg_chess.game import ActionIndex, Board, PlayerID


class Player(ABC):
    """Base class for a chess player."""

    def __init__(self, player_id: PlayerID = WHITE_PLAYER):
        self._player = player_id

    @property
    def id(self) -> PlayerID:
        return self._player

    @abstractmethod
    def choose_move(self, board: Board) -> chess.Move:
        """Choose (but don't make) a move given the board."""

    def __call__(self, board: Board) -> ActionIndex:
        """Call to the player chooses (but doesn't take) an action."""
        return move_to_action(self.choose_move(board))


class RandomPlayer(Player):
    """Player that randomly selects a valid action."""

    DEFAULT_SEED = 42

    def __init__(self, player_id: PlayerID = WHITE_PLAYER, seed: int = DEFAULT_SEED):
        super().__init__(player_id)
        self._rng = np.random.default_rng(seed)

    def choose_move(self, board: Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))


class HumanChessPlayer(Player):
    """Player that chooses an action based on user input of a UCI string."""

    def choose_move(self, board: Board) -> chess.Move:
        while True:
            uci_input = input("Please input a valid UCI string: ")
            try:
                return chess.Move.from_uci(uci_input)
            except chess.InvalidMoveError:
                print(f"Invalid UCI {uci_input}.")


class StockfishPlayer(Player):
    """
    Player whose decisions are made by the Stockfish chess engine.

    NOTE: development was done with Stockfish 15.1.
    SEE: https://stockfishchess.org/
    """

    # NOTE: this is the path for macOS after `brew install stockfish`
    DEFAULT_ENGINE_PATH = "/opt/homebrew/bin/stockfish"
    # SEE: https://www.chess.com/forum/view/general/what-is-a-good-chess-rating
    DEFAULT_ELO = 1600  # Lower bound of class B (good) player

    def __init__(
        self,
        player_id: PlayerID = WHITE_PLAYER,
        engine_path: str = DEFAULT_ENGINE_PATH,
        engine_elo: int = DEFAULT_ELO,
    ):
        super().__init__(player_id)
        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # NOTE: requires Stockfish 11 per here:
        # https://github.com/official-stockfish/Stockfish/issues/3358
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": engine_elo})

    def choose_move(self, board: Board) -> chess.Move:
        # SEE: https://python-chess.readthedocs.io/en/latest/engine.html#playing
        result = self._engine.play(board, limit=chess.engine.Limit(time=0.1))
        assert result.move is not None, "Stockfish didn't pick a best move."
        return result.move

    def __del__(self) -> None:
        self._engine.close()
