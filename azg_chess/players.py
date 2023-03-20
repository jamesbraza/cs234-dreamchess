from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar

import chess
import chess.engine
import numpy as np
import torch
from azg.MCTS import MCTS
from azg.utils import dotdict

import geochri.src.alpha_net as alpha_net
import geochri.src.chess_utils as chess_utils
import geochri.src.MCTS_chess as MCTS_chess
import geochri.src.encoder_decoder as encoder_decoder


from azg_chess.game import WHITE_PLAYER, Board, action_to_move, move_to_action
from azg_chess.nn import NNetWrapper

if TYPE_CHECKING:
    from azg_chess.game import ActionIndex, ChessGame, PlayerID


TBoard_contra = TypeVar("TBoard_contra", contravariant=True)


class Player(Protocol[TBoard_contra]):
    """Player can choose a move given a board."""

    def __call__(self, board: TBoard_contra) -> ActionIndex:
        """
        Call to the player chooses (but doesn't take) an action.

        Args:
            board: Canonicalized board, don't mutate.

        Returns:
            Action index of a move on a canonicalized board.
        """


class ChessPlayer(Player[Board], ABC):
    """Base class for a chess player."""

    def __init__(self, player_id: PlayerID = WHITE_PLAYER):
        super().__init__()
        self._player = player_id

    @property
    def id(self) -> PlayerID:
        return self._player

    @abstractmethod
    def choose_move(self, board: Board) -> chess.Move:
        """Choose (but don't make) a move given the board."""

    def __call__(self, board: Board) -> ActionIndex:
        return move_to_action(self.choose_move(board))


class RandomChessPlayer(ChessPlayer):
    """Player that randomly selects a valid action."""

    DEFAULT_SEED = 42

    def __init__(self, player_id: PlayerID = WHITE_PLAYER, seed: int = DEFAULT_SEED):
        super().__init__(player_id)
        self._rng = np.random.default_rng(seed)

    def choose_move(self, board: Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))  # type: ignore[arg-type]


class HumanChessPlayer(ChessPlayer):
    """Player that chooses an action based on user input of a UCI string."""

    def choose_move(self, board: Board) -> chess.Move:
        while True:
            uci_input = input("Please input a valid UCI string: ")
            try:
                return chess.Move.from_uci(uci_input)
            except chess.InvalidMoveError:
                print(f"Invalid UCI {uci_input}.")


class StockfishChessPlayer(ChessPlayer):
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


class MCTSArgs(NamedTuple):
    """
    Data structure to configure Monte-Carlo Tree Search object.

    TODO: remove in favor of MCTS.default_args from
    https://github.com/suragnair/alpha-zero-general/pull/300
    """

    numMCTSSims: int = 25  # Number of moves for MCTS to simulate.
    cpuct: float = 1.0  # PUCT exploration constant


class AlphaZeroChessPlayer(ChessPlayer):
    """Players whose decides via a trained AlphaGo Zero-style network."""

    DEFAULT_MCTS_ARGS = MCTSArgs()

    def __init__(
        self,
        game: ChessGame,
        player_id: PlayerID = WHITE_PLAYER,
        mcts_args: MCTSArgs | dotdict = DEFAULT_MCTS_ARGS,
        parameters_path: tuple[str, str] | None = None,
        **nnet_wrapper_kwargs,
    ):
        super().__init__(player_id)
        self._nnet = NNetWrapper(game, **nnet_wrapper_kwargs)
        if parameters_path is not None:
            self._nnet.load_checkpoint(*parameters_path)
        self._mcts = MCTS(game, self._nnet, mcts_args)

    def choose_move(self, board: Board) -> chess.Move:
        return action_to_move(action=self(board))

    def __call__(self, board: Board) -> ActionIndex:
        return np.argmax(self._mcts.getActionProb(board, temp=0))


class GeochriPlayer(ChessPlayer):
    """
    Player whose decisions are made by the NN from the [Geochri repository][1].

    [1]: https://github.com/geochri/AlphaZero_Chess
    """

    def __init__(
        self,
        player_id: PlayerID = WHITE_PLAYER,
        parameters_file: str | None = None,
        mcts_steps_per_move: int = 50,
    ):
        super().__init__(player_id)
        self.mcts_steps_per_move = mcts_steps_per_move

        self._net = alpha_net.ChessNet()
        if torch.cuda.is_available():
            self._net.cuda()
        self._net.share_memory()
        self._net.eval()
        if parameters_file is not None:
            checkpoint = torch.load(
                parameters_file,
                map_location=None if torch.cuda.is_available() else torch.device("cpu"),
            )
            self._net.load_state_dict(checkpoint["state_dict"])

    def choose_move(self, board: Board) -> chess.Move:
        geochri_board = chess_utils.load_chessboard_to_Geochri(board)

        best_move, _ = MCTS_chess.UCT_search(
            geochri_board, num_reads=self.mcts_steps_per_move, net=self._net
        )
        i_pos, f_pos, prom = encoder_decoder.decode_action(
            geochri_board, encoded=best_move
        )

        if prom[0] is None:
            promotion_piece = None
        else:
            if prom[0] in chess.PIECE_SYMBOLS:
                index = chess.PIECE_SYMBOLS.index(prom[0])
                promotion_piece = chess.PIECE_TYPES[index]
            elif prom[0].lower() in chess.PIECE_SYMBOLS:
                index = chess.PIECE_SYMBOLS.index(prom[0].lower())
                promotion_piece = chess.PIECE_TYPES[index]

        from_square = chess.parse_square(chr(i_pos[0][1] + 97) + str(8 - i_pos[0][0]))
        to_square = chess.parse_square(chr(f_pos[0][1] + 97) + str(8 - f_pos[0][0]))
        return chess.Move(from_square, to_square, promotion_piece)
