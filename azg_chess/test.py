import math
from functools import partial
from operator import gt, lt
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

import chess
import pytest
from azg.Arena import Arena
from azg.Coach import Coach

from azg_chess.chess_utils import ICC_K_FACTOR, get_k_factor, update_elo
from azg_chess.game import (
    BLACK_PLAYER,
    BOARD_DIMENSIONS,
    INVALID_MOVE,
    NUM_PIECES,
    NUM_SQUARES,
    VALID_MOVE,
    WHITE_PLAYER,
    ChessGame,
    PlayerID,
)
from azg_chess.nn import NNetWrapper
from azg_chess.players import AlphaZeroChessPlayer, MCTSArgs, StockfishChessPlayer

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(name="chess_game")
def fixture_chess_game() -> ChessGame:
    return ChessGame()


@pytest.fixture(name="chess_board")
def fixture_chess_board() -> chess.Board:
    return chess.Board()


class TestGame:
    def test_constants(self) -> None:
        # Module-level constants
        assert math.prod(BOARD_DIMENSIONS) == NUM_SQUARES
        assert NUM_PIECES == 6
        assert WHITE_PLAYER == int(chess.WHITE)
        assert BLACK_PLAYER != WHITE_PLAYER
        assert not INVALID_MOVE
        assert VALID_MOVE

        # Game constants
        assert ChessGame.LOST_REWARD == -1
        assert ChessGame.WON_REWARD == 1
        assert ChessGame.UNFINISHED_REWARD == 0
        assert ChessGame.DRAW_REWARD not in [-1, 0, 1]
        assert 0 < abs(ChessGame.DRAW_REWARD) < 1

    def test_game_sizings(self, chess_game: ChessGame) -> None:
        assert chess_game.getBoardSize() == (8, 8)
        assert chess_game.getActionSize() == 4096
        assert isinstance(chess_game.getInitBoard(), chess.Board)

    def test_stringRepresentation(
        self, chess_game: ChessGame, chess_board: chess.Board
    ) -> None:
        repr_before_move = chess_game.stringRepresentation(chess_board)
        assert isinstance(repr_before_move, str)

        chess_board.push(chess.Move.from_uci("a2a4"))
        repr_after_move = chess_game.stringRepresentation(chess_board)

        reprs = [repr_after_move, repr_before_move]
        assert len(set(reprs)) == len(reprs)  # Confirm unique
        for rep in reprs:
            assert isinstance(rep, str)

    @pytest.mark.parametrize(
        ("player", "board_turn", "board_result", "ended_return", "winner"),
        [
            (WHITE_PLAYER, chess.WHITE, "1-0", ChessGame.WON_REWARD, WHITE_PLAYER),
            (WHITE_PLAYER, chess.WHITE, "0-1", ChessGame.LOST_REWARD, BLACK_PLAYER),
            (WHITE_PLAYER, chess.BLACK, "1-0", ChessGame.WON_REWARD, WHITE_PLAYER),
            (WHITE_PLAYER, chess.BLACK, "0-1", ChessGame.LOST_REWARD, BLACK_PLAYER),
            (BLACK_PLAYER, chess.WHITE, "1-0", ChessGame.LOST_REWARD, WHITE_PLAYER),
            (BLACK_PLAYER, chess.WHITE, "0-1", ChessGame.WON_REWARD, BLACK_PLAYER),
            (BLACK_PLAYER, chess.BLACK, "1-0", ChessGame.LOST_REWARD, WHITE_PLAYER),
            (BLACK_PLAYER, chess.BLACK, "0-1", ChessGame.WON_REWARD, BLACK_PLAYER),
            # fmt: off
            (WHITE_PLAYER, chess.WHITE, "1/2-1/2", ChessGame.DRAW_REWARD, ChessGame.DRAW_REWARD),
            (WHITE_PLAYER, chess.BLACK, "1/2-1/2", ChessGame.DRAW_REWARD, ChessGame.DRAW_REWARD),
            (BLACK_PLAYER, chess.WHITE, "1/2-1/2", ChessGame.DRAW_REWARD, -ChessGame.DRAW_REWARD),
            (BLACK_PLAYER, chess.BLACK, "1/2-1/2", ChessGame.DRAW_REWARD, -ChessGame.DRAW_REWARD),
            # fmt: on
        ],
    )
    def test_getGameEnded(
        self,
        chess_game: ChessGame,
        chess_board: chess.Board,
        player: PlayerID,
        board_turn: bool,
        board_result: str,
        ended_return: float,
        winner: float,
    ) -> None:
        with (
            patch.object(chess_board, "result", return_value=board_result),
            patch.object(chess_board, "turn", board_turn),
        ):
            assert (
                chess_game.getGameEnded(board=chess_board, player=player)
                == ended_return
            )
            assert (
                chess_game.getGameEnded(board=chess_board, player=player) * player
                == winner
            )

    @patch.object(chess.Board, "apply_mirror")
    def test_getCanonicalForm(
        self,
        mock_apply_mirror: MagicMock,
        chess_game: ChessGame,
        chess_board: chess.Board,
    ) -> None:
        with patch.object(chess_board, "turn", chess.WHITE):
            assert isinstance(
                chess_game.getCanonicalForm(chess_board, WHITE_PLAYER), chess.Board
            )
            assert isinstance(
                chess_game.getCanonicalForm(chess_board, BLACK_PLAYER), chess.Board
            )
        mock_apply_mirror.assert_not_called()

        with patch.object(chess_board, "turn", chess.BLACK):
            with pytest.raises(NotImplementedError, match="(?i)unreachable"):
                chess_game.getCanonicalForm(chess_board, WHITE_PLAYER)
            mock_apply_mirror.assert_not_called()
            assert isinstance(
                chess_game.getCanonicalForm(chess_board, BLACK_PLAYER), chess.Board
            )
            mock_apply_mirror.assert_called_once_with()

    @pytest.mark.parametrize(
        ("white_elo", "black_elo", "comparison"), [(1400, 2600, lt), (2600, 1400, gt)]
    )
    def test_full_game(
        self,
        chess_game: ChessGame,
        white_elo: int,
        black_elo: int,
        comparison: "Callable[[int, int], bool]",
    ) -> None:
        white_player = StockfishChessPlayer(engine_elo=white_elo)
        black_player = StockfishChessPlayer(BLACK_PLAYER, engine_elo=black_elo)
        arena = Arena(
            white_player,
            black_player,
            chess_game,
            display=partial(chess_game.display, verbosity=2),
        )
        n_p1_wins, n_p2_wins, _ = arena.playGames(4, verbose=True)
        assert comparison(n_p1_wins, n_p2_wins)


class CoachArgs(MCTSArgs):
    """Data structure to configure the Coach class."""

    # Number of training iterations
    numIters: int = 1000
    # Number of self-play games (episodes) per training iteration
    numEps: int = 100
    # Number of iterations to pass before increasing MCTS temp by 1
    tempThreshold: int = 15
    # Threshold win percentage of arena games to accept a new neural network
    updateThreshold: float = 0.6
    # Number of arena games to assess neural network for acceptance
    arenaCompare: int = 40
    # Number of game examples to train the neural networks
    maxlenOfQueue: int = 200_000
    # Folder name to save checkpoints
    checkpoint: str = "checkpoints"
    # Set True to load in the model weights from checkpoint and training
    # examples from the load_folder_file
    load_model: bool = False
    # Two-tuple of folder and filename where training examples are housed
    load_folder_file: tuple[str, str] = "models", "best.pt.tar"
    # Max amount of training examples to keep in the history, dropping the
    # oldest example beyond that before adding a new one (like a FIFO queue)
    numItersForTrainExamplesHistory: int = 20


class TestNNet:
    @pytest.mark.parametrize("coach_args", [CoachArgs()])
    def test_coach(self, chess_game: ChessGame, coach_args: CoachArgs) -> None:
        coach = Coach(chess_game, NNetWrapper(chess_game), coach_args)
        coach.learn()

    def test_full_game(self, chess_game: ChessGame, mcts_args: MCTSArgs) -> None:
        az_player = AlphaZeroChessPlayer(
            chess_game, player_id=WHITE_PLAYER, mcts_args=mcts_args
        )
        black_player = StockfishChessPlayer(BLACK_PLAYER, engine_elo=1400)
        arena = Arena(
            player1=az_player,
            player2=black_player,
            game=chess_game,
            display=partial(chess_game.display, verbosity=2),
        )
        n_p1_wins, n_p2_wins, n_ties = arena.playGames(2, verbose=True)
        # TODO: add assertions


class TestChessUtils:
    @pytest.mark.parametrize(
        ("p1_elo", "p2_elo", "winner", "k", "expected"),
        [
            (2400, 2000, 1, get_k_factor(2400, 2000), (2403, 1997)),
            (2400, 2000, -1, ICC_K_FACTOR, (2371, 2029)),
        ],
    )
    def test_update_elo(
        self,
        p1_elo: int,
        p2_elo: int,
        winner: Literal[-1, 0, 1],
        k: int,
        expected,
    ) -> None:
        assert update_elo(p1_elo, p2_elo, winner, k) == expected
