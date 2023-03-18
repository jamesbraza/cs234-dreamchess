import collections
import math
from functools import partial
from operator import gt, lt
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

import chess
import pytest
from azg.Arena import Arena
from azg.Coach import Coach

from azg_chess.chess_utils import ICC_K_FACTOR, Elo, get_k_factor, update_elo
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
from azg_chess.players import (
    NULL_ELO,
    AlphaZeroChessPlayer,
    ChessPlayer,
    MCTSArgs,
    StockfishChessPlayer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


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
    @pytest.mark.parametrize(
        ("coach_args", "parameters_path"),
        [(CoachArgs(), ("checkpoints", "temp.pth.tar"))],
    )
    def test_coach(
        self,
        chess_game: ChessGame,
        coach_args: CoachArgs,
        parameters_path: tuple[str, str] | None,
    ) -> None:
        nnet_wrapper = NNetWrapper(chess_game)
        if parameters_path is not None:
            nnet_wrapper.load_checkpoint(*parameters_path)
        coach = Coach(chess_game, nnet_wrapper, coach_args)
        coach.learn()

    @pytest.mark.parametrize("mcts_args", [MCTSArgs()])
    def test_full_game(self, chess_game: ChessGame, mcts_args: MCTSArgs) -> None:
        az_player = AlphaZeroChessPlayer(
            chess_game, player_id=WHITE_PLAYER, mcts_args=mcts_args
        )

        win_percentage = self.play_against_stockfish(
            chess_game, az_player, StockfishChessPlayer(BLACK_PLAYER, engine_elo=1400)
        )
        assert 0.0 <= win_percentage <= 1.0

    @staticmethod
    def play_against_stockfish(
        chess_game: ChessGame,
        unknown_player: ChessPlayer,
        stockfish_player: StockfishChessPlayer,
        n_games: int = 10,
        verbose: bool = False,
        include_ties: bool = False,
    ) -> float:
        assert unknown_player.id * -1 == stockfish_player.id

        arena = Arena(
            unknown_player,
            player2=stockfish_player,
            game=chess_game,
            display=partial(chess_game.display, verbosity=2),
        )
        n_p1_wins, n_p2_wins, n_ties = arena.playGames(n_games, verbose)
        if include_ties:
            n_total = n_p1_wins + n_p2_wins + n_ties
        else:
            n_total = n_p1_wins + n_p2_wins
        return n_p1_wins / n_total

    @staticmethod
    def play_game_update_elo(
        chess_game: ChessGame,
        p1_p1elo: "Sequence[ChessPlayer, Elo]",
        p2_p2elo: "Sequence[ChessPlayer, Elo]",
        k_factor: int = ICC_K_FACTOR,
        verbose: bool = False,
    ) -> tuple[Elo, Elo]:
        (p1, p1_elo), (p2, p2_elo) = p1_p1elo, p2_p2elo
        assert p1.id == WHITE_PLAYER
        assert p2.id == BLACK_PLAYER

        arena = Arena(
            p1, p2, game=chess_game, display=partial(chess_game.display, verbosity=2)
        )
        winner_id: Literal[-1, 0, 1] = arena.playGame(verbose)
        return update_elo(p1_elo, p2_elo, winner_id, k_factor)

    @classmethod
    def discern_elo(
        cls,
        chess_game: ChessGame,
        unknown_elo_player: ChessPlayer,
        unknown_elo_assumption: int = 1350,
        stockfish_initial_elo: Elo = 1350,
        window_width: int = 10,
        desired_stability: Elo = 100,
        n_exit: int = 10,
        verbose: bool = False,
    ) -> Elo:
        """
        Discern the Elo of an unknown player.

        Args:
            chess_game: Chess game to use in the arena.
            unknown_elo_player: Player of unknown Elo.
            unknown_elo_assumption: Assumption of unknown player's Elo.
            stockfish_initial_elo: Initial Elo assumption for Stockfish player.
            window_width: Number of games in the stability sliding window.
            desired_stability: Range of values in the sliding window to
                consider Elo as having stabilized.
            n_exit: Failover max number of games if Elo doesn't stabilize.  If
                the failover threshold is hit, use the last updated Elo.
            verbose: Set True to print out game updates.

        Returns:
            Stabilized or last seen Elo of the known player.
        """
        assert unknown_elo_player.id == WHITE_PLAYER
        make_stockfish = partial(StockfishChessPlayer, player_id=BLACK_PLAYER)
        known = [
            make_stockfish(engine_elo=stockfish_initial_elo),
            stockfish_initial_elo,
        ]

        def update_known(elo: Elo) -> None:
            elo = known[0].clip_elo(elo)
            if known[1] != elo:
                known[0] = make_stockfish(engine_elo=elo)
                known[1] = elo

        unknown_elos_window = collections.deque(
            [NULL_ELO] * (window_width - 1) + [unknown_elo_assumption],
            maxlen=window_width,
        )
        for _ in range(n_exit):
            if (
                max(unknown_elos_window) - min(unknown_elos_window) <= desired_stability
                or all(
                    NULL_ELO < elo < known[0].elo_range[0]
                    for elo in unknown_elos_window
                )
                or all(elo > known[0].elo_range[1] for elo in unknown_elos_window)
            ):
                break
            updated_unknown_elo, _ = cls.play_game_update_elo(
                chess_game,
                p1_p1elo=(unknown_elo_player, unknown_elos_window[-1]),
                p2_p2elo=known,
                verbose=verbose,
            )
            unknown_elos_window.append(updated_unknown_elo)
            # Match Stockfish to the Elo of the unknown player each game
            update_known(elo=updated_unknown_elo)
        return unknown_elos_window[-1]


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
        p1_elo: Elo,
        p2_elo: Elo,
        winner: Literal[-1, 0, 1],
        k: int,
        expected: tuple[Elo, Elo],
    ) -> None:
        assert update_elo(p1_elo, p2_elo, winner, k) == expected
