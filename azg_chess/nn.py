from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
from azg.NeuralNet import NeuralNet
from torch import nn

from azg_chess.game import BOARD_DIMENSIONS, NUM_PIECES

if TYPE_CHECKING:
    import numpy.typing as npt

    from azg_chess.game import Board, ChessGame, Policy

# Match nn.Conv3d shape of D, X, Y
EMBEDDING_SHAPE: tuple[int, int, int] = NUM_PIECES, *BOARD_DIMENSIONS


def embed_board(board: Board, add_batch_dim: bool = False) -> npt.NDArray[int]:
    """
    Embed the board for input to a neural network.

    Args:
        board: Board to embed.
        add_batch_dim: Set True to inject an axis up front for batching.

    Returns:
        Embedded representation of the board of shape (8, 8, 6), where white
            is 1, black is -1, no piece is 0, and layers are pawns (0),
            knight (1), bishop (2), rook (3), queen (4), and king (5).
    """
    embedding = np.zeros(EMBEDDING_SHAPE, dtype=int)
    for sq, pc in board.piece_map().items():
        xyz = pc.piece_type - 1, chess.square_rank(sq), chess.square_file(sq)
        embedding[xyz] = 1 if pc.color else -1
    return embedding if not add_batch_dim else embedding[np.newaxis, :]


def conv3d_calc(
    d_h_w_in: tuple[int, int, int],
    kernel_size: int,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
) -> tuple[int, int, int]:
    """Perform a Conv3d calculation matching nn.Conv3D's defaults."""

    def to_3_tuple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, int):
            value = (value,) * 3
        return value

    k = to_3_tuple(kernel_size)
    p = to_3_tuple(padding)
    dil = to_3_tuple(dilation)
    s = to_3_tuple(stride)
    return tuple(
        int((d_h_w_in[i] + 2 * p[i] - dil[i] * (k[i] - 1) - 1) / s[i] + 1)
        for i in range(3)
    )


class NNet(nn.Module):
    KERNEL_SIZE = 3  # Square

    def __init__(self, game: ChessGame, num_channels: int = 64, dropout_p: float = 0.5):
        """
        Initialize.

        Args:
            game: Game, to extract board dimensions and action sizes.
            num_channels: Number of channels for beginning Conv3d's.
            dropout_p: Dropout percentage.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, num_channels, self.KERNEL_SIZE, padding="same"),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
            nn.Conv3d(num_channels, num_channels, self.KERNEL_SIZE, padding="same"),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
            nn.Conv3d(num_channels, num_channels, self.KERNEL_SIZE),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
            nn.Conv3d(num_channels, num_channels, self.KERNEL_SIZE),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
        )
        # NOTE: this matches the above sequential conv's hyperparameters
        assert game.getBoardSize() == EMBEDDING_SHAPE[1:]
        conv_out_shape = conv3d_calc(
            conv3d_calc(EMBEDDING_SHAPE, self.KERNEL_SIZE), self.KERNEL_SIZE
        )
        self._fc_layers_shape = num_channels * math.prod(conv_out_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(self._fc_layers_shape, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_p),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(512, game.getActionSize()), nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(nn.Linear(512, 1), nn.Tanh())

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass given an embedded board batch of shape (B, D, X, Y)."""
        s = board.unsqueeze(1)  # B, 1, D, X, Y
        s = self.conv_layers(s)  # B, C, D - 2 * 2, X - 2 * 2, Y - 2 * 2
        s = s.reshape(-1, self._fc_layers_shape)  # B, C * (D - 4) * (X - 4) * (Y - 4)
        s = self.fc_layers(s)  # B, 512
        pi = self.policy_head(s)  # B, |A|
        v = self.value_head(s)  # B, 1
        return pi, v


class NNetWrapper(NeuralNet):
    """Neural network adaptation for chess."""

    def __init__(self, game: ChessGame, **nnet_kwargs):
        super().__init__(game)
        self.nnet = NNet(game, **nnet_kwargs)

    def train(
        self,
        examples: list[tuple[Board, Policy, int]],
        epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        optimizer = torch.optim.Adam(self.nnet.parameters())
        self.nnet.train()

        # TODO

    def predict(self, board: Board) -> tuple[Policy, float]:
        """
        Run an inference on the board.

        Args:
            board: Canonicalized board, don't mutate.

        Returns:
            Two-tuple of policy logits, expected value.
        """
        # FloatTensor is needed for gradient
        batch_embedding = torch.FloatTensor(embed_board(board, add_batch_dim=True))
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch_embedding)
        hi = pi[0].numpy(), float(v[0])  # Unbatch
        assert all((hi[0] >= 0) & (hi[0] <= 1)), f"Negative logprob in {hi[0]}."
        return hi

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """Save the NN's parameters to the folder/filename."""
        os.makedirs(folder, exist_ok=True)  # NOTE: recursive
        torch.save(
            {"model_state_dict": self.nnet.state_dict()},
            f=os.path.join(folder, filename),
        )

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load in NN's parameters from the folder/filename."""
        checkpoint = torch.load(f=os.path.join(folder, filename))
        self.nnet.load_state_dict(checkpoint["model_state_dict"])
