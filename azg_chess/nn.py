from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
from azg.NeuralNet import NeuralNet
from azg.utils import AverageMeter
from torch import nn
from tqdm import tqdm

from azg_chess.game import BOARD_DIMENSIONS, NUM_PIECES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from azg_chess.game import Board, ChessGame, Policy

# Match nn.Conv3d shape of D, X, Y
EMBEDDING_SHAPE: tuple[int, int, int] = NUM_PIECES, *BOARD_DIMENSIONS


def embed(*boards: Board) -> npt.NDArray[int]:
    """
    Embed the boards for input to a neural network.

    Args:
        boards: Variable amount of Boards to embed.

    Returns:
        Embedded representation of the board of shape (B, 8, 8, 6), where B is
            the batch size (number passed in), white is 1, black is -1, no
            piece is 0, and layers are pawns (0), knight (1), bishop (2),
            rook (3), queen (4), and king (5).
    """
    embedding = np.zeros((len(boards), *EMBEDDING_SHAPE), dtype=int)
    for i, board in enumerate(boards):
        for sq, pc in board.piece_map().items():
            bxyz = i, pc.piece_type - 1, chess.square_rank(sq), chess.square_file(sq)
            embedding[bxyz] = 1 if pc.color else -1
    return embedding


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
    """Neural network takes in a board embedding to predict policy logits and value."""

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
    """Neural network wrapper adaptation for chess."""

    def __init__(self, game: ChessGame, **nnet_kwargs):
        super().__init__(game)
        self.nnet = NNet(game, **nnet_kwargs)

    def train(
        self,
        examples: Sequence[tuple[Board, Policy, int]],
        epochs: int = 10,
        batch_size: int = 64,
        l2_coefficient: float = 1e-4,  # TODO: verify this
    ) -> None:
        """
        Train on a bunch of examples.

        Args:
            examples: Pre-shuffled sequence of examples, where each is a tuple
                of canonical board, policy logits, player won (1) or lost (-1).
            epochs: Number of epochs.
            batch_size: Batch size.
            l2_coefficient: Coefficient for L2 regularization.
                To defeat L2 regularization set to 0.0.
                Default was chosen to match default of
                https://jonathan-laurent.github.io/AlphaZero.jl/dev/reference/params/.
        """
        optimizer = torch.optim.Adam(
            self.nnet.parameters(), weight_decay=l2_coefficient
        )
        self.nnet.train()

        for epoch in range(1, epochs + 1):
            pi_losses, v_losses = AverageMeter(), AverageMeter()

            t = tqdm(
                range(math.ceil(len(examples) / batch_size)), desc=f"Epoch {epoch}"
            )
            for i in t:
                boards, pis, vs = list(
                    zip(*examples[i * batch_size : (i + 1) * batch_size])
                )
                # FloatTensor is needed for gradient
                out_pi, out_v = self.nnet(torch.FloatTensor(embed(*boards)))
                # TODO: move to torch functions for this
                target_pis = torch.FloatTensor(pis)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = nn.functional.mse_loss(torch.FloatTensor(vs), out_v.reshape(-1))
                pi_losses.update(val=l_pi.item(), n=len(boards))
                v_losses.update(val=l_v.item(), n=len(boards))
                # TODO: check how this looks
                t.set_postfix(policy_loss=pi_losses, value_loss=v_losses)

                optimizer.zero_grad()
                torch.Tensor.backward(l_pi + l_v)  # Backprop on total loss
                optimizer.step()

    # TODO: remove
    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def predict(self, board: Board) -> tuple[Policy, float]:
        """
        Run an inference on the board.

        Args:
            board: Canonicalized board, don't mutate.

        Returns:
            Two-tuple of policy logits, expected value.
        """
        self.nnet.eval()
        with torch.no_grad():
            # FloatTensor is needed for gradient
            pi, v = self.nnet(torch.FloatTensor(embed(board)))
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
