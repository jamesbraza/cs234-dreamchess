from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
import torch.utils.tensorboard
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
            piece is 0, and players are pawn (0), knight (1), bishop (2),
            rook (3), queen (4), and king (5).
    """
    embedding = np.zeros((len(boards), *EMBEDDING_SHAPE), dtype=int)
    for i, board in enumerate(boards):
        for sq, pc in board.piece_map().items():
            bxyz = i, pc.piece_type - 1, chess.square_rank(sq), chess.square_file(sq)
            embedding[bxyz] = 1 if pc.color else -1
    return embedding


def conv_conversion(
    in_shape: tuple[int, ...],
    kernel_size: int | tuple[int, ...],
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    stride: int | tuple[int, ...] = 1,
) -> tuple[int, ...]:
    """Perform a Conv layer calculation matching nn.Conv's defaults."""

    def to_tuple(value: int | tuple[int, ...]) -> tuple[int, ...]:
        return (value,) * len(in_shape) if isinstance(value, int) else value

    k, p = to_tuple(kernel_size), to_tuple(padding)
    dil, s = to_tuple(dilation), to_tuple(stride)
    return tuple(
        int((in_shape[i] + 2 * p[i] - dil[i] * (k[i] - 1) - 1) / s[i] + 1)
        for i in range(len(in_shape))
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
        conv_out_shape = conv_conversion(
            conv_conversion(EMBEDDING_SHAPE, self.KERNEL_SIZE),
            self.KERNEL_SIZE,
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
        # NOTE: leave as raw scores to directly use nn.functional.cross_entropy
        self.policy_head = nn.Linear(512, game.getActionSize())
        self.value_head = nn.Sequential(nn.Linear(512, 1), nn.Tanh())

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass on embedded board batch, getting policy scores and value.

        Args:
            board: Embedded canonical board batch of shape (B, D, X, Y).

        Returns:
            Tuple of policy scores of shape (B, |A|), value in [-1, 1] of shape (B, 1).
        """
        s = board.unsqueeze(1)  # B, 1, D, X, Y
        s = self.conv_layers(s)  # B, C, D - 2 * 2, X - 2 * 2, Y - 2 * 2
        s = s.reshape(-1, self._fc_layers_shape)  # B, C * (D - 4) * (X - 4) * (Y - 4)
        s = self.fc_layers(s)  # B, 512
        return self.policy_head(s), self.value_head(s)  # (B, |A|), (B, 1)


class NNetWrapper(NeuralNet):
    """Neural network wrapper adaptation for chess."""

    def __init__(self, game: ChessGame, **nnet_kwargs):
        super().__init__(game)
        self.nnet = NNet(game, **nnet_kwargs)

    def _calculate_losses(
        self,
        boards: Sequence[Board],
        true_pis: Sequence[Policy],
        true_vs: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate pi, V, and total loss given example data."""
        # FloatTensor is needed for gradient
        pred_pis, pred_vs = self.nnet(torch.FloatTensor(embed(*boards)))
        loss_pi = nn.functional.cross_entropy(
            input=pred_pis, target=torch.FloatTensor(true_pis)
        )
        loss_v = nn.functional.mse_loss(
            input=pred_vs.reshape(-1),
            target=torch.FloatTensor(true_vs),
        )
        return loss_pi, loss_v, loss_pi + loss_v

    def train(  # pylint: disable=too-many-locals
        self,
        examples: Sequence[tuple[Board, Policy, float]],
        epochs: int = 10,
        batch_size: int = 64,
        l2_coefficient: float = 1e-4,
    ) -> None:
        """
        Train on a bunch of examples.

        Args:
            examples: Pre-shuffled sequence of examples, where each is a tuple
                of canonical board, policy logits, player won (1) or lost (-1),
                or tie (+/- 1e-5).
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
        writer = torch.utils.tensorboard.SummaryWriter()
        self.nnet.train()

        for epoch in range(epochs):
            train_losses_pi, train_losses_v = AverageMeter(), AverageMeter()
            # NOTE: give partial batch contents to validation set
            num_batches = int(len(examples) / batch_size)
            t = tqdm(range(num_batches - 1), desc=f"Epoch {epoch + 1}/{epochs}")
            for i in t:
                loss_pi, loss_v, loss_total = self._calculate_losses(
                    *list(zip(*examples[i * batch_size : (i + 1) * batch_size]))
                )
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                train_losses_pi.update(val=loss_pi.item(), n=batch_size)
                train_losses_v.update(val=loss_v.item(), n=batch_size)
                t.set_postfix(pi_loss=train_losses_pi, v_loss=train_losses_v)

            # Use remaining examples as a validation set
            loss_pi, loss_v, _ = self._calculate_losses(
                *list(zip(*examples[(num_batches - 1) * batch_size :]))
            )
            writer.add_scalars(
                "loss",
                {
                    "train_pi": train_losses_pi.avg,
                    "train_V": train_losses_v.avg,
                    "val_pi": loss_pi,
                    "val_V": loss_v,
                },
                epoch,
            )

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
        # NOTE: [0] is to unbatch
        return nn.functional.softmax(pi, dim=1)[0].numpy(), float(v[0])

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
