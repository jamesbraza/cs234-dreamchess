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
    Embed the input boards into a numpy array.

    Args:
        boards: Variable amount of Boards to embed.

    Returns:
        Embedded representation of the board of shape (B, 6, 8, 8), where B is
            the batch size (number passed in), white is 1, black is -1, no
            piece is 0, and players are pawn (0), knight (1), bishop (2),
            rook (3), queen (4), and king (5).
    """
    batch_embedding = np.zeros((len(boards), *EMBEDDING_SHAPE), dtype=int)
    for i, board in enumerate(boards):
        for sq, pc in board.piece_map().items():
            bxyz = i, pc.piece_type - 1, chess.square_rank(sq), chess.square_file(sq)
            batch_embedding[bxyz] = 1 if pc.color else -1
    return batch_embedding


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


class ResidualBlock(nn.Module):
    """Basic residual block based on two Conv3d with BatchNorm3ds."""

    def __init__(self, in_channels: int, out_channels: int, **conv_kwargs):
        super().__init__()
        conv_kwargs = {"kernel_size": 3, "padding": 1} | conv_kwargs
        self.non_residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **conv_kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, **conv_kwargs),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x + self.non_residual(x))


class NNet(nn.Module):
    """Neural network takes in a board embedding to predict policy logits and value."""

    def __init__(self, game: ChessGame, num_channels: int = 64, dropout_p: float = 0.8):
        """
        Initialize.

        Args:
            game: Game, to extract action sizes and confirm board size.
            num_channels: Number of channels for beginning Conv3d's.
            dropout_p: Dropout percentage.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, num_channels, 3, padding="same"),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
            ResidualBlock(num_channels, num_channels),
            ResidualBlock(num_channels, num_channels),
            nn.Conv3d(num_channels, num_channels, 3),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(),
        )
        # NOTE: confirm this matches, as otherwise the fully-connected layers'
        # input shape would not be correct
        assert game.getBoardSize() == EMBEDDING_SHAPE[1:]
        self._fc_layers_in_shape = num_channels * math.prod(
            conv_conversion(EMBEDDING_SHAPE, 3)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._fc_layers_in_shape, game.getActionSize()),
            nn.BatchNorm1d(game.getActionSize()),
            nn.Dropout(dropout_p),  # Apply before ReLU, for computational efficiency
            nn.ReLU(),
        )
        # NOTE: leave pi as raw scores (don't apply Softmax) to directly use
        # nn.functional.cross_entropy
        self.policy_head = nn.Linear(game.getActionSize(), game.getActionSize())
        self.value_head = nn.Sequential(
            nn.Linear(game.getActionSize(), 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_p),  # Apply before ReLU, for computational efficiency
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    @classmethod
    def embed(cls, *boards: Board) -> torch.Tensor:
        """Embed the input boards into a Tensor for the forward pass."""
        # FloatTensor is needed for gradient computations
        return torch.FloatTensor(embed(*boards))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass on embedded board batch, getting policy scores and value.

        Args:
            boards: Embedded canonical board batch of shape (B, D, X, Y).

        Returns:
            Tuple of policy scores of shape (B, |A|), value in [-1, 1] of shape (B, 1).
        """
        s = boards.unsqueeze(1)  # B, 1, D, X, Y
        s = self.conv_layers(s)  # B, C, D - 2 * 2, X - 2 * 2, Y - 2 * 2
        s = self.fc_layers(s)  # B, |A|
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
        pred_pis, pred_vs = self.nnet(self.nnet.embed(*boards))
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
        epochs: int = 20,
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
                "loss/pi", {"train": train_losses_pi.avg, "val": loss_pi}, epoch
            )
            writer.add_scalars(
                "loss/V", {"train": train_losses_v.avg, "val": loss_v}, epoch
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
            pi, v = self.nnet(self.nnet.embed(board))
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
