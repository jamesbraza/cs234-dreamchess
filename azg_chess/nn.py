from __future__ import annotations

import math
import os
from functools import partial
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
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from azg_chess.game import Board, ChessGame, Policy


def discern_gpu_mac_cpu_device() -> torch.device:
    """Discern which torch.device to use in cuda (1st), mps (2nd), cpu (3rd)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # As of 3/18/2023, aten::nonzero was not supported on macOS 12.6 with
        # torch 2.0.0, and torch 1.3.1 didn't support aten::nonzero at all
        # SEE: https://github.com/pytorch/pytorch/issues/77764
        # SEE: https://github.com/pytorch/pytorch/commit/38de981e160732bce5d90bace0b40d63dba31bf1
        pass
    return torch.device("cpu")  # Fallback


# Match nn.Conv3d shape of D, X, Y
SIGNED_EMBEDDING_SHAPE: tuple[int, int, int] = NUM_PIECES, *BOARD_DIMENSIONS
UNSIGNED_EMBEDDING_SHAPE: tuple[int, int, int] = 2 * NUM_PIECES, *BOARD_DIMENSIONS


def embed(*boards: Board, signed: bool = True) -> npt.NDArray[int]:
    """
    Embed the input boards into a numpy array.

    Args:
        boards: Variable amount of Boards to embed.
        signed: Set True to embed white as 1 and black as -1 on one board.
            Set False to embed both as 1 on separate boards.

    Returns:
        Embedded representation of the board of shape (B, 6, 8, 8), where B is
            the batch size (number passed in), white is 1, black is -1, no
            piece is 0, and players are pawn (0), knight (1), bishop (2),
            rook (3), queen (4), and king (5).
    """
    shape = SIGNED_EMBEDDING_SHAPE if signed else UNSIGNED_EMBEDDING_SHAPE
    batch_embedding = np.zeros((len(boards), *shape), dtype=int)
    for i, board in enumerate(boards):
        for sq, pc in board.piece_map().items():
            slice_index = pc.piece_type - 1
            if not signed and pc.color == chess.BLACK:
                slice_index += NUM_PIECES
            bxyz = i, slice_index, chess.square_rank(sq), chess.square_file(sq)
            batch_embedding[bxyz] = -1 if signed and pc.color == chess.BLACK else 1
    return batch_embedding


signed_embed_pair = embed, SIGNED_EMBEDDING_SHAPE
unsigned_embed_pair = partial(embed, signed=False), UNSIGNED_EMBEDDING_SHAPE


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
    """Basic residual block based on two Conv2d with BatchNorm2ds."""

    DEFAULT_CONV_KWARGS = {"kernel_size": 3, "padding": 1, "bias": False, "stride": 1}

    def __init__(self, in_channels: int, out_channels: int, **conv_kwargs):
        super().__init__()
        conv_kwargs = self.DEFAULT_CONV_KWARGS | conv_kwargs
        self.non_residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x + self.non_residual(x))


class NNet(nn.Module):
    """Neural network takes in a board embedding to predict policy logits and value."""

    def __init__(
        self,
        game: ChessGame,
        residual_channels: int = 256,
        num_residual_layers: int = 19,
        policy_channels: int = 96,
        value_hidden_units: int = 64,
        dropout_p: float = 0.2,
        embed_func_shape: tuple[
            Callable[[Board, ...], npt.NDArray[int]], tuple[int, int, int]
        ] = signed_embed_pair,
    ):
        """
        Initialize.

        Args:
            game: Game, to extract action sizes.
            residual_channels: Number of channels in the residual layers,
                referred to as C sometimes below.
            num_residual_layers: Number of residual layers.
            policy_channels: Number of channels in the policy network.
            value_hidden_units: Number of hidden units in the value network.
            dropout_p: Probability of zeroing inside all Dropout layers.
            embed_func_shape: Two tuple of embedding function and its per-board
                output shape.
        """
        super().__init__()
        self._embed_func, shape = embed_func_shape
        self.residual_tower = nn.Sequential(
            nn.Conv2d(shape[0], residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels),
            nn.ReLU(),
            *(
                ResidualBlock(residual_channels, residual_channels)
                for _ in range(num_residual_layers)
            ),
        )

        # NOTE: confirm this matches, as otherwise the fully-connected layers'
        # input shape would not be larger than the action size
        policy_hidden_channels = policy_channels * math.prod(shape[1:])
        assert policy_hidden_channels >= game.getActionSize()
        # NOTE: leave pi as raw scores (don't apply Softmax) to directly use
        # nn.functional.cross_entropy
        self.policy_head = nn.Sequential(
            nn.Conv2d(residual_channels, policy_channels, kernel_size=1),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_hidden_channels, game.getActionSize()),
            nn.Dropout(dropout_p),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(residual_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * math.prod(shape[1:]), value_hidden_units),
            nn.Dropout(dropout_p),  # Apply before ReLU, for computational efficiency
            nn.ReLU(),
            nn.Linear(value_hidden_units, 1),
            nn.Tanh(),
        )

    def embed(self, *boards: Board, device: torch.device | None = None) -> torch.Tensor:
        """Embed the input boards into a Tensor for the forward pass."""
        return torch.as_tensor(
            self._embed_func(*boards), dtype=torch.float32, device=device
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass on embedded board batch, getting policy scores and value.

        Args:
            boards: Embedded canonical board batch of shape (B, D, X, Y).

        Returns:
            Tuple of policy scores of shape (B, |A|), value in [-1, 1] of shape (B, 1).
        """
        # NOTE: this sets requires_grad = True, if not already set
        s = self.residual_tower(boards)  # B, C, X, Y
        return self.policy_head(s), self.value_head(s)  # (B, |A|), (B, 1)


class NNetWrapper(NeuralNet):
    """Neural network wrapper adaptation for chess."""

    def __init__(
        self,
        game: ChessGame,
        *,
        device: torch.device | bool | None = True,
        **nnet_kwargs,
    ):
        """
        Initialize.

        NOTE: Coach.py instantiates a pnet via NeuralNet.__class__ and
        positionally passes it just the game. So:
        1. Required that the signature's only positional arg be a game.
        2. All other arguments required should be defaulted (hence device's
           default to True). This may force a subclassing, when composition
           would normally be used.

        Args:
            game: Chess game.
            device: Device to cast Tensors internally.
                Set True (default) to discern the appropriate torch.device.
                Set to a torch.device directly.
                Set False or None to not specify a torch.device.
            **nnet_kwargs: Keyword arguments to pass onto the internal NNet.
        """
        super().__init__(game)
        self.nnet = NNet(game, **nnet_kwargs)
        if isinstance(device, bool) and device:
            device = discern_gpu_mac_cpu_device()
        if isinstance(device, torch.device):
            self._device: torch.device | None = device
            self.nnet.to(device=self._device)
        else:  # None or False
            self._device = None

    def _calculate_losses(
        self,
        boards: Sequence[Board],
        true_pis: Sequence[Policy],
        true_vs: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate pi, V, and total loss given example data."""
        pred_pis, pred_vs = self.nnet(self.nnet.embed(*boards, device=self._device))
        # Why cast to np.array? SEE: https://github.com/pytorch/pytorch/issues/13918
        true_pis, true_vs = (
            torch.as_tensor(np.array(x, dtype=np.float32), device=self._device)
            for x in [true_pis, true_vs]
        )
        loss_pi = nn.functional.cross_entropy(input=pred_pis, target=true_pis)
        loss_v = nn.functional.mse_loss(input=pred_vs.reshape(-1), target=true_vs)
        return loss_pi, loss_v, loss_pi + loss_v

    def train(  # pylint: disable=too-many-locals
        self,
        examples: Sequence[tuple[Board, Policy, float]],
        epochs: int = 20,
        batch_size: int = 64,
        l2_coefficient: float = 1e-4,
    ) -> None:
        """
        Train the internal NNet on a bunch of examples.

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
            pi, v = self.nnet(self.nnet.embed(board, device=self._device))
        # NOTE: [0] is to unbatch
        return nn.functional.softmax(pi, dim=1)[0].numpy(), float(v[0])

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """Save the internal NNet's parameters to the folder/filename."""
        os.makedirs(folder, exist_ok=True)  # NOTE: recursive
        torch.save(
            {"model_state_dict": self.nnet.state_dict()},
            f=os.path.join(folder, filename),
        )

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load in internal NNet's parameters from the folder/filename."""
        checkpoint = torch.load(f=os.path.join(folder, filename))
        self.nnet.load_state_dict(checkpoint["model_state_dict"])


class UnsignedNNetWrapper(NNetWrapper):
    """Wrapper that specifies the unsigned embedding function by default."""

    def __init__(self, game: ChessGame, **kwargs):
        # SEE: parent class's __init__ docstring for why this subclass exists to
        # specify the unsigned embedding statically (instead of via composition)
        super().__init__(game, **({"embed_func_shape": unsigned_embed_pair} | kwargs))
