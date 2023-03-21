# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 01:06:22 2023

@author: h_min
"""

import chess
import torch
import numpy as np

class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def serialize(self):
        assert self.board.is_valid()

        bstate = np.zeros(64,np.uint8)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                # print(i, pp.symbol())
                bstate[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                             "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[pp.symbol()]

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert bstate[0] == 4
            bstate[0] = 7

        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert bstate[7] == 4
            bstate[7] = 7

        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert bstate[56] == 8+4
            bstate[56] = 8+7

        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert bstate[63] == 8+4
            bstate[63] = 8+7

        if self.board.ep_square is not None:
            assert bstate[self.board.ep_square] == 0
            bstate[self.board.ep_square] = 8

        # Binary representation
        bstate = bstate.reshape(8, 8)
        state = np.zeros((5, 8, 8), np.uint8)

        # 0-3 column to binary
        state[0] = (bstate >> 3) & 1
        state[1] = (bstate >> 2) & 1
        state[2] = (bstate >> 1) & 1
        state[3] = (bstate >> 0) & 1

        # 4th column is who's turn it is
        state[4] = (self.board.turn*1.0)
        return state

    def edges(self):
        return list(self.board.legal_moves)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # 1x64
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        # value output
        return F.tanh(x)