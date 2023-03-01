# cs234-dreamchess

Stanford CS234 Reinforcement Learning: Class Project

## Datasets

We used the following dataset from [Kaggle](https://www.kaggle.com/):
[3.5 Million Chess Games][1], a text file export of [ChessDB][2]
([docs](https://chess-research-project.readthedocs.io/en/latest/)).

Here's how to easily download the datasets with the [Kaggle API][3]:

```console
kaggle datasets download -p data/chess-games --unzip milesh1/35-million-chess-games
```

### Interpreting Entries

There is a slight learning curve to understand the dataset.

- Each row of the `all_with_filtered_anotations_since1998.txt` file
  contains the data for one chess game.
- There is a header (game information) and body (moves), separated by `###`.
- The header contains information like game date, ELO of each player,
  and if an initial position was specified (e.g. Fischer Random Chess).

A chess board is an 8 x 8 grid where:

- Rows ("rank"): numbered 1 (bottom row) to 8 (top row).
- Cols ("file"): lettered a (left column) to h (right column).

Moves are written in [Portable Game Notation][8],
which uses [Algebraic Notation][9]
(also known as standard algebraic notation (SAN) or standard notation)
to describe each move as [movetext][7].

Stated with regex-like notation,
movetext is governed by `ab.(c?d?e|O-O|O-O-O|e=f)`:

1. `a`: player, either white `W` or black `B`.
1. `b`: one-indexed turn number.
1. `.`: separator.

The rest could be one of several options:

- Normal move:
  1. `c?`: piece, either pawn (no letter, `P` in other contexts), king `K`,
     queen `Q`, rook `R`, bishop `B`, or knight `N` (`S` in other contexts).
  1. `d?`: capture, either no capture (no letter) or capture `x`.
  1. `e`: board destination.  When 2+ pieces could have reached the destination,
     the piece's original rank (row), file (column), or both are included.
     For example, `g5` means the piece moved to `g5`,
     `df8` means the piece moved from `d8` to `f8`,
     `1a3` means the piece moved from `a1` to `a3`,
     and `h4e1` means the piece moved from `h4` to `e1`.
- Kingside castle: `O-O`.
- Queenside castle: `O-O-O`.
- Pawn promotion:
  1. `e`: board destination (see above).
  1. `=`: indicates promotion.
     - Note: 'underpromotion' is a term for promoting a pawn to a non-queen piece.
  1. `f`: exchanged piece.

And lastly if the move leads to a check `+` or checkmate `#`.

To help connect the dots, check
[this May 1783 blindfolded match's visualization][10].

## Chess Engine

We are using the open source chess engine [`stockfish`][4] ([source code][6]).
To integrate with the `stockfish` engine,
we use the Python wrapper [`stockfish`][5].

## Developers

This project was developed using Python 3.10.

### Getting Started: OS

To install OS-level dependencies on macOS:

```console
brew install stockfish
```

After, run `whereis stockfish` to find
the engine was installed at `/opt/homebrew/bin/stockfish`.

### Getting Started: Python

Here is how to create a virtual environment
and install the core project dependencies:

```console
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

#### QA Tooling

Here is how to also install code quality assurance tooling:

```console
python -m pip install -r requirements-qa.txt
pre-commit install
```

[1]: https://www.kaggle.com/datasets/milesh1/35-million-chess-games
[2]: https://chessdb.sourceforge.net/
[3]: https://github.com/Kaggle/kaggle-api
[4]: https://stockfishchess.org/
[5]: https://github.com/zhelyabuzhsky/stockfish
[6]: https://github.com/official-stockfish/Stockfish
[7]: https://en.wikipedia.org/wiki/Portable_Game_Notation#Movetext
[8]: https://en.wikipedia.org/wiki/Portable_Game_Notation
[9]: https://en.wikipedia.org/wiki/Algebraic_notation_(chess)
[10]: https://www.chessgames.com/perl/chessgame?gid=1440134
