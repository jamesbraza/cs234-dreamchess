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
   - Note: in Forsyth–Edwards Notation (FEN) notation,
     used by the `chess` library,
     white pieces are designated with capital letters,
     and black pieces are designated with lowercase letters.
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

### Getting Started: AWS

I launched several AWS
Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230315's
with instance type t2.micro (free tier).
Here's how they were configured:

```console
> source activate pytorch
> python --version
Python 3.9.16
```

Whew, I almost used `conda`, that was a close call.

**Step 1**: install and configure Python 3.10:

```console
python3 --version  # 3.8.10
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-venv
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
python3 --version  # 3.10.10
```

**Step 2**: `git clone` and install requirements into a `venv`:

```console
git clone --recurse-submodules https://github.com/jamesbraza/cs234-dreamchess.git
cd cs234-dreamchess
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt --progress-bar off \
    -r requirements-qa.txt \
    torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Note the `torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`
trick installs a CPU-only version of PyTorch 1.13.1 (since this AMI has no GPU).

I didn't want to use PyTorch 2.0 since it was released this week,
and likely has some bugs.

**Step 3**: kick off your `azg_chess` script:

```console
tmux
source venv/bin/activate
python -m azg_chess.script
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
