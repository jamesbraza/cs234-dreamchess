# cs234-dreamchess

Stanford CS234 Reinforcement Learning: Class Project

## Datasets

We used the following datasets from [Kaggle](https://www.kaggle.com/):

- [3.5 Million Chess Games][1]: a text file export of [ChessDB][2]

Here's how to easily download the datasets with the [Kaggle API][3]:

```console
kaggle datasets download -p data/chess-games --unzip milesh1/35-million-chess-games
```

## Developers

This project was developed using Python 3.10.

### Getting Started

Here is how to create a virtual environment:

```console
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

#### Code QA Tooling

Here is how to also install code QA tooling:

```console
python -m pip install -r requirements-qa.txt
pre-commit install
```

[1]: https://www.kaggle.com/datasets/milesh1/35-million-chess-games
[2]: https://chessdb.sourceforge.net/
[3]: https://github.com/Kaggle/kaggle-api
