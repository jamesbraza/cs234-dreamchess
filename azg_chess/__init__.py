"""
Chess implemented against the [Alpha Zero General repo][1]'s framework.

Useful Links:
- [AlphaZero Go paper](https://arxiv.org/abs/1712.01815)
- [AlphaZero paper][2]

[1]: https://github.com/suragnair/alpha-zero-general
[2]: https://www.deepmind.com/publications/a-general-reinforcement-
     learning-algorithm-that-masters-chess-shogi-and-go-through-self-play
"""

import os
import sys

# Enable azg submodule to be seen
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "azg"))
