"""Online(regressor, inputs, window): any scikit-learn regressor with return_std, run online."""
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

COLUMNS = {'none': [], 'time': [0], 'z': [1, 2], 'time+z': [0, 1, 2]}


class Frozen:
    """Never adapts: the pretrained model as is (predicts zero error)."""

    def reset(self, n):
        self.n = n

    def predict(self, x):
        return np.zeros(self.n), np.ones(self.n)

    def update(self, x, y):
        pass


class Online:
    """Refits on past readings each step (one copy per output) and predicts the next one.
    inputs: 'none' (constant), 'time', 'z', or 'time+z'. window: last k readings only. every: refit interval."""

    def __init__(self, regressor, inputs='time', window=None, every=1):
        self.regressor, self.columns, self.window, self.every = regressor, COLUMNS[inputs], window, every

    def reset(self, n):
        self.n, self.X, self.Y, self.fits = n, [], [], None

    def _inputs(self, x):
        return np.r_[1., np.asarray(x)[self.columns]]            # a constant column, then the chosen inputs

    def predict(self, x):
        if self.fits is None:
            return np.zeros(self.n), np.ones(self.n)
        out = [fit.predict(self._inputs(x)[None], return_std=True) for fit in self.fits]
        return np.array([m[0] for m, _ in out]), np.array([s[0] for _, s in out])

    def update(self, x, y):
        self.X.append(self._inputs(x)); self.Y.append(y)
        if len(self.Y) % self.every:
            return
        X, Y = np.array(self.X[-self.window:] if self.window else self.X), np.array(self.Y[-self.window:] if self.window else self.Y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            self.fits = [clone(self.regressor).fit(X, Y[:, j]) for j in range(self.n)]
