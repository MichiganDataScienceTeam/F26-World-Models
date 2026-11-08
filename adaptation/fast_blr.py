import numpy as np


class FastBLR:
    """
    Bayesian linear regression stored as running sums (X^T X, X^T y, y^T y).
    labels: one per coefficient; equal labels share a parameter, -1 fixes it at 0.
    nu: Student-t noise; a batch with a large error counts less. eta: each row counts eta.
    """

    def __init__(self, n_features, prior_std=1., noise_std=1., nu=None, eta=1.):
        self.n_features, self.prior_std, self.noise_std, self.nu, self.eta = n_features, prior_std, noise_std, nu, eta
        self.G, self.g = np.zeros((n_features, n_features)), np.zeros(n_features)
        self.yy = self.ldn = 0.

    def partial_fit(self, X, y, weight=1., error=None):
        """add rows X, y; error sets the batch's weight when nu is given"""
        X, y = np.atleast_2d(np.asarray(X, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        precision = self.eta * weight / self.noise_std**2
        if self.nu is not None:
            error = y - X @ self.mean() if error is None else np.asarray(error, dtype=float)
            precision *= (self.nu + len(y)) / (self.nu + error @ error / self.noise_std**2)
        self.G = self.G + precision * X.T @ X
        self.g = self.g + precision * X.T @ y
        self.yy += precision * y @ y
        self.ldn += .5 * len(y) * np.log(precision / (2 * np.pi))
        return self

    def __add__(self, other):
        out = FastBLR(self.n_features, self.prior_std, self.noise_std, self.nu, self.eta)
        out.G, out.g, out.yy, out.ldn = self.G + other.G, self.g + other.g, self.yy + other.yy, self.ldn + other.ldn
        return out

    def _fit(self, labels):
        """T maps parameters to coefficients; returns T, posterior precision, right-hand side"""
        if labels is None:
            T = np.eye(self.n_features)
        else:
            labels = np.ravel(labels)
            k = labels.max() + 1
            if labels.size != self.n_features or set(labels[labels >= 0]) != set(range(k)):
                raise ValueError(f"labels: one per coefficient ({self.n_features}), using 0, 1, ..., k-1 and -1")
            T = (labels[:, None] == np.arange(k)).astype(float)
        return T, np.eye(T.shape[1]) / self.prior_std**2 + T.T @ self.G @ T, T.T @ self.g

    def mean(self, labels=None):
        """posterior mean of the coefficients"""
        T, precision, rhs = self._fit(labels)
        return T @ np.linalg.solve(precision, rhs)

    def cov(self, labels=None):
        """posterior covariance of the coefficients"""
        T, precision, _ = self._fit(labels)
        return T @ np.linalg.inv(precision) @ T.T

    def log_evidence(self, labels=None):
        """log p(y | X, model)"""
        _, precision, rhs = self._fit(labels)
        out = self.ldn - .5 * self.yy
        if len(rhs) == 0:
            return out
        C = np.linalg.cholesky(precision)
        v = np.linalg.solve(C, rhs)
        return out + .5 * v @ v - np.log(np.diag(C)).sum() - len(rhs) * np.log(self.prior_std)
