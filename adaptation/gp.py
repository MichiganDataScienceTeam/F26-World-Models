import numpy as np


class GaussianProcessRegression:
    """Zero-mean GP regression with an RBF kernel and independent outputs.

    covariance_[j] stores the training covariance, including noise, for output j.
    """

    def __init__(self, n_features, n_outputs=1, *, length_scale=1., prior_std=1., noise_std=.1):
        if type(n_features) is not int or type(n_outputs) is not int or min(n_features, n_outputs) < 1:
            raise ValueError("Positive integer dimensions are required")
        if not np.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_std = float(prior_std)
        self.length_scale = np.broadcast_to(length_scale, (n_features,)).astype(float).copy()
        self.noise_std = np.broadcast_to(noise_std, (n_outputs,)).astype(float).copy()
        if (not np.isfinite(self.length_scale).all() or (self.length_scale <= 0).any()
                or not np.isfinite(self.noise_std).all() or (self.noise_std <= 0).any()):
            raise ValueError("Length scales and noise scales must be finite and positive")

        self.reset()

    def reset(self):
        self.X_ = np.empty((0, self.n_features))
        self.y_ = np.empty((0, self.n_outputs))
        self.covariance_ = np.empty((self.n_outputs, 0, 0))
        self.n_samples_seen_ = 0
        return self

    def _kernel(self, X, Y):
        """Return RBF covariances between X and Y."""

        raise NotImplementedError("Implement _kernel")

    def fit(self, X, y):
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        """Append X (samples, features) and y (samples, outputs); rebuild covariance_."""

        raise NotImplementedError("Implement partial_fit")

    def predict(self, X, return_std=False, include_noise=True):
        """Return means or (means, standard deviations) for X (..., features)."""

        raise NotImplementedError("Implement predict")
