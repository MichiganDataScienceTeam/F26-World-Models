import numpy as np


class BayesianLinearRegression:
    """Independent Gaussian weight posteriors for each output.

    coef_[j] and sigma_[j] hold the mean and covariance for output j.
    Include a constant feature for an intercept. Implement partial_fit and predict.
    """

    def __init__(self, n_features, n_outputs=1, prior_std=1., noise_std=1.):
        if n_features < 1 or n_outputs < 1 or not np.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("Positive dimensions and prior_std are required")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_std = prior_std
        self.noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (n_outputs,)).copy()
        if not np.isfinite(self.noise_std).all() or (self.noise_std <= 0).any():
            raise ValueError("noise_std must be finite and positive")

        self.reset()

    def reset(self):
        self.coef_ = np.zeros((self.n_outputs, self.n_features))
        covariance = self.prior_std**2 * np.eye(self.n_features)
        self.sigma_ = np.tile(covariance, (self.n_outputs, 1, 1))
        self.n_samples_seen_ = 0
        return self

    def fit(self, X, y):
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        """Update from X (samples, features) and y (samples, outputs).

        A single output also accepts y (samples,). Return self.
        """

        raise NotImplementedError("Implement BayesianLinearRegression.partial_fit")

    def predict(self, X, return_std=False, include_noise=True):
        """Return means, or (means, standard deviations), for X (..., features).

        Outputs have shape (..., outputs), or (...) for one output.
        include_noise adds observation noise to the uncertainty.
        """

        raise NotImplementedError("Implement BayesianLinearRegression.predict")
