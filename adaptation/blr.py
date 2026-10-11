import numpy as np


class BayesianLinearRegression:
    """Independent Gaussian weight posteriors for each output.

    coef_[j] and sigma_[j] hold the mean and covariance for output j.
    Include a constant feature for an intercept.
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

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        if self.n_outputs == 1 and y.ndim <= 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 1:
            y = y[None, :]

        noise_variance = self.noise_std**2
        for features, target in zip(X, y):
            covariance_feature = self.sigma_ @ features
            prediction_variance = noise_variance + features @ covariance_feature.T
            gain = covariance_feature / prediction_variance[:, None]
            prediction_error = target - self.coef_ @ features

            self.coef_ += gain * prediction_error[:, None]
            self.sigma_ -= gain[:, :, None] * covariance_feature[:, None, :]
            self.n_samples_seen_ += 1

        return self

    def predict(self, X, return_std=False, include_noise=True):
        """Return means, or (means, standard deviations), for X (..., features).

        Outputs have shape (..., outputs), or (...) for one output.
        include_noise adds observation noise to the uncertainty.
        """

        X = np.asarray(X, dtype=float)

        mean = X @ self.coef_.T

        if return_std:
            variance = np.empty_like(mean)
            for output, covariance in enumerate(self.sigma_):
                variance[..., output] = np.sum(X * (X @ covariance), axis=-1)
            variance = np.maximum(variance, 0)
            if include_noise:
                variance += self.noise_std**2
            std = np.sqrt(variance)

        if self.n_outputs == 1:
            mean = mean[..., 0]
            if return_std:
                std = std[..., 0]

        return (mean, std) if return_std else mean
