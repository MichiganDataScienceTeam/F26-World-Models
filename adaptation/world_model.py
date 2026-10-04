import numpy as np


class AdaptiveWorldModel:
    """The correction model must provide predict, partial_fit, and reset.

    Features default to the frozen model's hidden layer. output_scale converts
    corrections to observation units. Uncertainty comes from the correction model.
    """

    def __init__(self, model, correction_model, feature_fn=None, output_scale=None):
        self.model = model
        self.correction_model = correction_model
        self.state_size = model.state_size
        self.feature_fn = feature_fn

        scale = 1. if output_scale is None else output_scale
        self.scale = np.broadcast_to(scale, (self.state_size,)).astype(float).copy()
        if not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("output_scale must be finite and positive")

    def reset(self):
        self.correction_model.reset()
        return self

    def _base_and_features(self, observations, actions):
        if self.feature_fn is None and hasattr(self.model, "predict_with_features"):
            return self.model.predict_with_features(observations, actions)

        features = self.model.features if self.feature_fn is None else self.feature_fn
        return self.model.predict(observations, actions), features(observations, actions)

    def partial_fit(self, observations, actions, next_observations):
        frozen, features = self._base_and_features(observations, actions)
        residual = (np.asarray(next_observations)[..., :self.state_size]
                    - frozen[..., :self.state_size]) / self.scale

        self.correction_model.partial_fit(
            features.reshape(-1, features.shape[-1]), residual.reshape(-1, self.state_size))
        return self

    def predict(self, observations, actions, return_std=False):
        frozen, features = self._base_and_features(observations, actions)
        result = self.correction_model.predict(features, return_std=return_std)
        correction, std = result if return_std else (result, None)
        shape = features.shape[:-1] + (self.state_size,)

        mean = frozen.copy()
        mean[..., :self.state_size] += np.asarray(correction).reshape(shape) * self.scale
        if not return_std:
            return mean

        std = np.asarray(std).reshape(shape) * self.scale
        context_std = np.zeros_like(mean[..., self.state_size:])
        return mean, np.concatenate((std, context_std), axis=-1)
