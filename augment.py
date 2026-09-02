from copy import deepcopy

import numpy as np


class GaussianPositionObservationAugmentor:
    def __init__(self, mu=0.0, sigma=0.0):
        self.mu = float(mu)
        self.sigma = float(sigma)

    @property
    def enabled(self):
        return abs(self.mu) > 0.0 or abs(self.sigma) > 0.0

    def apply(self, raw_obs, norm_obs):
        if not self.enabled:
            return raw_obs, norm_obs
        return self._apply_single(raw_obs), self._apply_single(norm_obs)

    def _sample_scale(self, shape):
        return 1.0 + np.random.normal(loc=self.mu, scale=self.sigma, size=shape)

    def _scale_self_obs(self, values):
        idx = np.asarray([0, 1, 4, 5], dtype=np.int64)
        values[idx] = values[idx] * self._sample_scale(values[idx].shape)
        return values

    def _scale_fixed_neighbor_obs(self, values):
        token_dim = 5
        if values.size == 0 or values.size % token_dim != 0:
            return values
        reshaped = values.reshape(-1, token_dim)
        reshaped[:, 0:2] = reshaped[:, 0:2] * self._sample_scale(reshaped[:, 0:2].shape)
        return reshaped.reshape(-1)

    def _scale_radar_obs(self, values):
        if values.size == 0:
            return values
        values[:] = values * self._sample_scale(values.shape)
        return values

    def _scale_variable_neighbors(self, values):
        arr = np.asarray(values, dtype=np.float64).copy()
        squeezed = False
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = np.squeeze(arr, axis=1)
            squeezed = True
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr[:, 0:2] = arr[:, 0:2] * self._sample_scale(arr[:, 0:2].shape)
        if squeezed:
            arr = np.expand_dims(arr, axis=1)
        return arr

    def _apply_single(self, obs):
        augmented = deepcopy(obs)
        if len(augmented) >= 1:
            augmented[0] = [self._scale_self_obs(np.asarray(item, dtype=np.float64).copy()) for item in augmented[0]]
        if len(augmented) >= 2:
            augmented[1] = [
                self._scale_fixed_neighbor_obs(np.asarray(item, dtype=np.float64).copy())
                for item in augmented[1]
            ]
        if len(augmented) >= 3:
            augmented[2] = [self._scale_radar_obs(np.asarray(item, dtype=np.float64).copy()) for item in augmented[2]]
        if len(augmented) >= 4:
            augmented[3] = [self._scale_variable_neighbors(item) for item in augmented[3]]
        return augmented
