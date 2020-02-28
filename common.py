import numpy as np
import scipy
import os
import glob

import torch
from torch.utils.data import DataLoader, Dataset

import fym.logging as logging


class OuNoise:
    """
    A Ornstein Uhlenbeck action noise,
    this is designed to approximate brownian motion with friction.
    Based on https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/noise.py
    """

    def __init__(self, mean, sigma, theta=.1, dt=1e-2, max_t=1,
                 decay=10, initial_noise=None, preprocessing=True):
        self.mu = np.asarray(mean)
        self.decay = decay
        self.p1 = theta * dt
        self.p2 = sigma * np.sqrt(dt)
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

        self.preprocessing = preprocessing
        if self.preprocessing:
            self.propagate(dt, max_t)

    def get(self, t):
        if not self.preprocessing:
            return self._get(t)
        else:
            return self.interp(t)

    def _get(self, t):
        noise = np.exp(-t / self.decay) * (
            self.noise_prev
            + self.p1 * (self.mu - self.noise_prev)
            + self.p2 * np.random.normal(size=self.mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        if self.initial_noise is not None:
            self.noise_prev = self.initial_noise
        else:
            self.noise_prev = np.zeros_like(self.mu)

    def propagate(self, dt, max_t):
        thist = np.arange(0, max_t, dt)
        res = np.zeros(thist.shape + self.mu.shape)
        for i, t in enumerate(thist):
            res[i] = self._get(t)

        self.interp = scipy.interpolate.interp1d(
            thist, res, axis=0, bounds_error=False,
            fill_value=(res[0], res[-1]))


class DictDataset(Dataset):
    def __init__(self, file_names, keys, transform=None):
        if isinstance(file_names, str):
            file_names = (file_names, )

        data_all = [logging.load(name) for name in file_names]
        self.keys = keys if not isinstance(keys, str) else (keys, )

        _data = {
            k: np.vstack([data[k] for data in data_all])
            for k in self.keys
        }

        self.data = _data
        self.len = len(self.data[self.keys[0]])
        self.transform = transform

    def __getitem__(self, idx):
        data = [self.data[k][idx] for k in self.keys]

        if self.transform:
            data = self.transform(*data)

        return [torch.from_numpy(d).float() for d in data]

    def __len__(self):
        return self.len


def get_dataloader(sample_files, keys=("state", "action"),
                   transform=None, **kwargs):
    dataloader = DataLoader(
        DictDataset(sample_files, keys, transform=transform),
        **kwargs
    )
    return dataloader


def parse_file(files, ext="h5"):
    if isinstance(files, str):
        files = [files]
    target = []
    for file in files:
        if os.path.isdir(file):
            target += sorted(glob.glob(os.path.join(file, "*." + ext)))
        else:
            target += [file]
    return target
