import numpy as np


def sample_noisy_data(n_samples: int, data: np.array, scale: float = 0.2):
    samples_ids = np.random.choice(data.shape[0], n_samples)
    samples = data[samples_ids]
    noisy_samples = samples + 0.2 * np.random.normal(scale=scale, size=samples.shape)
    return noisy_samples
