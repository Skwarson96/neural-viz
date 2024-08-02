import numpy as np
import torch


def generate_data(start, stop, resolution, amplitude, noise_level, func):
    # length = np.pi * 2 * cycles

    X = np.linspace(start=start, stop=stop, num=resolution).reshape(-1, 1)
    if func == 'sin':
        y = amplitude * np.sin(X)
    elif func == 'rectangle':
        y = amplitude * np.sign(np.sin(X))
    elif func == 'sawtooth':
        y = amplitude * (X / np.pi - np.floor(0.5 + X / np.pi))
    elif func == 'polynomial':
        y = 0.005 * X ** 3 + 0.1 * X ** 2 + 0.001 * X - 2
    else:
        raise ValueError("Unsupported function type")

    noise = np.random.normal(0, noise_level, y.shape)
    y_noisy = y + noise

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X[400:], dtype=torch.float32)
    y_test = torch.tensor(y_noisy[400:], dtype=torch.float32).view(-1, 1)
    X_all = torch.tensor(X, dtype=torch.float32)

    return X, y, X_train, y_train, X_all
