import numpy as np
import torch


def generate_data(cycles, resolution, amplitude):
    length = np.pi * 2 * cycles

    X = np.linspace(0, length, resolution).reshape(-1, 1)
    y = amplitude * np.sin(X).ravel()


    noise = np.random.normal(0, 0.1, y.shape)
    y_noisy = y + noise

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X[400:], dtype=torch.float32)
    y_test = torch.tensor(y_noisy[400:], dtype=torch.float32).view(-1, 1)
    X_all = torch.tensor(X, dtype=torch.float32)

    return X, y, X_train, y_train, X_all
