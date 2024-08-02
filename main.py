import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_nn import SimpleNN
from utils.plot_utils import plot_results
from data.data_generation import generate_data


def main():
    cycles = 3
    resolution = 500
    amplitude = 2

    X, y, X_train, y_train, X_all = generate_data(cycles, resolution, amplitude)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 3000
    interval = 50
    plt.figure(figsize=(10, 6))

    for epoch in range(0, epochs + 1, interval):
        for _ in range(interval):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_all).numpy()

        func = 'sinus'
        plot_results(X, y, X_train, y_train, y_pred, loss, epoch, func)

    plt.show()


if __name__ == "__main__":
    main()
