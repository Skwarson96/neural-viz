import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models.simple_nn import SimpleNN
from utils.plot_utils import plot_results
from data.data_generation import generate_data

def train_and_plot(model, criterion, optimizer, X_train, y_train, X, y, epochs=3000, interval=50, func='sin'):
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
            y_pred = model(X).numpy()

        plot_results(X, y, X_train, y_train, y_pred, loss, epoch, func)

def main(args):
    X, y, X_train, y_train, X_all = generate_data(args.cycles, args.resolution, args.amplitude, args.noise_level)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # plt.figure(figsize=(10, 6))

    train_and_plot(model, criterion, optimizer, X_train, y_train, X_all, y, epochs=args.epochs, interval=args.interval, func=args.func)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NeuroViz Project")

    parser.add_argument('--cycles', type=int, default=3, help='Number of sinusoidal cycles')
    parser.add_argument('--resolution', type=int, default=500, help='Resolution of the generated data')
    parser.add_argument('--amplitude', type=float, default=2.0, help='Amplitude of the sinusoidal function')
    parser.add_argument('--function', type=str, default='sin', choices=['sin', 'rectangle', 'sawtooth', 'polynomial'], help='Type of function to learn')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--interval', type=int, default=50, help='Interval for plotting results')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level added to the data')

    args = parser.parse_args()

    main(args)
