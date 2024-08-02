import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models.simple_nn import SimpleNN
from utils.plot_utils import plot_results
from data.data_generation import generate_data

def train_and_plot(model, criterion, optimizer, X_train, y_train, X, y, args):
    for epoch in range(0, args.epochs + 1, args.interval):
        for _ in range(args.interval):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X).numpy()

        plot_results(X, y, X_train, y_train, y_pred, loss, epoch, args)

def main(args):

    X, y, X_train, y_train, X_all = generate_data(args.start, args.stop, args.resolution, args.amplitude, args.noise_level, args.function)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # plt.figure(figsize=(10, 6))

    train_and_plot(model, criterion, optimizer, X_train, y_train, X_all, y, args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural-Viz Project")

    parser.add_argument('--start', type=int, default=-10, help='X axis start value')
    parser.add_argument('--stop', type=int, default=10, help='X axis end value')
    parser.add_argument('--resolution', type=int, default=500, help='Resolution of the generated data')
    parser.add_argument('--amplitude', type=float, default=2.0, help='Amplitude of the sinusoidal function')
    parser.add_argument('--function', type=str, default='sin', choices=['sin', 'rectangle', 'sawtooth', 'polynomial'], help='Type of function to learn')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--interval', type=int, default=50, help='Interval for plotting results')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level added to the data')

    # Optional arguments for polynomial functions
    parser.add_argument('--poly_degree', type=int, default=2, help='Degree of the polynomial function')
    parser.add_argument('--poly_coeffs', type=float, nargs='+', default=None,
                        help='Coefficients of the polynomial function (e.g., --poly_coeffs 1 -2 1)')

    args = parser.parse_args()

    main(args)
