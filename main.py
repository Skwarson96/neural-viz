import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import imageio
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from models.simple_nn import SimpleNN
from utils.plot_utils import plot_results, fig_to_array
from data.data_generation import generate_data


def train_nn(model, criterion, optimizer, X_train, y_train, X, y, args):
    figures = []
    fig, ax = plt.subplots(figsize=(10, 6))
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

        result_fig = plot_results(X, y, X_train, y_train, y_pred, loss, epoch, fig, ax, args)
        figures.append(result_fig)

    if args.save:
        figures[0].save(
            'animation.gif',
            save_all=True,
            append_images=figures[1:],
            duration=200,
            loop=0
        )


def main(args):

    X, y, X_train, y_train, X_all = generate_data(args.start, args.stop, args.resolution, args.amplitude, args.noise_level, args.function)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_nn(model, criterion, optimizer, X_train, y_train, X_all, y, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural-Viz Project")

    parser.add_argument('--start', type=int, default=-10, help='X axis start value')
    parser.add_argument('--stop', type=int, default=10, help='X axis end value')
    parser.add_argument('--resolution', type=int, default=500, help='Resolution of the generated data')
    parser.add_argument('--amplitude', type=float, default=2.0, help='Amplitude of the sinusoidal function')
    parser.add_argument('--function', type=str, default='sawtooth', choices=['sin', 'rectangle', 'sawtooth', 'polynomial'], help='Type of function to learn')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--interval', type=int, default=50, help='Interval for plotting results')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level added to the data')
    parser.add_argument('--save', action='store_true', required=False, help='Save gif file')

    args = parser.parse_args()

    main(args)
