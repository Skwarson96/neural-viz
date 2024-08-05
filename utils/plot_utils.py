import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image


def fig_to_array(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_results(X, y, X_train, y_train, y_pred, loss, epoch, fig, ax, args):
    plt.clf()
    plt.autoscale(False)
    plt.xlim(args.start, args.stop)
    plt.ylim(np.min(y) - 1, np.max(y) + 1)
    plt.plot(X, y, label="Original curve", color="g")
    plt.scatter(X_train, y_train, label="Noisy training data", color="b", s=5)
    plt.plot(X, y_pred, label="Neural network prediction", color="r")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.annotate(
        "Made by @quantech_ai",
        xy=(0, 0),
        xytext=((args.start + np.abs(args.stop)) / 2, np.min(y) - 0.75),
    )
    plt.annotate(
        f"MSE loss:{np.round(loss.item(), 5)}",
        xy=(0, 0),
        xytext=(args.start, np.max(y) - 2),
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(
        f"Fitting a neural network to a {args.function} curve after {epoch} epochs"
    )
    plt.pause(0.01)

    return fig_to_array(fig)
