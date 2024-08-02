import matplotlib.pyplot as plt
import numpy as np


def plot_results(X, y, X_train, y_train, y_pred, loss, epoch, func):
    plt.clf()
    ax = plt.subplot(111)
    plt.autoscale(False)
    plt.xlim(0, 18)
    plt.ylim(-3, 3)
    plt.plot(X, y, label='Original curve', color='g')
    plt.scatter(X_train, y_train, label='Noisy training data', color='b', s=5)
    plt.plot(X, y_pred, label='Neural network prediction', color='r')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.annotate('Made by @quantech_ai', xy=(0, 0), xytext=(14, -2.8))
    plt.annotate(f'MSE loss:{np.round(loss.item(), 5)}', xy=(0, 0), xytext=(0, 2.8))
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Fitting a neural network to a {func} curve after {epoch} epochs')
    plt.pause(0.01)
    # plt.show()
