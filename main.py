import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def main():
    # Parametry
    cycles = 3  # ile cykli sinusoidalnych
    resolution = 500  # ile punktów danych wygenerować
    amplitude = 2  # amplituda sinusoidy

    # Generowanie danych sinusoidalnych
    length = np.pi * 2 * cycles
    X = np.linspace(0, length, resolution).reshape(-1, 1)
    y = amplitude * np.sin(X).ravel()  # Zwiększenie amplitudy

    # Dodanie szumu do danych
    noise = np.random.normal(0, 0.1, y.shape)
    y_noisy = y + noise

    # Konwersja danych na tensory
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X[400:], dtype=torch.float32)
    y_test = torch.tensor(y_noisy[400:], dtype=torch.float32).view(-1, 1)
    X_all = torch.tensor(X, dtype=torch.float32)

    # Definiowanie modelu sieci neuronowej
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Trenowanie modelu i śledzenie postępów
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

        # Przewidywanie wartości
        model.eval()
        with torch.no_grad():
            y_pred = model(X_all).numpy()

        # Wizualizacja wyników
        plt.clf()
        ax = plt.subplot(111)
        plt.autoscale(False)
        plt.xlim(0, 18)
        plt.ylim(-3, 3)
        plt.plot(X, y, label='Original sine curve', color='g')
        plt.scatter(X_train, y_train, label='Noisy training data', color='b', s=5)
        plt.plot(X, y_pred, label='Neural network prediction', color='r')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        plt.annotate('Made by @quantech_ai', xy=(0, 0), xytext=(14, -2.8),)
        plt.annotate(f'MSE loss:{np.round(loss.item(), 5)}', xy=(0, 0), xytext=(0, 2.8),)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Fitting a neural network to a sinusoidal curve after {epoch} epochs')
        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()
