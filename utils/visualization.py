import matplotlib.pyplot as plt
import numpy as np

def plot_loss(errors):
    plt.plot(errors)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy")
    plt.grid()
    plt.show()

def plot_decision_boundary(model, X, y, scaler):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid).reshape(-1, 2, 1)
    predictions = model.predict(grid_scaled)
    Z = np.round(predictions).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    plt.title("Decision Boundary")
    plt.show()
