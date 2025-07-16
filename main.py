from model.layers import Dense
from model.activations import Relu, Sigmoid
from model.losses import binary_cross_entropy, binary_cross_entropy_prime
from model.network import Sequential
from utils.dataset import load_moons
from utils.visualization import plot_loss, plot_decision_boundary
from utils.metrics import binary_classification_metrics

from sklearn.model_selection import train_test_split

# Load and split dataset
x_all, y_all, X_raw, y_raw, scaler = load_moons()

# Split both for training/testing and visualization
x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42
)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(2, 6))
model.add(Relu())
model.add(Dense(6, 1))
model.add(Sigmoid())
model.use(binary_cross_entropy, binary_cross_entropy_prime)

errors = model.fit(x_train, y_train, epochs=1000, learning_rate=0.01)

plot_loss(errors)
plot_decision_boundary(model, X_test_raw, y_test_raw, scaler)

y_pred_probs = model.predict(x_test)                # shape: (n_samples, 1)
y_pred_labels = (y_pred_probs > 0.5).astype(int)
y_true_labels = y_test.reshape(-1, 1)

print("\nTest Set Evaluation:")
binary_classification_metrics(y_true_labels, y_pred_labels)
