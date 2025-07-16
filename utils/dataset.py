from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def load_moons():
    # Generate moon dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    # Keep unscaled for visualization
    X_raw = X.copy()
    y_raw = y.copy()

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for model (features, 1)
    X_scaled = X_scaled.reshape(-1, 2, 1)
    y = y.reshape(-1, 1, 1)

    return X_scaled, y, X_raw, y_raw, scaler
