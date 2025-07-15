from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def load_moons():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.reshape(-1, 2, 1), y.reshape(-1, 1, 1), X, y, scaler
