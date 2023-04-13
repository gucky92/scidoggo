import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scidoggo import Tikhonov

def test_tikhonov_fit():
    # Generate random regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1,
                           random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Fit Tikhonov model with default parameters
    model = Tikhonov()
    model.fit(X_train, y_train)

    # Test predictions on testing set
    y_pred = model.predict(X_test)

    # Check that mean squared error is low
    assert mean_squared_error(y_test, y_pred) < 0.5

def test_tikhonov_fit_with_L():
    # Generate random regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1,
                           random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Define regularization matrix L
    rng = np.random.default_rng(10)
    L = rng.random((X.shape[1], X.shape[1]))
    L = (L.T @ L + np.eye(X.shape[1]))/2

    # Fit Tikhonov model with L
    model = Tikhonov(L=L)
    model.fit(X_train, y_train)

    # Test predictions on testing set
    y_pred = model.predict(X_test)

    # Check that mean squared error is low
    assert mean_squared_error(y_test, y_pred) < 0.5
