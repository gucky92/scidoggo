import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from scidoggo import RankConstraint

def test_fit_transform():
    # Generate random regression data
    X, y = make_regression(
        n_samples=100, n_features=10, noise=0.1, random_state=0, 
        effective_rank=2
    )
    
    # Fit the rank-constrained linear regression model
    model = RankConstraint(rank=2)
    model.fit(X, y)
    
    # Test the transform method
    y_pred = model.predict(X)
    assert y_pred.shape == (10,)
    assert mean_squared_error(y, y_pred) < 0.2
