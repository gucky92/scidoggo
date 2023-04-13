"""
rank constraint linear regression
"""

import numpy as np
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class RankConstraint(BaseEstimator):
    """
    A custom estimator for performing least squares linear regression with a rank constraint on the weight matrix W. 
    This estimator uses iterative optimization to minimize the least square error while constraining the rank of W. 

    Parameters
    ----------
    n_iter : int, default=100000
        The maximum number of iterations to perform during training.

    rank : int, default=1
        The rank constraint on the weight matrix W.

    alpha : float, default=1e-9
        The learning rate for the gradient descent algorithm used to optimize W.

    Attributes
    ----------
    losses_ : list
        The list of loss values at each iteration during training.

    G_ : ndarray of shape (n_features, rank)
        The matrix G obtained from the SVD of the weight matrix W, where rank is the rank constraint on W.

    H_ : ndarray of shape (rank, n_features)
        The matrix H obtained from the SVD of the weight matrix W, where rank is the rank constraint on W.
    """

    def __init__(self, n_iter=100000, rank=1, alpha=1e-9):
        self.n_iter = n_iter
        self.rank = rank
        self.alpha = alpha

    def _init_GH(self, W_full_rank):
        """Initializes the matrices G and H from a full-rank weight matrix W_full_rank."""
        pca = PCA(n_components=self.rank, random_state=0)
        G = pca.fit_transform(W_full_rank)
        H = pca.components_.T
        return G, H

    def _compute_loss(self, G, H, X, y):
        """Computes the least square error between the predicted and actual values of the target variable."""
        y_pred = np.dot(X, np.dot(G, H.T))
        loss = LA.norm(y_pred - y)
        return loss

    def fit(self, X, y):
        """Fits the estimator to the training data X and y."""
        # Fit a linear regression model to get the full-rank weight matrix
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        W_full_rank = reg.coef_.T

        # Initialize G and H matrices
        G, H = self._init_GH(W_full_rank)

        # Iteratively optimize G and H matrices
        losses = []
        for _ in range(self.n_iter):
            y_pred = np.dot(X, np.dot(G, H.T))

            grad_G = np.dot(X.T, np.dot((y_pred - y), np.dot(H, np.dot(G.T, G))))
            grad_H = np.dot((y_pred - y).T, np.dot(X, np.dot(G, np.dot(H.T, H))))

            G = G - (self.alpha * grad_G)
            H = H - (self.alpha * grad_H)

            loss = self._compute_loss(G, H, X, y)
            losses.append(loss)
            
        self.G_ = G
        self.H_ = H
        self.losses_ = losses
        return self

    def predict(self, X):
        """Applies the fitted estimator to new data X."""
        y_pred = np.dot(X, np.dot(self.G_, self.H_.T))
        return y_pred