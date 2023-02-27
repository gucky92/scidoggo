"""
"""
from numbers import Number
from itertools import product

import numpy as np
from scipy.optimize import least_squares

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class SelectivityModel(BaseEstimator, RegressorMixin):
    
    def __init__(
        self, 
        kappas=np.logspace(-2, 1, 4+3*5), 
        alphas=np.logspace(-1, 1, 3+2*5), 
        linear_features=None,
        eps=1e-5
    ):
        self.eps = eps
        self.kappas = kappas
        self.alphas = alphas
        self.linear_features = linear_features

    def _selective_regression(self, X_normalized, X_norm, w_normalized, w_norm, alpha, kappa):
        return (
            np.sign(kappa) 
            * 
            X_norm ** alpha
            * 
            w_norm 
            * 
            (np.exp(kappa * (X_normalized @ w_normalized)) -1) 
            / (np.abs(kappa)+self.eps)
        )
        
    def _model(self, theta, X, X_norm, a, k):
        w = theta[self.n_linear_:]
        w_norm = np.linalg.norm(w)
        w_normalized = w / w_norm
        y_pred = self._selective_regression(
            X[:, self.n_linear_:], X_norm, w_normalized, w_norm, a, k
        ) + X[:, :self.n_linear_] @ theta[:self.n_linear_]
        return y_pred

    def _residuals(self, theta, X, X_norm, y, a, k, sample_weight):
        y_pred = self._model(theta, X, X_norm, a, k)
        return np.sqrt(sample_weight) * (y - y_pred)
    
    def _check_and_assign_params(self):
        if self.linear_features is None:
            self.linear_features_ = []
            self.n_linear_ = 0
        else:
            self.linear_features_ = (
                [self.linear_features] 
                if isinstance(self.linear_features, Number) 
                else list(self.linear_features)
            )
            self.n_linear_ = len(self.linear_features_)
    
    def _transformX(self, X, return_nonnormalized=False):
        idcs = list(range(X.shape[1]))
        idcs = list(set(idcs) - set(self.linear_features_))
        Xlinear = X[:, self.linear_features_]
        X = X[:, idcs]
        if return_nonnormalized:
            return np.hstack([Xlinear, X])
        
        xnorm = np.linalg.norm(X, axis=-1, ord=2)
        X = X.copy()
        X[xnorm != 0] /= xnorm[xnorm != 0, None]
        X[~np.isfinite(X)] = 0
        
        X = np.hstack([Xlinear, X])
        return X, xnorm
    
    def fit(self, X, y, sample_weight=None):
        
        self._check_and_assign_params()
        
        X4linear = self._transformX(X, return_nonnormalized=True)
        lin_model = LinearRegression(fit_intercept=False)
        lin_model.fit(X4linear, y, sample_weight=sample_weight)
        self.lin_model_ = lin_model
        
        self.w_linear_ = lin_model.coef_
        
        X, X_norm = self._transformX(X)
        sample_weight = (np.ones_like(y) if sample_weight is None else sample_weight)
        
        params = []
        r2s = []
        for a, k in product(self.alphas, self.kappas):
            result = least_squares(
                self._residuals, self.w_linear_, 
                args=(X, X_norm, y, a, k, sample_weight), 
            )
            test_y = self._model(
                result.x, X, X_norm, a, k
            )
            r2s.append(r2_score(test_y, y, sample_weight=sample_weight))
            params.append([result.x, a, k])
            
        self.r2s_ = np.array(r2s)
        self.params_ = params
        
        self.coef_, self.alpha_, self.kappa_ = params[np.argmax(r2s)]
        
    def predict(self, X):
        X, X_norm = self._transformX(X)
        return self._model(self.coef_, X, X_norm, self.alpha_, self.kappa_)
        