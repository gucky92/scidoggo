"""
Rank1 plus sparse matrix linear regression model.

sklearn >= 0.24.1
cvxpy
numpy
"""

import warnings

import numpy as np
import cvxpy as cp
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel


class Rank1PlusSparse(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Rank1-contrained and sparse matrix constrained Linear Regression:

        W = w1.T @ w2 + S
        Y = X @ W

    Parameters
    ----------
    normalize_w1 : bool, optional
        Whether to normalize the w1 vector by the l1-norm.
    normalize_w2 : bool, optional
        Whether to normalize the w2 vector by the l1-norm.
    S_mask : numpy.ndarray, optional
        The boolean mask for the sparse matrix.
    w1_sign : {-1, 1}, optional
        Whether the w1 vector has a particular sign (all positive or negative values).
    w2_sign : {-1, 1}, optional
        Whether the w2 vector has a particular sign (all positive or negative values).
    S_sign : {-1, 1}, optional
        Whether the S matrix has a particular sign (all positive or negative values).
    atol : float, optional
        Absolute tolerance for terminating fit, when comparing loss to loss of previous iteration.
    rtol : float, optional
        Relative tolerance for terminating fit, when comparing loss to loss of previous iteration.
    max_iter : int, optional
        Maximum number of iterations
    """

    def __init__(
        self, *,
        normalize_w1=True,
        normalize_w2=False,
        S_mask=None,
        w1_sign=None,
        w2_sign=None,
        S_sign=None,
        atol=1e-8,
        rtol=1e-5,
        max_iter=1000, 
        cp_kwargs={}
    ):
        self.normalize_w1 = normalize_w1
        self.normalize_w2 = normalize_w2
        self.S_mask = S_mask
        self.w1_sign = w1_sign
        self.w2_sign = w2_sign
        self.S_sign = S_sign
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.cp_kwargs = cp_kwargs

    def fit(self, X, y):
        """
        Fit model

        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = self._validate_data(X, y, accept_sparse=False,
                                   y_numeric=True, multi_output=True)

        X, y, _, _, _ = self._preprocess_data(
            X, y, fit_intercept=False, normalize=False,
            copy=False, sample_weight=None,
            return_mean=True)

        n = X.shape[1]
        m = y.shape[1]

        constraints = []

        S_sign = (1. if self.S_sign is None else self.S_sign)
        w1_sign = (1. if self.w1_sign is None else self.w1_sign)
        w2_sign = (1. if self.w2_sign is None else self.w2_sign)

        w1_ = np.random.random((n, 1)) * w1_sign
        # normalize w1
        if self.normalize_w1:
            w1_ /= np.sum(np.abs(w1_))
        w2_ = np.random.random((1, m)) * w2_sign
        # normalize w2
        if self.normalize_w2:
            w2_ /= np.sum(np.abs(w2_))
        if self.S_mask is None:
            S_ = 0.
        else:
            S = np.asarray(self.S_mask).astype(float)
            assert S.shape == (n, m)
            S_ = cp.Variable(S.shape, name='S', pos=(self.S_sign is not None))
            S_ = cp.multiply(S_, S_sign)
            constraints.append(
                S_[S == 0] == 0
            )

        loss = np.zeros(self.max_iter)

        for i in range(self.max_iter):
            one = i % 2

            if one:
                w1_ = cp.Variable((n, 1), name='w1', pos=(self.w1_sign is not None))
                w1_ = cp.multiply(w1_, w1_sign)
            else:
                w2_ = cp.Variable((1, m), name='w2', pos=(self.w2_sign is not None))
                w2_ = cp.multiply(w2_, w2_sign)

            W = cp.multiply(w1_, w2_) + S_
            ypred = X @ W
            obj = cp.Minimize(0.5 * cp.sum_squares(y - ypred))
            prob = cp.Problem(obj, constraints)
            loss[i] = prob.solve(**self.cp_kwargs)

            if one:
                w1_ = w1_.value
                # normalize w1
                if self.normalize_w1:
                    w1_ /= np.sum(np.abs(w1_))
            else:
                w2_ = w2_.value
                # normalize w2
                if self.normalize_w2:
                    w2_ /= np.sum(np.abs(w2_))

            # recalculate loss due to normalization step
            if self.normalize_w1 or self.normalze_w2:
                W = w1_ * w2_
                if self.S_mask is not None:
                    W += S_.value * S_sign
                ypred = X @ W
                loss[i] = 0.5 * np.sum(np.square(y - ypred))

            if np.isclose(loss[i-1], loss[i], atol=self.atol, rtol=self.rtol):
                loss[i+1:] = loss[i]
                if self.S_mask is not None:
                    S_ = S_.value
                break
        else:
            warnings.warn(
                "Maximum number of iterations reached, "
                "consider increasing `max_iter`, or changin `rtol`/`atol`."
            )
            if self.S_mask is not None:
                S_ = S_.value

        self.intercept_ = 0.
        self.w1_ = w1_
        self.w2_ = w2_
        self.S_ = S_
        self.loss_ = loss
        self.coef_ = (w1_ * w2_ + S_).T
        return self


def simple_model_test():
    n = 4
    m = 4
    s = 100
    mu = 0
    sigma = 1
    noise_sigma = 0.01

    np.random.seed(10)

    X = np.random.normal(mu, sigma, (s, n))
    w1 = np.random.normal(mu, sigma, n)
    w1 /= np.sum(np.abs(w1))
    w2 = -np.random.random(m)
    W = np.outer(w1, w2)

    S = np.eye(n, m)
    S = np.roll(S, 2, axis=1)

    Y = X @ (W + -S)
    y = Y + np.random.normal(0, noise_sigma, Y.shape)

    model = Rank1PlusSparse(S_mask=S, S_sign=-1, w2_sign=-1)
    model.fit(X, y)
    print("score: ", model.score(X, y))
    return model


def covariance_model_test():
    n = 4
    s = 100

    np.random.seed(10)
    cov = np.random.random((n, n))
    cov = (cov @ cov.T) + np.eye(n)

    X = np.random.multivariate_normal(np.zeros(n), cov, size=s)
    S_mask = np.roll(np.eye(n), 2, axis=1).astype(bool)
    model = Rank1PlusSparse(S_mask=S_mask, S_sign=1, w2_sign=1, w1_sign=1)
    model.fit(X, X)
    print("score: ", model.score(X, X))
    return model, cov
