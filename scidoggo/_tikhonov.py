"""
Tikhonov regression,

based on implementation by Stout and Kalivas, 2006. Journal of Chemometrics
L2-regularized regression using a non-diagonal regularization matrix

This can be done in two ways, by setting the original problem into
"standard space", such that regular ridge regression can be employed,
or solving the equation in original space. As number features increases,
rotating the original problem should be faster

Modified from Jeff Chiang's code <jeff.njchiang@gmail.com>
"""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model import Ridge
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler



def _qr(x):
    """
    QR factorization of matrix x.

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_features]
        Input matrix.

    Returns
    -------
    qp : array-like, shape = [n_samples, n_features]
        First block matrix of QR factorization of x.
    qo : array-like, shape = [n_samples, n_regularizers]
        Second block matrix of QR factorization of x.
    rp : array-like, shape = [n_features, n_features]
        Upper triangular matrix of QR factorization of x.
    """
    m, n = x.shape
    q, r = np.linalg.qr(x, mode='complete')
    rp = r[:n, :]
    qp = q[:, :n]
    qo = q[:, n:]
    # check for degenerate case
    if qo.shape[1] == 0:
        qo = np.array(1.0)
    return qp, qo, rp


def analytic_tikhonov(x, y, alpha, sigma=None):
    """
    Solves Tikhonov regularization problem with the covariance of the
    weights as a prior.

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_features]
        Training data.
    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values.
    alpha : float
        Regularization parameter.
    sigma : array-like, shape = [n_features, n_features], optional
        Covariance matrix of the prior.

    Returns
    -------
    beta_hat : array-like, shape = [n_features] or [n_features, n_targets]
        Beta weight estimates.
    """
    if sigma is None:
        sigma = np.eye(x.shape[1])
    return np.dot(np.linalg.pinv(np.dot(x.T, x) +
                  np.linalg.pinv(sigma) *
                  alpha), np.dot(x.T, y))


def find_tikhonov_from_covariance(x, cutoff=.0001, eps=1e-10):
    """
    Use truncated-SVD to find Tikhonov matrix.

    Parameters
    ----------
    x : array-like, shape = [n_features, n_features]
        Feature x feature covariance matrix. This is used to find a Tikhonov
        matrix L such that inv(x) = L.T * L.
    cutoff : float, optional
        Cutoff value for singular value magnitude. If it's too low, rank will
        suffer.
    eps : float, optional
        Tolerance for singular values to consider non-zero.

    Returns
    -------
    L : array-like, shape = [n_features, n_regularizers]
        The Tikhonov matrix for this situation.
    """
    if not np.allclose(x.T, x):
        raise ValueError("Input matrix is not symmetric. "
                         "Are you sure it is covariance?")
    _, s, vt = np.linalg.svd(x)
    return np.dot(np.diag(1/np.sqrt(s[s > cutoff])), vt[s > cutoff])


def _standardize_params(x, L):
    """
    Calculates parameters associated with rotating the data to standard form.
    :param x: {array-like}, shape = [n_samples, n_features]
        Training data
    :param L: {array-like}, shape = [n_features, n_regularizers]
        Tikhonov matrix
    :return:
    hq: {array-like}, shape = [n_features, k]
        First block matrix of QR factorization of L.T.
        kp * rp^-1.T is inv(L)
    kp: {array-like}, shape = [k, n_regularizers]
        Second block matrix of QR factorization of L.T.
        kp * rp^-1.T is inv(L)
    rp: {array-like}, shape = [k, n_regularizers]
        Upper triangular matrix of QR factorization of L.T.
    ko: {array-like}, shape = [n_samples, k] or None
        If L is not square, the matrix to transform y to standard form
    to: {array-like}, shape = [n_samples, k] or None
        If L is not square, the matrix to transform y to standard form
    ho: {array-like}, shape = [n_samples, k] or None
        If L is not square, the matrix to transform y to standard form
    """
    kp, ko, rp = _qr(L.T)
    if ko is None:  # there is no lower part of matrix
        ho, hq, to = np.ones(1), np.ones(1), np.ones(1)
    else:
        ho, hq, to = _qr(np.dot(x, ko))
    if hq.shape is ():  # special case where L is square
                        # (saves computational time later)
        ko, to, ho = None, None, None
    return hq, kp, rp, ko, ho, to


def to_standard_form(x: np.ndarray, y: np.ndarray, L: np.ndarray) -> tuple:
    """
    Converts x and y into "standard form" in order to efficiently
    solve the Tikhonov regression problem.

    Parameters
    ----------
    x : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_targets).
    L : np.ndarray
        Generally, L.T * L is the inverse covariance matrix of the data.
        Shape is (n_features, n_regularizers).

    Returns
    -------
    x_new : ndarray, shape (n_samples + n_features, n_features)
        The stacked data and regularization matrix in standard form.
    y_new : ndarray, shape (n_samples + n_features,)
        The stacked target array in standard form.
    """
    # this is derived by doing a bit of algebra:
    # x_new = hq.T * x * kp * inv(rp).T
    hq, kp, rp, _, _, _ = _standardize_params(x, L)
    x_new = solve_triangular(rp, np.dot(kp.T, np.dot(x.T, hq))).T
    y_new = np.dot(hq.T, y)
    return x_new, y_new


def to_general_form(b: np.ndarray, x: np.ndarray, y: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Converts weights back into general form space.

    Parameters
    ----------
    b : np.ndarray
        Regression coefficients of shape (n_features,) or (n_features, n_targets).
    x : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_targets).
    L : np.ndarray
        Generally, L.T* L is the inverse covariance matrix of the data.
        Shape is (n_features, n_regularizers).

    Returns
    -------
    np.ndarray
        Ridge coefficients rotated back to original space of shape (n_features,) or (n_features, n_targets).
    """
    hq, kp, rp, ko, ho, to = _standardize_params(x, L)

    if ko is to is ho is None:
        L_inv = np.dot(kp, np.linalg.pinv(rp.T))
        return np.dot(L_inv, b)
    else:
        L_inv = np.dot(kp, np.linalg.pinv(rp.T))
        kth = np.dot(ko, np.dot(np.linalg.pinv(to), ho.T))
        resid = y - np.dot(x, np.dot(L_inv, b))
        return np.dot(L_inv, b) + np.dot(kth, resid)


def fit_learner(x: np.ndarray, y: np.ndarray, L: np.ndarray, ridge: Ridge = None) -> Ridge:
    """
    Returns a trained model that works exactly the same as Ridge, but fit optimally.

    Parameters
    ----------
    x : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_targets).
    L : np.ndarray
        Generally, L.T * L is the inverse covariance matrix of the data.
        Shape is (n_features, n_regularizers).
    ridge : Ridge, optional
        A Ridge object used to fit the transformed data, by default None.

    Returns
    -------
    Ridge
        A trained Ridge object with optimized coefficients.
    """
    if ridge is None:
        ridge = Ridge(fit_intercept=False)
    x_new, y_new = to_standard_form(x, y, L)
    ta_est_standard = ridge.fit(x_new, y_new).coef_
    ta_est = to_general_form(ta_est_standard.T, x, y, L)
    ridge.coef_ = ta_est.T
    return ridge


class Tikhonov(Ridge):
    """
    Tikhonov regularization estimator.

    This estimator extends the Ridge estimator by allowing the user to provide a
    regularization matrix L, which is used to compute the coefficients of the
    regression problem. The L matrix is multiplied by the L2 norm of the
    coefficients to produce a new penalty term that is added to the residual sum
    of squares.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    max_iter : int or None, default=None
        Maximum number of iterations for conjugate gradient solver.
        ``None`` means that the solver will be iterated until convergence.
        Only used by the conjugate gradient solver.
    tol : float, default=1e-3
        Precision of the solution. Only used by the conjugate gradient solver.
    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},
             default='auto'
        Solver to use in the computational routines:
        - 'auto' chooses the solver automatically based on the type of data.
        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than 'cholesky'.
        - 'cholesky' uses the standard scipy.linalg.solve function to obtain a
          closed-form solution to the Ridge coefficients.
        - 'lsqr' uses the iterative least-squares solver.
        - 'sparse_cg' uses the conjugate gradient solver.
        - 'sag' uses a Stochastic Average Gradient descent.
        - 'saga' uses a Stochastic Average Gradient descent.
    random_state : int, RandomState instance, default=None
        Determines random number generation for shuffling data. Pass an int for
        reproducible results across multiple function calls.

    Attributes
    ----------
    coef_ : array of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if ``fit_intercept=False``.
    """

    def __init__(
        self, alpha=1.0, L=None, *, 
        fit_intercept=False, copy_X=True, 
        max_iter=None, tol=0.0001, solver='auto', 
        positive=False, random_state=None
    ):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         copy_X=copy_X, positive=positive,
                         max_iter=max_iter, tol=tol, solver=solver,
                         random_state=random_state)
        self.L = L

    def fit(self, X, y, sample_weight=None):
        # Check if solver is stochastic gradient descent, and set data type
        if self.solver in ('sag', 'saga'):
            dtype = np.float64
        else:
            dtype = [np.float64, np.float32]
            
        assert not self.fit_intercept, "fit intercept not yet implemented"
        
        X, y = self._validate_data(
            X, y, ['csr', 'csc', 'coo'], dtype=dtype,
            multi_output=True, y_numeric=True
        )
        # If L is not None, transform data and fit with Ridge
        if self.L is not None:
            X, y = to_standard_form(X, y, self.L)
            super().fit(X, y, sample_weight=sample_weight)
            self.coef_ = to_general_form(
                self.coef_.T, X, y, self.L
            ).T
        else:
            super().fit(X, y, sample_weight=sample_weight)
        return self
        