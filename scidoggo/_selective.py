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
    """
    The model fits a linear regression to a subset of the features and uses a selective regression method to
    fit non-linear relationships between the remaining features and the target variable.
    
    Parameters
    ----------
    kappas : array-like, default=np.logspace(-2, 1, 4+3*5)
        The range of kappa values to be used in selective regression.
    alphas : array-like, default=np.logspace(-1, 1, 3+2*5)
        The range of alpha values to be used in selective regression.
    linear_features : int, list, or None, default=None
        The indices of features to be treated as linear features.
        If an integer is given, that many features starting from the first one are considered linear.
        If a list is given, the specified feature indices are considered linear.
        If None, all features are considered nonlinear.
    eps : float, default=1e-5
        A small positive constant added to kappa to prevent divide-by-zero errors in selective regression.
        
    Attributes
    ----------
    lin_model_ : sklearn.linear_model.LinearRegression
        The linear regression model used to fit the linear features.
    w_linear_ : ndarray of shape (n_linear_features,)
        The coefficients learned for the linear features by `lin_model_`.
    n_linear_ : int
        The number of linear features in the dataset.
    coef_ : ndarray of shape (n_features,)
        The learned coefficients for all features, including the linear features.
    alpha_ : float
        The optimal value of alpha found during fitting.
    kappa_ : float
        The optimal value of kappa found during fitting.
    r2s_ : ndarray of shape (n_alphas, n_kappas)
        The R^2 score for each combination of alpha and kappa tried during fitting.
    params_ : list of (ndarray, float, float)
        The parameters learned for each combination of alpha and kappa tried during fitting. Each element of the
        list is a tuple containing the coefficients, alpha, and kappa.
    
    Examples
    --------
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Create a regression dataset
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the SelectivityModel class
    model = SelectivityModel()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    """
    
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
        """
        Computes the output of selective regression for a given set of input features and parameters.
        
        Parameters
        ----------
        X_normalized : numpy array of shape (n_samples, n_features)
            The normalized input features.
        X_norm : numpy array of shape (n_samples,)
            The L2 norm of each row of X_normalized.
        w_normalized : numpy array of shape (n_features,)
            The normalized weight vector for selective regression.
        w_norm : float
            The L2 norm of w_normalized.
        alpha : float
            The alpha parameter for selective regression.
        kappa : float
            The kappa parameter for selective regression.
        
        Returns
        -------
        numpy array of shape (n_samples,)
            The output of selective regression for each sample.
        """
        kappa = kappa + np.sign(kappa) * self.eps
        angle = X_normalized @ w_normalized
        return (
            X_norm ** alpha
            * 
            w_norm 
            * 
            (np.exp(kappa * angle) - 1) 
            / kappa
        )
        
    def _model(self, theta, X, X_norm, a, k):
        """
        Computes the predicted output of the model for a given set of input features and parameters.
        
        Parameters
        ----------
        theta : numpy array of shape (n_features,)
            The weight vector for the model.
        X : numpy array of shape (n_samples, n_features)
            The input features.
        X_norm : numpy array of shape (n_samples,)
            The L2 norm of each row of X.
        a : float
            The alpha parameter for selective regression.
        k : float
            The kappa parameter for selective regression.
        
        Returns
        -------
        numpy array of shape (n_samples,)
            The predicted output of the model for each sample.
        """
        w = theta[self.n_linear_:]
        w_norm = np.linalg.norm(w)
        if np.isclose(w_norm, 0):
            w_normalized = w
        else:
            w_normalized = w / w_norm
        y_pred = self._selective_regression(
            X[:, self.n_linear_:], X_norm, w_normalized, w_norm, a, k
        ) + X[:, :self.n_linear_] @ theta[:self.n_linear_]
        return y_pred

    def _residuals(self, theta, X, X_norm, y, a, k, sample_weight):
        """
        Calculate residuals for the given theta.

        Parameters
        ----------
        theta : array-like of shape (n_features,)
            Coefficients of the linear regression model.
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        X_norm : array-like of shape (n_samples,)
            Norm of the feature vectors.
        y : array-like of shape (n_samples,)
            Target values.
        a : float
            Alpha parameter for the selective regression.
        k : float
            Kappa parameter for the selective regression.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        residuals : array-like of shape (n_samples,)
            Residuals of the model.
        """
        y_pred = self._model(theta, X, X_norm, a, k)
        return np.sqrt(sample_weight) * (y - y_pred)
    
    def _check_and_assign_params(self):
        """
        Checks the parameters and assigns default values if needed.
        """
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
        """
        Transforms the feature matrix by normalizing the non-linear features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        return_nonnormalized : bool, default=False
            Whether to return the feature matrix without normalization.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            Transformed feature matrix.
        xnorm : array-like of shape (n_samples,)
            Norm of the feature vectors.
        """
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
        """
        Fit the model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        
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
            r2s.append(r2_score(y, test_y, sample_weight=sample_weight))
            params.append([result.x, a, k])
            
        self.r2s_ = np.array(r2s)
        self.params_ = params
        
        self.coef_, self.alpha_, self.kappa_ = params[np.argmax(r2s)]
        
    def predict(self, X):
        """
        Predicts the target values for the input data `X`.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values for the input data.
        """
        X, X_norm = self._transformX(X)
        return self._model(self.coef_, X, X_norm, self.alpha_, self.kappa_)
        