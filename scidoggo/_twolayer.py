"""Two layer encoding network
"""

from torch import cosine_similarity, nn
from torch.nn.functional import normalize
import torch
import skorch
import numpy as np
from scipy.optimize import nnls, curve_fit
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression


NONLINS = {
    "tanh": nn.Tanh, 
    "identity": nn.Identity,  
    "relu": nn.ReLU
}


def optimal_input_scaling(X, Y, mask_zero=False, nonneg=False):
    # TODO efficiency?
    # X (n_samples x n_dim), Y (n_samples x n_neurons)
    if not mask_zero:
        linear_model = LinearRegression(fit_intercept=False, copy_X=False, positive=nonneg)
        return linear_model.fit(X, Y).coef_.T
    
    W = np.zeros((X.shape[1], Y.shape[1]))
    for idx, y_ in enumerate(Y.T):
        nonzero = y_ != 0
        if not np.any(nonzero):
            continue
        y_ = y_[nonzero]
        X_ = X[nonzero]
            
        if nonneg:
            W[:, idx] = nnls(X_, y_)[0]
        else:
            W[:, idx] = lstsq(X_, y_)[0]
    return W


def scale_threshold_function(ypred, a=1, ymin=None, ymax=None):
    ypred = a * ypred
    if ymin is not None:
        ypred[ypred < a*ymin] = a*ymin
    if ymax is not None:
        ypred[ypred > a*ymax] = a*ymax
    return ypred


def optimal_scale_and_thresholding(
    X, Y, mask_zero=False, nonneg=False, 
    fit_ymin=True, fit_ymax=True, 
):
    # scaling and thresholding
    W = optimal_input_scaling(X, Y, mask_zero=mask_zero, nonneg=nonneg)
    if not fit_ymin and not fit_ymax:
        return W, 1, -np.inf, np.inf
    
    Ypred = X @ W
    
    if not fit_ymin:
        curve_function = lambda ypred, a, ymax : scale_threshold_function(
            ypred, a=a, ymax=ymax
        )
    else:
        curve_function = scale_threshold_function
        
    scalings = np.ones(Y.shape[1])
    lbs = scalings * -np.inf
    ubs = scalings * np.inf
    
    for idx, (ypred, y) in enumerate(zip(Ypred.T, Y.T)):
        if mask_zero:
            nonzero = y != 0
            if not np.any(nonzero):
                continue
            y = y[nonzero]
            ypred = ypred[nonzero]
            
        p0 = [1.0,]
        lb = [0.0,]
        ub = [np.inf,]
        
        ymin = np.min([y, ypred])
        ymax = np.max([y, ypred])
        
        if fit_ymin:
            p0.append(ymin)
            lb.append(-np.inf)
            ub.append(ymax)
        
        if fit_ymax:
            p0.append(ymax)
            lb.append(ymin)
            ub.append(np.inf)
        
        popt, _ = curve_fit(
            curve_function, 
            ypred, y, p0=p0, 
            bounds=(lb, ub)
        )
        
        scalings[idx] = popt[0]
        if fit_ymin:
            lbs[idx] = popt[1]
            if fit_ymax:
                ubs[idx] = popt[2]
        else:
            ubs[idx] = popt[1]
          
    return W, scalings, lbs, ubs


### -- Loss functions


class ScaledMSE(nn.MSELoss):
    
    def __init__(
        self, *args, mask_zero=False, nonneg=False, 
        alpha=0, 
        fit_ymin=False, fit_ymax=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_zero = mask_zero
        self.nonneg = nonneg
        self.alpha = alpha
        self.fit_ymin = fit_ymin
        self.fit_ymax = fit_ymax
    
    def forward(self, input, target):
        # input (n_samples x n_dim), target (n_samples x n_neurons)
        # assuming the last dimensions should just best a linear combinations
        ndim = input.shape[1]
        if self.alpha and ndim > 1:
            # TODO efficiency?
            sims = cosine_similarity(input[:, None, :], input[:, :, None], dim=0)
            # must be orthogonal!
            sims = torch.abs(sims)
            # sum up all pairwise cosine similarities
            reg = torch.triu(sims, diagonal=1).sum()
        else:
            reg = 0
        
        # no gradients here - this is adaptively adjusted
        if not self.fit_ymin and not self.fit_ymax:
            W = optimal_input_scaling(
                input.detach().numpy(), 
                target.detach().numpy(), 
                mask_zero=self.mask_zero, 
                nonneg=self.nonneg
            )
            W = torch.tensor(W, dtype=torch.float32)
            input = input @ W
        else:
            W, scalings, lbs, ubs = optimal_scale_and_thresholding(
                input.detach().numpy(), 
                target.detach().numpy(), 
                mask_zero=self.mask_zero, 
                nonneg=self.nonneg, 
                fit_ymin=self.fit_ymin, 
                fit_ymax=self.fit_ymax
            )
            W = torch.tensor(W, dtype=torch.float32)
            scalings = torch.tensor(scalings, dtype=torch.float32)
            lbs = torch.tensor(lbs, dtype=torch.float32)
            ubs = torch.tensor(ubs, dtype=torch.float32)
            
            input = (input @ W) * scalings
            input = input.clamp(scalings*lbs, scalings*ubs)
        
        if self.mask_zero:
            ybool = (target != 0)
            input = input * ybool.float()
        return super().forward(input, target) + reg


class TwoLayerNoBias(nn.Module):
    
    def __init__(
        self, 
        n_in, n_out,
        n_hidden=1,
        nonlin='relu', 
        output_nonlin='identity',
        linear_weights=None,
        linear_init=False, 
        output_weights=None, 
        output_init=False, 
        linear_norm=False, 
        output_norm=False
    ):
        super().__init__()
        
        self.linear_norm = linear_norm
        self.output_norm = output_norm

        self.linear = nn.Linear(n_in, n_hidden, bias=False)
        self.nonlin = NONLINS[nonlin]()
        
        self.output = nn.Linear(n_hidden, n_out, bias=False)
        self.output_nonlin = NONLINS[output_nonlin]()
        
        self._modify_weights(self.linear.weight, linear_weights, grad=linear_init)

        self._modify_weights(self.output.weight, output_weights, grad=output_init)
        
        self.output_nonlin = NONLINS[output_nonlin]()
        
    @staticmethod
    def _modify_weights(param, weights, grad=True):
        if weights is None:
            pass
        else:
            param.data = torch.tensor(weights, dtype=torch.float32)
            param.requires_grad = grad     
    
    def forward(self, x):
        if self.linear_norm:
            weight = normalize(self.linear.weight, dim=1)
            y = self.nonlin(torch.mm(x, weight.t()))
        else:
            y = self.nonlin(self.linear(x))
        if self.output_norm:
            weight = normalize(self.output.weight, dim=1)
            return self.output_nonlin(torch.mm(y, weight.t()))
        else:
            return self.output_nonlin(self.output(y))
    
    
def create_model(
    module=TwoLayerNoBias, 
    lr=0.001, 
    criterion=ScaledMSE,
    criterion__nonneg=True,
    criterion__reduction='sum',
    criterion__mask_zero=True,
    optimizer__momentum=0.95,
    max_epochs=1000,
    batch_size=50, 
    predict_nonlinearity=None, 
    module__nonlin='relu', 
    train_split=None, 
    verbose=True,
    module__n_in=4, 
    module__n_out=1, 
    module__n_hidden=2, 
    module__output_weights=np.array([[1, -1]]), 
    **kwargs
):
    """Convenience function to create sklearn-type estimator
    """
    return skorch.NeuralNetRegressor(
        module, 
        lr=lr, 
        criterion=criterion, 
        criterion__nonneg=criterion__nonneg,
        criterion__reduction=criterion__reduction,
        criterion__mask_zero=criterion__mask_zero,
        optimizer__momentum=optimizer__momentum,
        max_epochs=max_epochs,
        batch_size=batch_size, 
        predict_nonlinearity=predict_nonlinearity, 
        module__nonlin=module__nonlin, 
        train_split=train_split, 
        verbose=verbose,
        module__n_in=module__n_in, 
        module__n_out=module__n_out, 
        module__n_hidden=module__n_hidden, 
        module__output_weights=module__output_weights, 
        **kwargs
    )
    