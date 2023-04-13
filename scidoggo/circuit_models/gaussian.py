"""
Gaussian observations
"""

import numpy as np
from numpy.core.fromnumeric import mean
import torch

import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist

from .utils import FLOAT_TYPE, grouped_est


class FixedPriorModel(PyroModule):

    def __init__(self, axis=None, method=torch.mean, default=0):
        super().__init__()
        self.axis = axis
        self.method = method
        self.default = default

    def forward(self, x, *args, labels=None, nlabels=None, idcs=None):
        if self.axis is not None:
            x = self.method(x, axis=self.axis)
        if labels is not None:
            x = grouped_est(
                x, labels, estimator=torch.mean, nlabels=nlabels, 
                default=self.default
            )
        if idcs is not None:
            x = x[..., idcs, :]
        return x


class FixedScalePriorModel(FixedPriorModel):
    """
    Sample from a Gaussian distribution.
    """

    def __init__(self, scale=1, distr='normal', **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.distr = distr

    def forward(self, x, *args, labels=None, nlabels=None, idcs=None):
        x = super().forward(x, *args, labels=labels, nlabels=nlabels, idcs=None)
        # sample from gaussian
        if self.distr == 'normal':
            samples = pyro.sample(
                self._pyro_get_fullname("x"), 
                dist.Normal(
                    x, self.scale
                ).to_event(1)
            )
        elif self.distr == 'gamma':
            # positivity constraint
            beta = x / (self.scale ** 2)
            alpha = x * beta
            samples = pyro.sample(
                self._pyro_get_fullname("x"), 
                dist.Gamma(
                    alpha, beta
                ).to_event(1)
            )
        else:
            raise NameError(
                f"distr is `{self.distr}`, "
                "but should be either on of `gamma`, `normal`."
            )

        if idcs is not None:
            samples = samples[..., idcs, :]

        return samples


class UninformativePriorModel(PyroModule):
    """
    Sample from a Gaussian distribution.
    """

    def __init__(self, distr=dist.Normal(0, 1), ndim=None):
        super().__init__()
        self.distr = distr
        self.ndim = ndim

    def forward(self, x, *args, labels=None, nlabels=None, idcs=None):
        shape = x.shape
        if labels is not None:
            if nlabels is None:
                nlabels = len(np.unique(labels))
            shape = (*shape[:-1], nlabels)
        if self.ndim is not None:
            shape = shape[-self.ndim:]
        # sample from gaussian
        samples = pyro.sample(
            self._pyro_get_fullname("x"), 
            self.distr.expand(shape).to_event(1)  # TODO must shape be a list?
        )
        if idcs is not None:
            samples = samples[..., idcs, :]
        return samples


class FixedValueModel(PyroModule):
    """
    Fixed value samples.
    """

    def __init__(self, value=0, ndim=None):
        super().__init__()
        self.ndim = ndim
        self.value = value

    def forward(self, x, *args, labels=None, nlabels=None, idcs=None):
        shape = x.shape
        if labels is not None:
            if nlabels is None:
                nlabels = len(np.unique(labels))
            shape = (*shape[:-1], nlabels)
        if self.ndim is not None:
            shape = shape[-self.ndim:]
        samples = torch.ones(shape) * self.value
        if idcs is not None:
            samples = samples[..., idcs, :]
        return samples


class SampleGaussianModel(PyroModule):
    """
    Sample from a Gaussian distribution.
    """

    def forward(self, y, *args, y_scale=1):
        # sample from gaussian
        with pyro.plate(self._pyro_get_fullname("y_plate"), len(y)):
            return pyro.sample(
                self._pyro_get_fullname("y"), 
                dist.Normal(
                    y, y_scale
                )
            )


class FixedGaussianModel(PyroModule):
    """
    Sample from a fixed Gaussian distribution
    """

    def forward(self, y, *args, y_scale=1):
        return dist.Normal(y, y_scale).sample()


class GaussianObsModel(PyroModule):
    """
    Gaussian Observations of y

    Parameters
    ----------
    y_scale : float
        standard deviation of observations
    normalize : str, optional
        Whether to normalize predictions to the data.
    """

    def __init__(self, y_scale=1, normalize=None):
        super().__init__()
        self.y_scale = y_scale
        self.normalize = normalize

    def forward(self, y, ypred, *args, labels=None):
        """
        forward pass
        """

        y, ypred = self.normalize_ypred(y, ypred, labels)

        # sample from gaussian
        with pyro.plate(self._pyro_get_fullname("y_plate"), len(ypred)):
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.Normal(
                    ypred, self.y_scale
                ),
                obs=y
            )

    def normalize_ypred(self, y, ypred, labels):
        """
        Parameters
        ----------
        y : torch.tensor
            1D or 2D (stimuli x rois)
        ypred : torch.tensor
            1D or 2D (stimuli x neurons)
        labels : torch.tensor
            1D (rois) with integer indices for neurons
        """

        if y.ndim == 1:
            y = y[:, None]

        # cover nans with zeros
        nans = torch.isnan(y)
        if nans.any():
            # WHAT? - This may not work if y is not an actual observation
            # y = y.detach().clone()
            # y[nans] = 0
            # this is better
            y = torch.where(nans, torch.zeros_like(y), y)

        if ypred.ndim == 1:
            ypred = ypred[:, None]
        elif labels is not None:
            ypred = ypred[..., labels]  # same size as y

        assert ypred.shape == y.shape, f"Shape mismatch: {y.shape} != {ypred.shape}"

        if self.normalize is None:
            # save in param store - before reshaping
            pyro.deterministic(
                self._pyro_get_fullname("yobs"),
                y
            )
            pyro.deterministic(
                self._pyro_get_fullname("ypred"),
                ypred
            )
            return y[~nans], ypred[~nans]

        elif self.normalize == 'l1-norm':
            scaling = (
                torch.sum(
                    torch.abs(y), axis=tuple(i for i in range(y.ndim-1))
                )[..., None, :]
                /
                torch.sum(
                    torch.abs(ypred), axis=tuple(i for i in range(y.ndim-1))
                )[..., None, :]
            )

        elif self.normalize == 'l2-norm':
            scaling = (
                torch.sqrt(
                    torch.sum(y ** 2, axis=tuple(i for i in range(y.ndim-1)))
                )[..., None, :]
                /
                torch.sqrt(
                    torch.sum(ypred ** 2, axis=tuple(i for i in range(y.ndim-1)))
                )[..., None, :]
            )

        ypred = ypred * scaling
        # save in param store (scaled) - before reshaping
        pyro.deterministic(
            self._pyro_get_fullname("yobs"),
            y
        )
        pyro.deterministic(
            self._pyro_get_fullname("ypred"),
            ypred
        )
        return y[~nans], ypred[~nans]


class GaussianObsEncodingModel(GaussianObsModel):
    """
    Gaussian Observations of y

    Parameters
    ----------
    e_model : callable
        PyroModule
    y_scale : float
        standard deviation of observations
    """

    def __init__(self, e_model, y_scale=1, normalize=None):
        super().__init__(y_scale=y_scale, normalize=normalize)
        self.e_model = e_model

    def forward(self, X, y, *args, labels=None, **kwargs):
        """
        forward pass
        """

        ypred = self.e_model(X, *args, **kwargs)
        return super().forward(y, ypred, labels)
