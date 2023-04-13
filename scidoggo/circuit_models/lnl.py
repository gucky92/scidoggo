"""
LNL models
"""

import torch

import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist


from .utils import identity
from .weights import WeightFunc, MonotonicWeight, FuncWeight
from .gaussian import GaussianObsEncodingModel


class LnlModel(PyroModule):
    """
    Simple linear-nonlinear encoding model
    """

    def __init__(
        self,
        weight_func,
        nonlin=identity,
        func=torch.matmul
    ):
        super().__init__()
        self.weight_func = weight_func
        self.nonlin = nonlin
        self.func = func

    def forward(self, X, *args, **kwargs):
        W = self.weight_func(*args, **kwargs)
        pyro.deterministic(
            self._pyro_get_fullname("W"),
            W
        )
        ypred = self.func(X, W)
        pyro.deterministic(
            self._pyro_get_fullname("ylinear"),
            ypred
        )
        ypred = self.nonlin(ypred)
        pyro.deterministic(
            self._pyro_get_fullname("ypred"),
            ypred
        )
        return ypred


class DeepLnlModel(LnlModel):
    """
    Deep weights
    """

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_prior=dist.Normal(0, 100),
        nonlin=identity,
        **monotonic_kwargs
    ):
        weight_func = MonotonicWeight(
            torch.ones((input_ndim, output_ndim)),
            prior=weight_prior,
            **monotonic_kwargs
        )
        super().__init__(
            weight_func, nonlin=nonlin,
            func=lambda x, w: (x[..., None] * w).sum(axis=-2)
        )


class FuncLnlModel(LnlModel):
    """
    FuncWeight model
    """

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_nonlin,
        weight_prior=dist.Normal(0, 100),
        weight_offset=None,
        weight_normalize=None,
        nonlin=identity,
        **weight_kwargs
    ):
        weight_func = FuncWeight(
            torch.ones((input_ndim, output_ndim)),
            prior=weight_prior,
            offset=weight_offset,
            nonlin=weight_nonlin,
            normalize=weight_normalize,
            **weight_kwargs
        )
        super().__init__(
            weight_func, nonlin=nonlin,
            func=lambda x, w: (x[..., None] * w).sum(axis=-2)
        )


class SimpleLnlModel(LnlModel):
    """
    Simple LNL model
    """

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_prior=dist.Normal(0, 100),
        weight_offset=None,
        weight_normalize=None,
        nonlin=identity,
        **weight_kws
    ):
        weight_func = WeightFunc(
            torch.ones((input_ndim, output_ndim)),
            prior=weight_prior, offset=weight_offset,
            normalize=weight_normalize,
            **weight_kws
        )
        super().__init__(weight_func, nonlin=nonlin)


class GaussianObsLnlModel(GaussianObsEncodingModel):

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_prior=dist.Normal(0, 100),
        nonlin=identity,
        weight_offset=None,
        y_scale=1, normalize=None,
        weight_normalize=None,
        **weight_kws
    ):
        e_model = SimpleLnlModel(
            input_ndim, output_ndim,
            weight_prior=weight_prior,
            nonlin=nonlin, weight_offset=weight_offset,
            weight_normalize=weight_normalize,
            **weight_kws
        )
        super().__init__(e_model, y_scale=y_scale, normalize=normalize)


class GaussianObsFuncLnlModel(GaussianObsEncodingModel):

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_nonlin,
        weight_prior=dist.Normal(0, 100),
        nonlin=identity,
        weight_offset=None,
        y_scale=1, normalize=None,
        weight_normalize=None,
        **weight_kws
    ):
        e_model = FuncLnlModel(
            input_ndim, output_ndim,
            weight_nonlin=weight_nonlin,
            weight_prior=weight_prior,
            weight_normalize=weight_normalize,
            nonlin=nonlin, weight_offset=weight_offset, **weight_kws
        )
        super().__init__(e_model, y_scale=y_scale, normalize=normalize)


class GaussianObsDeepLnlModel(GaussianObsEncodingModel):

    def __init__(
        self,
        input_ndim,
        output_ndim,
        weight_prior=dist.Normal(0, 100),
        nonlin=identity,
        y_scale=1, normalize=None,
        **monotonic_kwargs
    ):
        e_model = DeepLnlModel(
            input_ndim, output_ndim,
            weight_prior=weight_prior,
            nonlin=nonlin, **monotonic_kwargs
        )
        super().__init__(e_model, y_scale=y_scale, normalize=normalize)
