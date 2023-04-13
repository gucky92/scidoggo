"""
Weight Sampling functions
"""

import torch
import numpy as np

import pyro
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist

from .utils import FLOAT_TYPE
from .deep import MonotonicNonlinearity


class WeightMixin:

    def normalize_weights(self, weights):
        normalize = self.normalize
        if normalize is None:
            pass
        elif normalize == 'l1-norm':
            weights = weights / torch.sum(torch.abs(weights), axis=-2, keepdims=True)
        elif normalize == 'l2-norm':
            weights = weights / torch.sqrt(torch.sum(weights**2, axis=-2, keepdims=True))
        elif normalize == 'max':
            weights = weights / torch.max(torch.abs(weights), axis=-2, keepdims=True)
        else:
            raise NameError(f"normalize parameter `{normalize}` unknown.")
        return weights


class FixedWeightFunc(PyroModule, WeightMixin):
    """
    Fixed weight function
    """

    def __init__(self, connectivity, offset=None, normalize=None):
        super().__init__()

        if not isinstance(connectivity, torch.Tensor):
            connectivity = torch.tensor(connectivity, dtype=FLOAT_TYPE)

        self.connectivity = connectivity
        self.offset = offset
        self.normalize = normalize

        self.weights = self.normalize_weights(self.connectivity)

    def forward(self):
        weights = self.weights
        pyro.deterministic(
            self._pyro_get_fullname("W"),
            weights
        )
        if self.offset is None:
            return weights
        else:
            return weights + self.offset


class WeightFunc(PyroModule, WeightMixin):
    """
    Function that returns weights (pre x post) -- (input x output)
    """

    def __init__(
        self,
        connectivity,
        prior=dist.Beta(1, 1),
        offset=None,
        normalize=None,
        fixed=False,
        **prior_kws
    ):
        super().__init__()

        if not isinstance(connectivity, torch.Tensor):
            connectivity = torch.tensor(connectivity, dtype=FLOAT_TYPE)
        if not isinstance(prior, (torch.Tensor, dist.Distribution)):
            prior = torch.tensor(prior, dtype=FLOAT_TYPE)

        self.connectivity = connectivity
        self.prior = prior
        self.offset = offset
        self.normalize = normalize
        self.fixed = fixed

        self.nonzero = connectivity != 0
        self.n_nonzero = self.nonzero.sum()
        if isinstance(prior, dist.Distribution):
            self.weights = PyroSample(
                self.prior.expand((self.n_nonzero,)).to_event(1),
                **prior_kws
            )
        elif fixed:
            self.weights = self.prior.expand((self.n_nonzero,))
        else:
            self.weights = PyroParam(
                self.prior.expand((self.n_nonzero,)),
                **prior_kws
            )

    def forward(self, *args, **kwargs):
        weights = torch.zeros_like(self.connectivity)
        weights[self.nonzero] = self.weights  # change somehow
        weights = self.normalize_weights(weights)

        # multiple by "sign"
        weights = weights * self.connectivity  # connectivity can have signs
        if self.offset is not None:
            weights = weights + self.offset
        pyro.deterministic(
            self._pyro_get_fullname("W"),
            weights
        )
        return weights


class InnerCircuitWeight(PyroModule):

    def __init__(
        self,
        prs=PyroSample(dist.Dirichlet(2*torch.tensor([[0.5, 0.5]]*4, dtype=FLOAT_TYPE))),
        dm9=PyroSample(dist.Dirichlet(4*torch.tensor([0.2, 0.2, 0.4, 0.4], dtype=FLOAT_TYPE)))
    ):
        super().__init__()
        self.prs = prs
        self.dm9 = dm9

        self.direct = torch.tensor(np.roll(np.eye(4), 2, axis=1), dtype=FLOAT_TYPE) != 0

    def forward(self, *args, **kwargs):
        w = self.dm9[:, None] * self.prs[None, :, 1]
        w[self.direct] += self.prs[:, 0]
        return w


class FuncWeight(WeightFunc):

    def __init__(
        self, connectivity,
        nonlin,
        prior=dist.Beta(1, 1),
        offset=None,
        normalize=None,
        fixed=False,
        **prior_kws,
    ):
        super().__init__(
            connectivity=connectivity,
            prior=prior, offset=offset, normalize=normalize,
            fixed=fixed,
            **prior_kws
        )
        self.nonlin = nonlin

    def forward(self, x):
        x = self.nonlin(x)
        w = super().forward(x)
        if x.ndim == 1:
            x = x[..., None, None]
        elif x.ndim == 2:
            x = x[..., None]
        return x * w  # x.shape[-1], presynaptic, postsynaptic


class MonotonicWeight(WeightFunc):

    def __init__(
        self, connectivity, prior=dist.Beta(1, 1), normalize=None,
        weight_kwargs={}, output_nonlin=torch.tanh, **kwargs
    ):
        # output_dim is the weight presynaptic dimension
        super().__init__(
            connectivity,
            prior=prior,
            normalize=normalize,
            **weight_kwargs
        )
        self.nonlin = MonotonicNonlinearity(
            output_dim=self.connectivity.shape[0],
            output_nonlin=output_nonlin,
            **kwargs
        )

    def forward(self, x):
        x = self.nonlin(x)  #
        w = super().forward()
        return x[..., None] * w  # x.shape[:-1], presynaptic, postsynaptic
