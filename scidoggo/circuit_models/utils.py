"""
Pyro utility modules
"""

import numpy as np
import torch
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist


FLOAT_TYPE = torch.float32


def identity(x, *args, **kwargs):
    return x


def grouped_est(x, labels, estimator=torch.mean, nlabels=None, default=np.nan):
    """
    Calculate estimate along last axis according to labels
    """
    xs = []
    if nlabels is None:
        ulabels = np.sort(np.unique(labels))
    else:
        ulabels = np.arange(nlabels)
    for ulabel in ulabels:
        lbool = labels == ulabel
        if np.all(~lbool):
            ix = torch.ones(x.shape[:-1], dtype=FLOAT_TYPE) * default
        else:
            ix = estimator(x[..., lbool], axis=-1)
        xs.append(ix)
    return torch.stack(xs, axis=-1)


class Tanh(PyroModule):
    """
    Tanh-like function

    Parameters
    ----------
    r0 : PyroSample, PyroParam, torch.tensor, or float
        zero point relative to x
    a : PyroSample or PyroParam, torch.tensor, or float
        scaling of function along y
    b : PyroSample or PyroParam, torch.tensor, or float
        scaling of function along x
    """

    def __init__(
        self,
        r0=PyroSample(dist.Normal(0, 0.2)),
        a=PyroSample(dist.LogNormal(0, 0.5)),
        b=PyroSample(dist.LogNormal(0, 0.5)),
        bias=0
    ):
        super().__init__()
        self.r0 = r0
        self.a = a
        self.b = b
        self.bias = bias

    def forward(self, x, r0=None, a=None, b=None, bias=None):
        if r0 is None:
            r0 = self.r0
        if not isinstance(r0, torch.Tensor):
            r0 = torch.tensor(r0, dtype=FLOAT_TYPE)
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if bias is None:
            bias = self.bias

        # for correct broadcasting
        y = torch.tanh(b * x - r0) + torch.tanh(r0)
        return y * a + bias


class TanhLike(PyroModule):
    """
    Tanh-like function with parameters

    Parameters
    ----------
    r0 : PyroSample, PyroParam, torch.tensor, or float
        zero point relative to x
    a : PyroSample or PyroParam, torch.tensor, or float
        scaling of function along y
    b : PyroSample or PyroParam, torch.tensor, or float
        scaling of function along x
    """

    def __init__(
        self,
        r0=PyroSample(dist.Normal(0, 0.2)),
        a=PyroSample(dist.LogNormal(0, 0.5)),
        b=PyroSample(dist.LogNormal(0, 0.5)),
        bias=0
    ):
        super().__init__()
        self.r0 = r0
        self.a = a
        self.b = b
        self.bias = bias

    def forward(self, x):
        r0 = self.r0
        a = self.a
        b = self.b
        bias = self.bias

        # for correct broadcasting
        y1 = torch.tanh(b * x / (1 - r0)) * (1 - r0)
        y2 = torch.tanh(b * x / (1 + r0)) * (1 + r0)

        x = x.expand(y1.shape)
        y = torch.zeros_like(x)

        # parameterized non-linearity
        y[x <= 0] = y1[x <= 0]
        y[x > 0] = y2[x > 0]
        return y * a + bias


class HillEquation(PyroModule):

    def __init__(
        self, 
        ka=PyroSample(dist.LogNormal(0, 1)), 
        n=1, 
        amax=1, 
        offset=0, 
    ):
        super().__init__()
        self.ka = ka
        self.n = n
        self.amax = amax
        self.offset = offset

    def forward(self, x):
        y = 1 / (1 + (self.ka / x)**self.n)
        y = self.amax * (y + self.offset)
        return y

class Scalar(PyroModule):
    """
    Scaling function
    """

    def __init__(
        self,
        a=PyroSample(dist.LogNormal(0, 1.0))
    ):
        super().__init__()
        self.a = a

    def forward(self, x):
        return self.a * x


def get_ommatidia_separation(z=1e-4):
    """
    Separation Matrix of pale and yellow opsins
    """
    return torch.tensor([
        [1, 1, 1, 1, 1],  # rh1
        [1, 1, z, 1, z],  # rh3
        [1, z, 1, z, 1],  # rh4
        [1, 1, z, 1, z],  # rh5
        [1, z, 1, z, 1]  # rh6
    ], dtype=FLOAT_TYPE)


def get_pr_signs():
    return torch.tensor([
        [-1, -1, -1, -1],
        [1, -1, -1, -1],
        [-1, 1, -1, -1],
        [-1, -1, 1, -1],
        [-1, -1, -1, 1]
    ], dtype=FLOAT_TYPE)


def get_dirichlet_adaptation_prior(certainty=1000., off=1e-3, z=1e-4):
    """
    Get Dirichlet Adaptation prior
    """
    eye = torch.eye(5, dtype=FLOAT_TYPE)
    eye[eye == 0] = off  # adapt prior influence
    return PyroSample(
        dist.Dirichlet(
            certainty * eye * get_ommatidia_separation(z)
        ).to_event(1)
    )


def wrap_pyrosample_as_func(obj):

    class SamplePrior(PyroModule):
        def __init__(self):
            super().__init__()
            self.sample = obj

        def forward(self):
            return self.sample

    return SamplePrior()


class DirichletAdaptationPrior(PyroModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.prior = get_dirichlet_adaptation_prior(*args, **kwargs)

    def forward(self, q, qb):
        return q @ self.prior


class FlyAdaptationPrior(PyroModule):
    """
    Adaptation Prior for fruit fly.

    Gap-junctional assumption: Adaptation is the same in both directions
    """

    def __init__(
        self,
        # priors should draw from a distribution bounded between 0 and 1
        r1to6_with_r8=PyroSample(dist.Beta(1, 3)),
        r1to6_with_r7=0.0,
        r7_with_r8=PyroSample(dist.Beta(1, 3)),
        nonlin=identity,
        scale_by_bg=True,
        include_r1to6=True
    ):
        super().__init__()
        self.r1to6_with_r7 = r1to6_with_r7
        self.r1to6_with_r8 = r1to6_with_r8
        self.r7_with_r8 = r7_with_r8
        self.nonlin = nonlin
        self.scale_by_bg = scale_by_bg
        self.include_r1to6 = include_r1to6

    def forward(self, q, qb):
        assert q.shape == qb.shape, f"Shape mismatch: {q.shape} != {qb.shape}"
        # rh1 not separated by pale and yellow
        joined_rh1 = q.shape[-1] == 5

        if joined_rh1:
            # add rh1 twice
            q = torch.stack([q[..., :1], q], dim=-1)
            qb = torch.stack([qb[..., :1], qb], dim=-1)
        elif q.shape[-1] != 6:
            raise ValueError(f"Number of opsins of `q`: {q.shape[-1]}")

        q = q[..., None, :]
        qb = qb[..., None, :]
        X = torch.eye(6, dtype=FLOAT_TYPE)
        q, qb, X = torch.broadcast_tensors(q, qb, X)

        pale_idx = [0, 2, 4]
        yellow_idx = [1, 3, 5]
        r1to6_idx = [0, 1]
        r8_idx = [4, 5]
        r7_idx = [2, 3]

        if self.scale_by_bg:
            qb_totals = torch.stack(
                [qb[..., pale_idx].sum(-1), qb[..., yellow_idx].sum(-1)],
                dim=-1
            )
            X[..., pale_idx, pale_idx] /= qb_totals[..., :1]
            X[..., yellow_idx, yellow_idx] /= qb_totals[..., 1:]

        x1 = self.r1to6_with_r8
        # scales by the overall background -> e.g. stronger background
        # => less adaptation from other photoreceptors
        if self.scale_by_bg:
            x1 = x1 / qb_totals
        X[..., r1to6_idx, r8_idx] = x1
        X[..., r8_idx, r1to6_idx] = x1

        x2 = self.r1to6_with_r7
        if self.scale_by_bg:
            x2 = x2 / qb_totals
        X[..., r1to6_idx, r7_idx] = x2
        X[..., r7_idx, r1to6_idx] = x2

        x3 = self.r7_with_r8
        if self.scale_by_bg:
            x3 = x3 / qb_totals
        X[..., r7_idx, r8_idx] = x3
        X[..., r8_idx, r7_idx] = x3

        qa = self.nonlin((q * X).sum(-1))
        if self.include_r1to6:
            if joined_rh1:
                return torch.stack([
                    qa[..., r1to6_idx].mean(axis=-1, keepdims=True),
                    qa[..., r7_idx+r8_idx]
                ], dim=-1)
            return qa
        else:
            return qa[..., 2:]
