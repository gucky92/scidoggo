"""
Adaptation Functions
"""

import torch

from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist


from .utils import identity, FLOAT_TYPE, get_dirichlet_adaptation_prior


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
