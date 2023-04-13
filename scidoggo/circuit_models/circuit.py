"""
Circuit modeling
"""

import warnings
import torch

import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist

from .utils import FLOAT_TYPE, identity
from .gaussian import FixedValueModel, GaussianObsModel, UninformativePriorModel, FixedScalePriorModel
from .weights import WeightFunc


class ConductanceBasedIntegration(PyroModule):

    def __init__(
        self,
        reversal=0, 
        sign=1,
        reversal_idcs=None
    ):
        super().__init__()
        self.reversal = reversal
        self.sign = sign
        self.reversal_idcs = reversal_idcs

    def forward(self, X, W, Y=None, reversal=None):
        """
        X : (..., n_inputs)
        W : (..., n_inputs, n_outputs)
        Y : (..., n_outputs)
        """
        if reversal is None:
            reversal = self.reversal * self.sign
            if self.reversal_idcs is not None:
                reversal = reversal[self.reversal_idcs]
            # reversal (n_inputs, n_outputs)

        if Y is None:
            Y = X

        return (X[..., None] * W * (Y[..., None, :] - reversal)).sum(axis=-2)


def matmul(a, b, *args, **kwargs):
    return torch.matmul(a, b)


def multsum(a, b, *args, **kwargs):
    return (a[..., None] * b).sum(axis=-2)


class CircuitModel(PyroModule):

    def __init__(
        self, 
        recurrent_weights,  # weight function
        input_weights,  # weight function
        recurrent_obs, 
        input_obs=identity,
        recurrent_prior=identity, 
        input_prior=identity, 
        recurrent_func=matmul, 
        input_func=matmul, 
        combine_func=torch.add,
        nonlin=identity, 
        inv_tau=1, 
        inv_tau_idcs=None,
        gain=1, 
        gain_idcs=None,
        offset=0, 
        offset_idcs=None,
        tsteps=10, 
        recenter=False,
        center_weighting=False,
        eps=1e-4
    ):  
        super().__init__()
        self.recurrent_weights = recurrent_weights
        self.input_weights = input_weights
        self.input_prior = input_prior
        self.recurrent_prior = recurrent_prior
        self.recurrent_func = recurrent_func
        self.input_func = input_func
        self.inv_tau = inv_tau
        self.inv_tau_idcs = inv_tau_idcs
        self.gain = gain
        self.gain_idcs = gain_idcs
        self.offset = offset
        self.offset_idcs = offset_idcs
        self.tsteps = tsteps
        self.nonlin = nonlin
        self.input_obs = input_obs
        self.recurrent_obs = recurrent_obs
        self.combine_func = combine_func
        self.eps = eps
        self.recenter = recenter 
        self.center_weighting = center_weighting

        if isinstance(inv_tau, (float, int)) and isinstance(gain, (float, int)):
            self.static = inv_tau == gain == 1
        else:
            self.static = False

    def forward(
        self, x, y, x_labels=None, y_labels=None, 
        iw_kws={}, rw_kws={}, 
        x_nlabels=None, y_nlabels=None, 
        idcs=None, 
        offset=None, 
        inv_tau=None, 
        gain=None,
        **nonlin_kwargs
    ):
        """
        Forward method.

        Parameters
        ----------
        x : torch.tensor (time x samples x inputs)
        y : torch.tensor (time x samples x neurons)
        x_labels : numpy.ndarray (inputs)
        y_labels : numpy.ndarray (neurons)
        iw_kws/rw_kws : dict
        nlabels : int
        idcs : numpy.ndarray (samples)
        """
        # idcs should give groupings/conditions
        # if offset, gain, or inv_tau, or any weights differ
        # what if offset, gain or inv_tau also depend on some other inputs
        assert idcs is None, 'Not implemented!'
        
        if inv_tau is None:
            # need to index x, y, x0, y0, xts
            inv_tau = self.inv_tau
            # allow for shared taus
            if self.inv_tau_idcs is not None:
                inv_tau = inv_tau[self.inv_tau_idcs]
        
        if gain is None:
            gain = self.gain
            # allow for shared gains
            if self.gain_idcs is not None:
                gain = gain[self.gain_idcs]

        if offset is None:
            offset = self.offset
            # allow for shared offsets
            if self.offset_idcs is not None:
                offset = offset[self.offset_idcs]

        static = self.static

        assert x.ndim == y.ndim, 'dimensionality mismatch'
        assert x.shape[:-1] == y.shape[:-1], 'shape mismatch'
        if x.ndim == 2:
            temporal = False
            tsteps = self.tsteps
            x = x.expand(tsteps, *x.shape)
            y = y.expand(tsteps, *y.shape)
        else:
            temporal = True
            tsteps = x.shape[0]

        # get weights
        Wi = self.input_weights(**iw_kws)
        Wr = self.recurrent_weights(**rw_kws)

        # get input priors across time
        xts = self.input_prior(x, labels=x_labels, nlabels=x_nlabels)
        if xts.ndim == 2:
            xt_ = xts
            assert xts.shape[:-1] == x.shape[1:-1], 'shape mismatch for input prior'
            
            # compare observed inputs to prior
            if not temporal:
                self.input_obs(
                    x[0],  # assume x was passed as 1-d
                    xts, 
                    labels=x_labels
                )
            else:
                self.input_obs(
                    x, 
                    xts.expand(*x.shape), 
                    labels=x_labels
                )
        
        else:
            xt_ = xts[0]

            # compare observed inputs to prior
            self.input_obs(
                x, 
                xts, 
                labels=x_labels
            )

            assert xts.shape[:-1] == x.shape[:-1], 'shape mismatch for input prior'
        
        # get prior for recurrent neurons
        y0 = self.recurrent_prior(
            y, labels=y_labels, nlabels=y_nlabels
        ) + offset # return samples x neurons
        # add offset as it is the leaky reversal potential
        if y0.ndim == 3:
            if y0.shape[0] > 1:
                warnings.warn(
                    "recurrent prior returns 3D tensor "
                    "that won't be used as is"
                )
            # just use first element of 3D tensor
            y0 = y0[0]
        assert y0.shape[:-1] == y.shape[1:-1], 'shape mismatch for recurrent prior'
        yt_ = y0

        # create empty tensor to fill
        # ypreds = torch.zeros((tsteps,)+y0.shape, dtype=FLOAT_TYPE)
        ypreds = []
        
        for t in range(tsteps):
            # if center weighting - effective change will be zero if input at offset instead of at 0
            if self.center_weighting:
                yin = yt_ - offset
            else:
                yin = yt_
            # calculate time step
            rhs = self.nonlin(
                self.combine_func(
                    self.input_func(xt_, Wi, yt_),  # add yt_ for conductance based models
                    self.recurrent_func(yin, Wr, yt_)  # add yt_ for conductance based models
                ), **nonlin_kwargs
            )
            if static:
                yt_ = rhs + offset
            else:
                yt_ = (- gain * (yt_ - offset) + rhs) * (inv_tau + self.eps) + yt_

            # ypreds[t] = yt_
            ypreds.append(yt_)

            # if using the identity function this basically does nothing
            if xts.ndim == 3:
                xt_ = xts[t]

        ypreds = torch.stack(ypreds)
        if self.recenter:
            # recenter to initial condition
            ypreds = ypreds - y0

        # compare observations in circuit to predictions
        self.recurrent_obs(
            y, 
            ypreds, 
            y0.expand(*ypreds.shape), 
            labels=y_labels
        )


class ObservationModel(PyroModule):
    """
    Classic Paccman observation model
    """

    def __init__(self, obs_scale=1, latent_scale=1, normalize=None, latent=False):
        super().__init__()
        self.obs_scale = obs_scale
        self.latent_scale = latent_scale
        self.latent = latent
        self.normalize = normalize

        self.obs_model = GaussianObsModel(self.obs_scale, normalize=self.normalize)
        if self.latent:
            self.latent_model = GaussianObsModel(self.latent_scale, normalize=None)

    def forward(self, xobs, xpred, xlatent=None, labels=None):
        self.obs_model(xobs, xpred, labels=labels)
        if self.latent and xlatent is not None:
            self.latent_model(xpred, xlatent)



class StandardDynamicCircuitModel(CircuitModel):
    """
    * Fixed input
    * some nonlinearity
    * linear integration
    * simple weight function based on prior distributions
    * known connectivity
    * possible data normalization
    """

    def __init__(
        self,
        recurrent_connectivity,
        input_connectivity,
        recurrent_wprior=dist.Beta(1,1), 
        input_wprior=dist.Beta(1,1),
        prior_distr=dist.Normal(0,1),
        obs_scale=1, 
        nonlin=identity, 
        normalize=None,
        fixed_winput=False, 
        fixed_wrecurrent=False, 
        latent=False,
        **kwargs
    ):
        super().__init__(
            recurrent_weights=WeightFunc(recurrent_connectivity, recurrent_wprior, fixed=fixed_wrecurrent), 
            input_weights=WeightFunc(input_connectivity, input_wprior, fixed=fixed_winput), 
            recurrent_obs=ObservationModel(obs_scale, normalize=normalize, latent=latent), 
            recurrent_prior=(
                FixedValueModel(prior_distr, ndim=2)
                if isinstance(prior_distr, (float, int)) else
                UninformativePriorModel(prior_distr, ndim=2)
            ), 
            nonlin=nonlin, 
            **kwargs
        ) 


class ObservedInputDynamicCircuitModel(CircuitModel):
    """
    * Observed noisy input
    * some nonlinearity
    * linear integration
    * simple weight function based on prior distributions
    * known connectivity
    * possible data normalization
    """

    def __init__(
        self,
        recurrent_connectivity,
        input_connectivity,
        recurrent_wprior=dist.Beta(1,1), 
        input_wprior=dist.Beta(1,1),
        prior_distr=dist.Normal(0,1),
        input_prior_scale=1,
        input_prior_distr='normal',
        obs_scale=1, 
        input_obs_scale=1,
        nonlin=identity, 
        normalize=None,
        input_normalize=None, 
        fixed_winput=False, 
        fixed_wrecurrent=False, 
        latent=False,
        **kwargs
    ):
        super().__init__(
            recurrent_weights=WeightFunc(recurrent_connectivity, recurrent_wprior, fixed=fixed_wrecurrent), 
            input_weights=WeightFunc(input_connectivity, input_wprior, fixed=fixed_winput), 
            recurrent_obs=ObservationModel(obs_scale, normalize=normalize, latent=latent), 
            recurrent_prior=UninformativePriorModel(prior_distr, ndim=2), 
            input_prior=FixedScalePriorModel(input_prior_scale, input_prior_distr),
            input_obs=ObservationModel(input_obs_scale, normalize=input_normalize),
            nonlin=nonlin, 
            **kwargs
        )


class UnobservedInputDynamicCircuitModel(CircuitModel):
    """
    * unknown input
    * some nonlinearity
    * linear integration
    * simple weight function based on prior distributions
    * known connectivity
    * possible data normalization
    """

    def __init__(
        self,
        recurrent_connectivity,
        input_connectivity,
        recurrent_wprior=dist.Beta(1,1), 
        input_wprior=dist.Beta(1,1),
        prior_distr=dist.Normal(0,1),
        input_prior_scale=1,
        input_prior_distr='normal',
        obs_scale=1, 
        input_obs_scale=1,
        nonlin=identity, 
        normalize=None,
        input_normalize=None, 
        fixed_winput=False, 
        fixed_wrecurrent=False, 
        latent=False,
        **kwargs
    ):
        super().__init__(
            recurrent_weights=WeightFunc(recurrent_connectivity, recurrent_wprior, fixed=fixed_wrecurrent), 
            input_weights=WeightFunc(input_connectivity, input_wprior, fixed=fixed_winput), 
            recurrent_obs=ObservationModel(obs_scale, normalize=normalize, latent=latent), 
            recurrent_prior=UninformativePriorModel(prior_distr, ndim=2), 
            input_prior=FixedScalePriorModel(input_prior_scale, input_prior_distr),
            input_obs=ObservationModel(input_obs_scale, normalize=input_normalize),
            nonlin=nonlin, 
            **kwargs
        )