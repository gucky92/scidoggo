"""
Nln chromatic encoding model
"""

import torch
import pyro.distributions as dist

from ..chromatic import (
    ChromaticEncodingModel, NoiseThresholdedPrAdaptation,
    FlyStavenga1993SensitivityModel, 
    FlyStavenga1993InnerSensitivityModel
)
from ..lnl import LnlModel
from ..weights import WeightFunc
from ..gaussian import GaussianObsEncodingModel
from ..utils import FLOAT_TYPE, identity


def construct_nln_chromatic(
    wls,  
    connectivity, 
    nonlin=identity,
    weight_prior=dist.Normal(0, 1),
    data_normalize='l1-norm'
):
    """
    connectivity : (pre x post)
    """
    weight_func = WeightFunc(connectivity, prior=weight_prior)
    lnl_model = LnlModel(weight_func, nonlin=nonlin)
    e_model = GaussianObsEncodingModel(lnl_model, normalize=data_normalize)

    p_func = NoiseThresholdedPrAdaptation()

    if connectivity.shape[0] == 4:
        s_func = FlyStavenga1993InnerSensitivityModel(wls)
    else:
        s_func = FlyStavenga1993SensitivityModel(wls)

    model = ChromaticEncodingModel(
        wls, 
        s_func, 
        e_model, 
        p_func=p_func
    )
    return model