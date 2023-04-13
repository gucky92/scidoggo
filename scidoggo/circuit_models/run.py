"""
running fitting procedure
"""

import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import Predictive
import pandas as pd
from tqdm import tqdm


def fit_model(
    model,
    *args,
    n_steps=1000,
    clear=True,
    auto_guide=AutoDelta,
    guide=None,
    adam_params={"lr": 0.01},
    print_step=100,
    **kwargs
):
    """
    Fit Pyro Module
    """
    if clear:
        pyro.clear_param_store()

    if guide is None:
        guide = auto_guide(model)
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for _ in tqdm(range(n_steps)):
        loss = svi.step(*args, **kwargs)
        losses.append(loss)
        # if step % print_step == 0:
        #     print(f'{loss}-', end='')

    return {
        "losses": losses,
        "guide": guide,
        "optimizer": optimizer,
        "svi": svi,
        "model": model
    }


def summary(samples, ci=0.95):
    ci = (1 - ci) / 2
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0).detach().numpy(),
            "std": torch.std(v, 0).detach().numpy(),
            "lower": v.kthvalue(int(len(v) * ci), dim=0)[0].detach().numpy(),
            "upper": v.kthvalue(int(len(v) * 1-ci), dim=0)[0].detach().numpy(),
        }
    return site_stats


def predict_model(
    model, guide=None, *args,
    return_sites=(),
    num_samples=100, ci=0.95,
    posterior_samples=None,
    **kwargs
):
    predictive = Predictive(
        model, posterior_samples, 
        guide=guide, num_samples=num_samples,
        return_sites=return_sites)
    samples = predictive(*args, **kwargs)
    pred_summary = summary(samples, ci=ci)
    return pd.DataFrame(pred_summary)
