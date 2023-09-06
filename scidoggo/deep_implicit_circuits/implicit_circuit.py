import torch
from numbers import Number
import numpy as np
import pandas as pd
from typing import Optional, Union
from torch import nn
import torch.distributions as dist
import torch.autograd as autograd
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score

from pytorch_lightning import LightningModule, Trainer


def tanh_like(x, r0=0, a=1, b=1):
    """
    tanh-like function from Rajan, Abbott, Sompolinsky (2010).
    """
    # tanh = getattr(module, 'tanh')
    # zeros_like = getattr(module, 'zeros_like')
    y = torch.zeros_like(x)
    x = b * x
    
    y1 = (1 - r0) * torch.tanh(x / (1 - r0 + 1e-6))
    y2 = (1 + r0) * torch.tanh(x / (1 + r0 + 1e-6))

    y[x <= 0] = y1[x <= 0]
    y[x > 0] = y2[x > 0]

    return a * y     


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    if mask is not None:
        input = input[mask]
        target = target[mask]
        if weight is not None:
            weight = weight[mask]
    if weight is None:
        res_squared = (input - target) ** 2
    else:
        res_squared = weight * (input - target) ** 2
    return res_squared.mean()


def step_forward(
    X: torch.Tensor,  # inputs to recurrent circuit
    Yt: torch.Tensor,  # state of recurrent circuit Y(t-1)
    Wi: torch.Tensor,  # input weights
    Wr: torch.Tensor,  # recurrent weights
    offset: torch.Tensor,  # offset
    gain: torch.Tensor,  # gain
    nonlin: callable,  # nonlinearity
):
    Yt = torch.matmul(X, Wi) + torch.matmul(Yt, Wr)
    Yt = nonlin(gain * Yt - offset) + nonlin(offset)
    return Yt


def matrix_to_weightdict(W: Union[np.ndarray, torch.Tensor, pd.DataFrame, list[list]]):
    if isinstance(W, pd.DataFrame):
        W = W.values
    elif isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()

    weight_dict = {}
    for ipre, values in enumerate(W):
        for ipost, value in enumerate(values):
            weight_dict[(ipre, ipost)] = value

    return weight_dict

def values_to_dict(values: Union[np.ndarray, torch.Tensor, pd.Series, list]):
    if isinstance(values, pd.Series):
        values = values.values
    elif isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    values_dict = {}
    for i, value in enumerate(values):
        values_dict[i] = value

    return values_dict


class WeightDict:
    def __init__(self, weight_dict: dict, shape: tuple):
        self.shape = shape

        fixed_idcs = []
        fixed_jdcs = []
        fixed_value = []
        idcs = []
        jdcs = []
        vidcs = []
        lower = []
        upper = []
        inits = []
        labels = []  # labels for unfixed
        n_params = 0
        for k, v in weight_dict.items():
            # fixed values
            if isinstance(v, Number) and isinstance(k[0], Number):
                fixed_value.append(v)
                fixed_idcs.append(k[0])
                fixed_jdcs.append(k[1])
                continue
            elif isinstance(v, Number):
                fixed_value.extend([v] * len(k[0]))
                fixed_idcs.extend(k[0])
                fixed_jdcs.extend(k[1])
                continue

            if isinstance(k[0], Number):
                idcs.append(k[0])
                jdcs.append(k[1])
                vidcs.append(n_params)

            else:
                idcs.extend(k[0])
                jdcs.extend(k[1])
                vidcs.extend([n_params] * len(k[0]))

            lower.append(v[0])
            upper.append(v[1])
            # add inits
            if len(v) == 3:
                inits.append(v[2])
            else:
                inits.append(np.nan)

            labels.append(k)
            n_params += 1

        self.fixed_idcs = np.array(fixed_idcs)
        self.fixed_jdcs = np.array(fixed_jdcs)
        self.fixed_value = np.array(fixed_value)

        self.idcs = np.array(idcs)
        self.jdcs = np.array(jdcs)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.inits = np.array(inits)
        self.labels = labels
        self.vidcs = np.array(vidcs)

        self.n_params = len(self.lower)

        assert len(idcs) == len(vidcs)

    def get_weights(self, w: torch.Tensor):
        weights = torch.zeros(self.shape).to(w)
        if len(self.fixed_idcs):
            weights[self.fixed_idcs, self.fixed_jdcs] = torch.tensor(
                self.fixed_value
            ).to(w)
        if len(self.idcs):
            weights[self.idcs, self.jdcs] = w[self.vidcs]
        return weights

    def clip_values(self, w: nn.Parameter):
        with torch.no_grad():
            w.clip_(torch.tensor(self.lower).to(w), torch.tensor(self.upper).to(w))
        return w

    def sample_values(self, w: nn.Parameter):
        isnull = np.isnan(self.inits)
        with torch.no_grad():
            w[isnull] = dist.Uniform(
                torch.tensor(self.lower[isnull]).to(w),
                torch.tensor(self.upper[isnull]).to(w),
            ).sample()
            w[~isnull] = torch.tensor(self.inits[~isnull]).to(w)
            return w


class ValuesDict:
    def __init__(self, values_dict, length, default=0.0):
        self.length = length
        self.default = default

        fixed_idcs = []
        fixed_value = []
        idcs = []
        vidcs = []
        lower = []
        upper = []
        inits = []
        labels = []

        n_params = 0
        for k, v in values_dict.items():
            # fixed values
            if isinstance(v, Number) and isinstance(k, Number):
                fixed_value.append(v)
                fixed_idcs.append(k)
                continue
            elif isinstance(v, Number):
                fixed_value.extend([v] * len(k))
                fixed_idcs.extend(k)
                continue

            # non fixed values
            if isinstance(k, Number):
                idcs.append(k)
                vidcs.append(n_params)
            else:
                # same parameter for multiple neurons
                idcs.extend(k)
                vidcs.extend([n_params] * len(k))

            lower.append(v[0])
            upper.append(v[1])
            if len(v) == 3:
                inits.append(v[2])
            else:
                inits.append(np.nan)

            labels.append(k)
            n_params += 1

        assert len(lower) == n_params

        self.fixed_idcs = np.array(fixed_idcs)
        self.fixed_value = np.array(fixed_value)

        self.idcs = np.array(idcs)
        self.vidcs = np.array(vidcs)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.inits = np.array(inits)
        self.labels = labels
        self.n_params = len(self.lower)

        assert len(idcs) == len(vidcs)

    def get_values(self, w: torch.Tensor):
        values = torch.ones(self.length).to(w) * self.default
        if len(self.fixed_idcs):
            values[self.fixed_idcs] = torch.tensor(self.fixed_value).to(w)
        if len(self.idcs):
            values[self.idcs] = w[self.vidcs]
        return values

    def clip_values(self, w: nn.Parameter):
        with torch.no_grad():
            w.clip_(torch.tensor(self.lower).to(w), torch.tensor(self.upper).to(w))
        return w

    def sample_values(self, w: nn.Parameter):
        isnull = np.isnan(self.inits)
        with torch.no_grad():
            w[isnull] = dist.Uniform(
                torch.tensor(self.lower[isnull]).to(w),
                torch.tensor(self.upper[isnull]).to(w),
            ).sample()
            w[~isnull] = torch.tensor(self.inits[~isnull]).to(w)
            return w
        
        
class TanhLike(nn.Module):
    
    def __init__(
        self, length, r0_dict={}, a_dict={}, b_dict={}, 
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.length = length
        self.r0_dict = r0_dict
        self.a_dict = a_dict
        self.b_dict = b_dict
        self.r0_dict_obj = ValuesDict(r0_dict, length, default=0.0)
        self.a_dict_obj = ValuesDict(a_dict, length, default=1.0)
        self.b_dict_obj = ValuesDict(b_dict, length, default=1.0)
        
        factory_kwargs = dict(dtype=dtype, device=device)
        self._r0 = nn.Parameter(
            torch.empty(self.r0_dict_obj.n_params, **factory_kwargs)
        )
        self._a = nn.Parameter(
            torch.empty(self.a_dict_obj.n_params, **factory_kwargs)
        )
        self._b = nn.Parameter(
            torch.empty(self.b_dict_obj.n_params, **factory_kwargs)
        )
        
        self.reset_parameters()
        
        self.num_params = sum(param.numel() for param in self.parameters())
        
    @property
    def r0(self):
        return self.r0_dict_obj.get_values(self._r0)
    
    @property
    def a(self):
        return self.a_dict_obj.get_values(self._a)
    
    @property
    def b(self):
        return self.b_dict_obj.get_values(self._b)
    
    def reset_parameters(self):
        self.r0_dict_obj.sample_values(self._r0)
        self.a_dict_obj.sample_values(self._a)
        self.b_dict_obj.sample_values(self._b)
        
    def clip_parameters(self):
        self.r0_dict_obj.clip_values(self._r0)
        self.a_dict_obj.clip_values(self._a)
        self.b_dict_obj.clip_values(self._b)
        
    def get_dict(self):
        return {
            "r0_dict": values_to_dict(self.r0),
            "a_dict": values_to_dict(self.a),
            "b_dict": values_to_dict(self.b),
        }
        
    def forward(self, x):
        return tanh_like(x, r0=self.r0, a=self.a, b=self.b)


class Circuit(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        winp_dict: dict,
        wrec_dict: dict,
        offset_dict: dict = {},
        gain_dict: dict = {},
        output_gain_dict: dict = {},
        nonlin: callable = torch.tanh,
        normalize_weights: bool = True,
        device=None,
        dtype=None,
        anderson_kwargs=None,
    ):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.winp_dict = winp_dict
        self.wrec_dict = wrec_dict
        self.gain_dict = gain_dict
        self.output_gain_dict = output_gain_dict
        self.offset_dict = offset_dict
        self.anderson_kwargs = anderson_kwargs or {}
        # pre x post
        self.winp_dict_obj = WeightDict(winp_dict, (n_inputs, n_neurons))
        self.wrec_dict_obj = WeightDict(wrec_dict, (n_neurons, n_neurons))
        self.gain_dict_obj = ValuesDict(gain_dict, n_neurons, default=1.0)
        self.output_gain_dict_obj = ValuesDict(output_gain_dict, n_neurons, default=1.0)
        self.offset_dict_obj = ValuesDict(offset_dict, n_neurons, default=0.0)
        self.nonlin = nonlin
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        self.hook = None

        # parameter objects
        factory_kwargs = dict(dtype=dtype, device=device)
        self._wi = nn.Parameter(
            torch.empty(self.winp_dict_obj.n_params, **factory_kwargs)
        )
        self._wr = nn.Parameter(
            torch.empty(self.wrec_dict_obj.n_params, **factory_kwargs)
        )
        self._gain = nn.Parameter(
            torch.empty(self.gain_dict_obj.n_params, **factory_kwargs)
        )
        self._output_gain = nn.Parameter(
            torch.empty(self.output_gain_dict_obj.n_params, **factory_kwargs)
        )
        self._offset = nn.Parameter(
            torch.empty(self.offset_dict_obj.n_params, **factory_kwargs)
        )

        self.reset_parameters()

        self.num_params = sum(param.numel() for param in self.parameters())

    @property
    def Wi(self):
        return self.winp_dict_obj.get_weights(self._wi)

    @property
    def Wr(self):
        return self.wrec_dict_obj.get_weights(self._wr)

    @property
    def offset(self):
        return self.offset_dict_obj.get_values(self._offset)

    @property
    def gain(self):
        return self.gain_dict_obj.get_values(self._gain)
    
    @property
    def output_gain(self):
        return self.output_gain_dict_obj.get_values(self._output_gain)

    def reset_parameters(self):
        self.winp_dict_obj.sample_values(self._wi)
        self.wrec_dict_obj.sample_values(self._wr)
        self.offset_dict_obj.sample_values(self._offset)
        self.gain_dict_obj.sample_values(self._gain)
        self.output_gain_dict_obj.sample_values(self._output_gain)

    def clip_parameters(self):
        self.winp_dict_obj.clip_values(self._wi)
        self.wrec_dict_obj.clip_values(self._wr)
        self.offset_dict_obj.clip_values(self._offset)
        self.gain_dict_obj.clip_values(self._gain)
        self.output_gain_dict_obj.clip_values(self._output_gain)
        if hasattr(self.nonlin, "clip_parameters"):
            self.nonlin.clip_parameters()
            
    def get_dict(self):
        d = {
            "winp_dict": matrix_to_weightdict(self.Wi),
            "wrec_dict": matrix_to_weightdict(self.Wr),
            "offset_dict": values_to_dict(self.offset),
            "gain_dict": values_to_dict(self.gain),
            "output_gain_dict": values_to_dict(self.output_gain),
        }
        if hasattr(self.nonlin, "get_dict"):
            d["nonlin"] = self.nonlin.get_dict()
        return d

    def forward(
            self, X: torch.Tensor, Y=None, 
            silence_inputs=None,
            silence_recs=None
        ):
        if Y is None:
            Y = torch.zeros((X.shape[0], self.n_neurons)).to(X)

        Wi, Wr = self.Wi, self.Wr
        if silence_inputs is not None:
            Wi.index_fill_(0, torch.tensor(silence_inputs), 0.0)
        if silence_recs is not None:
            Wr.index_fill_(0, torch.tensor(silence_recs), 0.0)
        
        if self.normalize_weights:
            norm = torch.linalg.norm(torch.vstack([Wi, Wr]), ord=1, axis=0)
            Wi = Wi / norm
            Wr = Wr / norm

        offset = self.offset
        gain = self.gain
        nonlin = self.nonlin

        with torch.no_grad():
            Y = anderson(
                lambda Y: step_forward(
                    X,
                    Y,
                    Wi,
                    Wr,
                    offset=offset,
                    gain=gain,
                    nonlin=nonlin,
                ),
                Y,
                **self.anderson_kwargs,
            )["result"]
        z = step_forward(
            X,
            Y,
            Wi,
            Wr,
            offset=offset,
            gain=gain,
            nonlin=nonlin,
        )
        if self.num_params:
            # set up Jacobian vector product (without additional forward calls)
            z0 = z.requires_grad_()
            f0 = step_forward(
                X,
                z0,
                Wi,
                Wr,
                offset=offset,
                gain=gain,
                nonlin=nonlin,
            )

            if self.training:

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                    new_grad = anderson(
                        lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        **self.anderson_kwargs,
                    )["result"]
                    return new_grad

                self.hook = z.register_hook(backward_hook)
        return z * self.output_gain


def anderson(
    f: callable,
    x0: torch.Tensor,
    m=6,
    lam=1e-4,
    threshold=50,
    eps=1e-3,
    stop_mode="rel",
    beta=1.0,
    **kwargs
):
    """Anderson acceleration for fixed point iteration."""
    bsz, L = x0.shape
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out


class LitModel(LightningModule):
    """PyTorch Lightning module for training a model on Numerai data."""
    
    model_class = Circuit

    def __init__(
        self,
        loss=mse_loss,
        learning_rate=1e-3,
        optimizer_type="adamw",
        opt_args=None,
        schedule="linear",
        warmup_steps=0,
        total_steps=1000,
        model_kwargs={},
    ):
        super().__init__()
        self.loss = loss
        self.model_kwargs = model_kwargs
        self.model = self.model_class(**model_kwargs)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.opt_args = opt_args if opt_args is not None else {}
        self.save_hyperparameters(ignore=["model", "loss_function"])

    def forward(self, X, Y, W=None):
        Ypred = self.model(X)
        loss = self.loss(Ypred, Y, mask=torch.isfinite(Y), weight=W)
        return loss, Ypred
    
    def log_r2s(self, Ypred, Y, W, prefix="train"):
        r2s = 0.0
        for i, (y, ypred, w) in enumerate(zip(Y.T, Ypred.T, W.T)):
            isfinite = torch.isfinite(y)
            if isfinite.sum() < 2:
                continue
            w = torch.sqrt(w[isfinite])
            r2 = r2_score(ypred[isfinite]*w, y[isfinite]*w)
            r2s += r2
            self.log(f"{prefix}_r2_{i}", r2)
        self.log(f"{prefix}_r2", r2s / Y.shape[-1])

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch

        loss, Ypred = self.forward(X, Y, W)
        
        self.model.clip_parameters()
        
        self.log("train_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch
            
        loss, Ypred = self.forward(X, Y, W)
                
        self.log("val_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="val")
        return loss
        
    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch
            
        loss, Ypred = self.forward(X, Y, W)
                
        self.log("test_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="test")
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, **self.opt_args)
        elif self.optimizer_type == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, **self.opt_args)
        elif self.optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, **self.opt_args)
        else:
            raise ValueError("Optimizer type not recognized.")

        # TODO add scheduler for training
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     opt, max_epochs*len(train_loader), eta_min=1e-6
        # )
        return optimizer



if __name__ == "__main__":
    import time

    n = 64 * 4
    m = 4
    l = 6
    X = torch.randn((n, m))
    winp_dict = {(0, 1): 0.5, (1, 2): 0.5, (2, 4): 0.5, (3, 5): 0.5}
    wrec_dict = {
        (0, 1): -0.1,
        (1, 0): 0.1,
        (2, 3): -0.1,
        (3, 2): -0.1,
        (4, 5): -0.1,
        (5, 4): 0.1,
    }

    now = time.time()
    circuit = Circuit(l, m, winp_dict=winp_dict, wrec_dict=wrec_dict)

    Y = circuit.forward(X)

    print(circuit.parameters())
    print(time.time() - now)
    print(Y.shape)
    print(np.linalg.norm(Y))
    # print(circuit.forward_res[-1])

    wrec_dict = {
        (0, 1): (-1, 1),
        (1, 0): (-1, 1),
        (2, 3): (-1, 1),
        (3, 2): (-1, 1),
        (4, 5): (-1, 1),
        (5, 4): (-1, 1),
    }
    model_kwargs = dict(
        n_neurons=l, n_inputs=m, winp_dict=winp_dict, wrec_dict=wrec_dict
    )

    model = LitModel(model_kwargs=model_kwargs, learning_rate=5e-2)

    training_data = TensorDataset(X, Y)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    trainer = Trainer(max_epochs=100, log_every_n_steps=1)
    trainer.fit(model, train_dataloader)
