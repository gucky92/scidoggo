"""
"""

import torch.nn as nn
import torch
from torch.distributions import constraints

import pyro
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist


class ANN(PyroModule):
    """
    NN with arbitray many hidden layers
    """

    def __init__(
        self, input_dim=1, hidden_dim=100, output_dim=1,
        hidden_layers=1, nonlin=torch.relu,
        output_nonlin=None, bias=True,
    ):
        super().__init__()

        self.input_layer = PyroModule[nn.Linear](input_dim, hidden_dim, bias=bias)
        self.hidden_layers = hidden_layers

        for i in range(self.hidden_layers):
            setattr(
                self,
                self.get_hidden_layer_name(i),
                PyroModule[nn.Linear](hidden_dim, hidden_dim, bias=bias)
            )

        self.output_layer = PyroModule[nn.Linear](hidden_dim, output_dim, bias=bias)

        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def get_hidden_layer_name(self, i):
        return f'hidden_layer_{i}'

    def get_hidden_layer(self, i):
        return getattr(self, self.get_hidden_layer_name(i))

    def forward(self, x):
        x = self.nonlin(self.input_layer(x))

        for i in range(self.hidden_layers):
            x = self.nonlin(
                self.get_hidden_layer(i)(x)
            )

        y = self.output_layer(x)
        if self.output_nonlin is None:
            return y
        else:
            return self.output_nonlin(y)


class MonotonicNonlinearity(ANN):
    """
    Monotonically increasing nonlinearity
    """

    def __init__(
        self,
        hidden_dim=10,
        nonlin=torch.tanh,
        hidden_layers=1,
        output_dim=1,
        input_dim=1,
        output_nonlin=None,
        pos=True,
        input_scale=1,
        output_scale=1,
        bayesian=False,
        **kwargs
    ):
        # TODO make bayesian
        super().__init__(
            input_dim, hidden_dim, output_dim,
            hidden_layers=hidden_layers, nonlin=nonlin,
            output_nonlin=output_nonlin, **kwargs
        )

        self.pos = pos
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.bayesian = bayesian

        if self.pos:
            constraint = constraints.positive
            sign = 1
        else:
            constraint = constraints.less_than(0.0)
            sign = -1

        if self.bayesian:
            raise NotImplementedError('bayesian')

        else:
            self.input_layer.weight = PyroParam(
                sign * torch.abs(self.input_layer.weight.detach().clone()),
                constraint=constraint,
                event_dim=2
            )

            self.output_layer.weight = PyroParam(
                sign * torch.abs(self.output_layer.weight.detach().clone()),
                constraint=constraint,
                event_dim=2
            )

            for i in range(self.hidden_layers):
                layer = self.get_hidden_layer(i)
                layer.weight = PyroParam(
                    sign * torch.abs(layer.weight.detach().clone()),
                    constraint=constraint,
                    event_dim=2
                )

    def forward(self, x):
        return super().forward(x * self.input_scale) * self.output_scale
