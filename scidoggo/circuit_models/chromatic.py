"""
Chromatic Encoding models
"""

import torch

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist


from .interp1d import Interp1d
from .utils import identity, FLOAT_TYPE


interp1d = Interp1d()


def calc_capture(wls, s_filter, i_spectrum, b_spectrum=None):
    """
    Parameters
    ----------
    wls : torch.tensor (wavelengths)
    s_filter : torch.tensor (opsin x wavelengths)
    i_spectrum : torch.tensor (... wavelengths)
    b_spectrum : torch.tensor (... wavelengths)

    Returns
    -------
    captures : torch.tensor (... opsin)
    """
    i_spectrum = i_spectrum[..., None, :]
    i_captures = torch.trapz(s_filter * i_spectrum, wls)
    if b_spectrum is None:
        return i_captures

    b_captures = calc_capture(wls, s_filter, b_spectrum)
    return i_captures / b_captures


def calc_log_excitation(wls, s_filter, i_spectrum, b_spectrum=None):
    """
    Parameters
    ----------
    wls : torch.tensor (wavelengths)
    s_filter : torch.tensor (opsin x wavlengths)
    i_spectrum : torch.tensor (... wavelengths)
    b_spectrum : torch.tensor (... wavelengths)

    Returns
    -------
    excitation : torch.tensor (... opsin)
    """
    captures = calc_capture(wls, s_filter, i_spectrum, b_spectrum)
    return torch.log(captures)


class PhotoreceptorAdaptation(PyroModule):
    """
    Photoreceptor Adaptation model

    .. math::

    f(S, I, B) = f\left( g(f_a(q_s, q_b), f_a(q_b, q_b)) \right)
    """

    def __init__(
        self,
        adapt_func=identity,  # function that return W_s
        nonlin=torch.log,  # can be a parameterized function as well
        method=torch.div
    ):
        super().__init__()
        self.adapt_func = adapt_func
        self.nonlin = nonlin
        self.method = method

    def forward(self, wls, s_filter, i_spectrum, b_spectrum):
        qs = calc_capture(wls, s_filter, i_spectrum)  # stims x opsins
        qb = calc_capture(wls, s_filter, b_spectrum)  # stims x opsins

        num = self.adapt_func(qs, qb)
        denom = self.adapt_func(qb, qb)
        return self.nonlin(self.method(num, denom))


class NoiseThresholdedPrAdaptation(PyroModule):
    """
    Noise thresholded photoreceptor adaptation:

    .. math::

    f(S, I, B) = g \left(\frac{q_s+noise}{q_b+noise} \right)
    """

    def __init__(
        self, 
        noise=PyroSample(dist.HalfNormal(1)),  # fairly uninformative prior
        nonlin=torch.log, 
        method=torch.div
    ):
        super().__init__()
        self.noise = noise
        self.nonlin = nonlin
        self.method = method

    def forward(self, wls, s_filter, i_spectrum, b_spectrum):
        qs = calc_capture(wls, s_filter, i_spectrum)  # stims x opsins
        qb = calc_capture(wls, s_filter, b_spectrum)  # stims x opsins

        num = qs + self.noise
        denom = qb + self.noise
        return self.nonlin(self.method(num, denom))


class ChromaticEncodingModel(PyroModule):
    """
    Abstracted Encoding Model for spectral response data.

    Parameters
    ----------
    wls : torch.tensor (wavelengths)
    s_func : callable
        Spectral sensitivity function returns the spectral sensitivity of
        all photoreceptors.
        Returns shape: (... opsin x wavelengths).
    e_func : callable
        Encoding model that takes two arguments:
        The photorecpetor excitations and the neural responses.
        Returns the predicted neural response or something similar.
    p_func : callable
        Photoreceptor Model that takes four arguments: wavelengths,
        spectral sensitivity, illuminant spectrum, and background spectrum.
        This should return the photoreceptor excitations.
        Returns shape: (... [stimuli x] opsin).
    m_func : callable
        Measurement conversion function that accepts a tensor in the
        hardware stimulation device units and returns the corresponding
        spectrum in photonflux.
        Returns shape: (... [stimuli x] wavelengths)
    kwargs : dict
        Other attributes for the Encoding Model
    """

    def __init__(
        self, wls, s_func, e_func,
        p_func=calc_capture, m_func=identity,
        adapt_s=False,
        **kwargs
    ):
        super().__init__()
        self.wls = wls
        self.s_func = s_func
        self.e_func = e_func
        self.p_func = p_func
        self.m_func = m_func
        self.adapt_s = adapt_s

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, X, y=None, Xb=None, **kwargs):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.tensor (... [stimuli x] outputs)
            Tensor of the stimulation intensities.
            The outputs could already be in shape (... stimuli x wavelengths),
            if the spectra were created prior to training (i.e. fixed).
        y : torch.tensor (... [stimuli x] neurons/rois), optional
            Tensor of the responses of neurons.
        Xb : torch.tensor (... outputs), optional
            Tensor of the background stimulation intensities.
            The outputs could already be in shape (... wavelengths),
            if the spectra were created prior to training (i.e. fixed).
        """
        # get illuminant spectrum
        i_spectrum = self.m_func(X)
        # get background spectrum
        if Xb is None:
            b_spectrum = None
        else:
            b_spectrum = self.m_func(Xb)
        # get sensitivity
        if self.adapt_s:
            assert b_spectrum is not None
            s_filter = self.s_func(b_spectrum)
        else:
            s_filter = self.s_func()
        # apply photoreceptor model
        if b_spectrum is None:
            Xe = self.p_func(self.wls, s_filter, i_spectrum)
        else:
            Xe = self.p_func(self.wls, s_filter, i_spectrum, b_spectrum)
        # apply encoding model
        if y is None:
            return self.e_func(Xe, **kwargs)
        else:
            return self.e_func(Xe, y, **kwargs)


# class RecurrentBaselineModel(PyroModule):
#     """
#     Recurrent baseline model is the steady state of these two functions:

#     dr/dt = - gr (r - r0) - ghis (r - rhis) - gdm9 (r - rdm9) + Xe (relative excitation)
#     dx/dt = - gx (x - x0) - gort (x - xort)

#     r0 ~~ background excitation
#     """


class Stavenga1993SensitivityModel(PyroModule):
    """
    Spectral sensitivity based on the Stavenga 1993 opsin template.
    """

    def __init__(
        self,
        wls,
        alpha_max,
        a_alpha=torch.tensor(380., dtype=FLOAT_TYPE),
        b_alpha=torch.tensor(6.09, dtype=FLOAT_TYPE),
        beta_max=torch.tensor(350., dtype=FLOAT_TYPE),
        A_beta=torch.tensor(0.29, dtype=FLOAT_TYPE),
        a_beta=torch.tensor(247., dtype=FLOAT_TYPE),
        b_beta=torch.tensor(3.59, dtype=FLOAT_TYPE), 
        bg_transformer=None, 
        bg_transform_vars=[],  # can't be the bs
    ):
        super().__init__()
        self.wls = wls
        self.alpha_max = alpha_max
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.A_beta = A_beta
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.beta_max = beta_max
        self.bg_transformer = bg_transformer
        self.bg_transform_vars = bg_transform_vars

    def forward(self, b_spectrum=None):
        if b_spectrum is not None and self.bg_transformer is not None:
            output = self.bg_transformer(b_spectrum)
        
            idx = 0
            if 'alpha_max' in self.bg_transform_vars:
                alpha_max = output[idx][..., None]
                idx += 1
            else:
                alpha_max = self.alpha_max[..., None]

            if 'beta_max' in self.bg_transform_vars:
                beta_max = output[idx][..., None]
                idx += 1
            else:
                beta_max = self.beta_max[..., None]

            if 'a_alpha' in self.bg_transform_vars:
                a_alpha = output[idx][..., None]
                idx += 1
            else:
                a_alpha = self.a_alpha[..., None]

            if 'a_beta' in self.bg_transform_vars:
                a_beta = output[idx][..., None]
                idx += 1
            else:
                a_beta = self.a_beta[..., None]

            if 'A_beta' in self.bg_transform_vars:
                A_beta = output[idx][..., None]
                idx += 1
            else:
                A_beta = self.A_beta[..., None]
        else:
            alpha_max = self.alpha_max[..., None]
            beta_max = self.beta_max[..., None]
            a_alpha = self.a_alpha[..., None]
            a_beta = self.a_beta[..., None]
            A_beta = self.A_beta[..., None]


        x_alpha = torch.log10(self.wls / alpha_max)
        alpha_band = self._band_calculation(
            x_alpha, a_alpha, self.b_alpha[..., None]
        )
        x_beta = torch.log10(self.wls / beta_max)
        beta_band = self._band_calculation(
            x_beta, a_beta, self.b_beta[..., None]
        )
        sense = (alpha_band + A_beta * beta_band)
        pyro.deterministic(
            self._pyro_get_fullname("absorption"),
            sense
        )
        return sense

    @staticmethod
    def _band_calculation(x, a, b):
        return torch.exp(
            -a * x ** 2 * (
                1 + b * x + 3 / 8 * (b * x) ** 2
            )
        )


class FlyStavenga1993SensitivityModel(Stavenga1993SensitivityModel):
    """
    Fly model with defined priors.
    """
    _alpha_max = dist.Uniform(
        torch.tensor([450, 300, 340, 450, 525], dtype=FLOAT_TYPE),
        torch.tensor([550, 340, 380, 500, 675], dtype=FLOAT_TYPE)
    )
    # _beta_max = dist.Uniform(
    #     torch.tensor([320, 345, 345, 345, 345], dtype=FLOAT_TYPE),
    #     torch.tensor([380, 355, 355, 355, 355], dtype=FLOAT_TYPE)
    # )
    _a_alpha = dist.Uniform(
        torch.tensor([140]*5, dtype=FLOAT_TYPE),
        torch.tensor([700]*5, dtype=FLOAT_TYPE)
    )
    _a_beta = dist.Uniform(
        torch.tensor([70]*5, dtype=FLOAT_TYPE),
        torch.tensor([700]*5, dtype=FLOAT_TYPE)
    )
    _A_beta = dist.Uniform(
        torch.tensor([1]+[0]*2+[0]*2, dtype=FLOAT_TYPE),
        torch.tensor([4]+[0.01]*2+[0.6]*2, dtype=FLOAT_TYPE)
    )

    def __init__(
        self,
        wls=torch.arange(300, 700, dtype=FLOAT_TYPE),
        alpha_max=None,
        # beta_max=None,
        a_alpha=None,
        a_beta=None,
        A_beta=None, 
        **kwargs
    ):
        if alpha_max is None:
            alpha_max = PyroSample(self._alpha_max)
        # if beta_max is None:
        #     beta_max = PyroSample(self._beta_max)
        if a_alpha is None:
            a_alpha = PyroSample(self._a_alpha)
        if a_beta is None:
            a_beta = PyroSample(self._a_beta)
        if A_beta is None:
            A_beta = PyroSample(self._A_beta)

        super().__init__(
            wls,
            alpha_max=alpha_max,
            # beta_max=beta_max,
            a_alpha=a_alpha,
            a_beta=a_beta,
            A_beta=A_beta, 
            **kwargs
        )


class FlyStavenga1993InnerSensitivityModel(FlyStavenga1993SensitivityModel):
    """
    Fly model with defined priors.
    """
    _alpha_max = dist.Uniform(
        torch.tensor([325, 350, 425, 525], dtype=FLOAT_TYPE),
        torch.tensor([340, 365, 450, 675], dtype=FLOAT_TYPE)
    )
    # _beta_max = torch.tensor([350, 350, 350, 350], dtype=FLOAT_TYPE) 
    # dist.Uniform(
    #     torch.tensor([345, 345, 345, 345], dtype=FLOAT_TYPE),
    #     torch.tensor([355, 355, 355, 355], dtype=FLOAT_TYPE)
    # )
    # same alpha band width for all photoreceptors
    _a_alpha = dist.Uniform(
        torch.tensor(100, dtype=FLOAT_TYPE),
        torch.tensor(500, dtype=FLOAT_TYPE)
    )
    # same beta band width for all photoreceptors
    _a_beta = dist.Uniform(
        torch.tensor(200, dtype=FLOAT_TYPE),
        torch.tensor(300, dtype=FLOAT_TYPE)
    )
    # same beta band contribution across photoreceptors
    _A_beta = dist.Uniform(
        # torch.tensor([0]*2+[0]*2, dtype=FLOAT_TYPE),
        # torch.tensor([0.01]*2+[0.4, 0.7], dtype=FLOAT_TYPE)
        torch.tensor(0, dtype=FLOAT_TYPE),
        torch.tensor(0.8, dtype=FLOAT_TYPE)
    )



class SpectralSensitivityModel(PyroModule):
    """
    Abstract model for spectral sensitivity functions.

    Parameters
    ----------
    wls : torch.tensor (wavelenghts)
    mean : torch.tensor (opsin x wavelengths)
    sd : torch.tensor, optional (opsin x wavelengths)
    n : int, optional
    name : str, optional
    params : dict, optional
    """

    def __init__(self, wls, mean, sd=1, n=1, name='sensitivity', **params):
        super().__init__()
        self.wls = wls
        self.mean = mean
        self.sd = sd
        self.n = n
        self.name = name
        for key, value in params.items():
            setattr(self, key, value)

        self.prior = self.get_prior()

    def get_prior(self):
        raise NotImplementedError("`get_prior` method.")

    def forward(self, *args):
        raise NotImplementedError("`forward` method.")


class FixedSensitivity(SpectralSensitivityModel):

    def get_prior(self):
        pass

    def forward(self, *args):
        return self.mean


class StrongNormalSensitivityPrior(SpectralSensitivityModel):

    def get_prior(self):
        return PyroSample(
            dist.Normal(
                self.mean, self.sd / torch.sqrt(self.n)
            ).to_event(self.mean.ndim)
        )

    def forward(self, *args):
        # clamp sensitivity at zero
        return torch.relu(self.prior)


class WeakNormalSensitivityPrior(SpectralSensitivityModel):

    def get_prior(self):
        return PyroSample(
            dist.Normal(self.mean, self.sd).to_event(self.mean.ndim)
        )

    def forward(self, *args):
        # clamp sensitivity at zero
        return torch.relu(self.prior)


class MeasurementConversion(PyroModule):
    """
    Parameters
    ----------
    wls : torch.tensor (wavelengths)
    measurements : two-tuples of tensors (outputs, wavelengths x outputs)
    params : dict
    """
    output_postfix = '_output'

    def __init__(self, wls, *measurements, names='measurement', **params):
        super().__init__()
        self.wls = wls
        self.n_wls, = wls.shape
        names_ = []
        for idx, (output, measurement) in enumerate(measurements):
            if isinstance(names, str):
                name = f"{names}{idx}"
            else:
                name = names[idx]
            names.append(name)
            setattr(self, name, measurement)
            setattr(self, name + self.output_postfix, output)

        self.names = names_

        for key, value in params.items():
            setattr(self, key, value)

    def forward(self, outputs):
        """
        Parameters
        ----------
        output : torch.tensor (stimuli x outputs)

        Returns
        -------
        spectrum : torch.tensor (stimuli x wavelengths)
        """
        raise NotImplementedError("`forward` method")


class InterpolateSumMeasurement(MeasurementConversion):

    def forward(self, outputs):
        assert outputs.ndim == 2
        n_stimuli, _ = outputs.shape
        # init spectrum
        spectrum = torch.zeros((n_stimuli, self.n_wls))
        # interpolate for each measurement
        for idx, name in enumerate(self.names):
            # set and reshape output
            output = outputs[..., idx].expand(self.n_wls, n_stimuli)
            # get output x array and measurement y array
            moutput = getattr(self, name + self.output_postfix)
            measurement = getattr(self, name)
            # (wavlengths x samples).T
            # clamp spectrum to zero
            # get new y array
            spectrum += torch.relu(interp1d(moutput, measurement, output)).T
        return spectrum
