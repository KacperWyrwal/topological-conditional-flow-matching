"""Implements Conditional Flow Matcher Losses."""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math
import warnings
from typing import Union

import torch

from .optimal_transport import OTPlanSampler


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for Schrödinger bridge conditional flow matching method. This class implements
    the SB-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and
    sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        r"""Initialize the SchrodingerBridgeConditionalFlowMatcher class. It requires the hyper-
        parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
            we use exact as the default as we found this to perform better
            (more accurate and faster) in practice for reasonable batch sizes.
            We note that as batchsize --> infinity the correct choice is the
            sinkhorn method theoretically.
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) with respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise: bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch entropic OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """Albergo et al. 2023 trigonometric interpolants class. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t and compute_conditional_flow functions in
    order to compute [3]'s trigonometric interpolants.

    [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def compute_mu_t(self, x0, x1, t):
        r"""Compute the mean of the probability path (Eq.5) from [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: cos(pi t/2)x0 + sin(pi t/2)x1

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        r"""Compute the conditional vector field similar to [3].

        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(\pi*t/2) x0)

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


# Topological conditional flow matching
from .exprel import exprel 
from .fourier_transform import FourierTransform


class ConditionalTopologicalFlowMatcher(ConditionalFlowMatcher):
    """
    This class performs conditional topological flow matching.
    """
    def __init__(
        self, 
        c: float, 
        eigenvalues: torch.Tensor, 
        eigenvectors: torch.Tensor,
        fourier_transform: FourierTransform,
    ) -> None:
        """
        Initialize the ConditionalTopologicalFlowMatcher class.

        Parameters
        ----------
        c : float, heat flow rate in the drift -cLX_t dt. 
        eigenvalues : Tensor, (N, D) eigenvalues of the Laplacian.
        eigenvectors : Tensor, (N, ) eigenvectors of the Laplacian.
        """
        assert c >= 0, f"{c=} should be non-negative."
        assert torch.all(eigenvalues >= 0.0), f"Eigenvalues should be non-negative"
        super().__init__()
        self.eigenvalues = eigenvalues * c # absorb c into eigenvalues, reducing to the case c = 1
        self.eigenvectors = eigenvectors
        self.ft = fourier_transform
        self.zero_threshold = 1e-8 if torch.get_default_dtype() == torch.float64 else 1e-6

        # Precompute some stuff 
        self._exprel_2lambda = exprel(-2.0 * self.eigenvalues)

    def _Phi(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        Phi: Tensor, shape (bs, *dim)
            exp(-c \lambda t)
        """
        return torch.exp(-self.eigenvalues * t.unsqueeze(-1)) # (bs, *dim)

    def _m(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean m_t: Tensor, shape (bs, *dim)
        """
        return self._Phi(t) * y0

    def _sigma_tilde(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        sigma_tilde: Tensor, shape (bs, *dim)
        """
        # Prep for cases
        res = t.new_empty(*t.shape, self.eigenvalues.shape[0])
        zero_eigval_mask = self.eigenvalues < self.zero_threshold
        nonzero_eigvals = self.eigenvalues[~zero_eigval_mask]

        # Case 1: lambda == 0
        res[:, zero_eigval_mask] = t.unsqueeze(-1) # (bs, *dim)

        # Case 2: lambda > 0
        Phi2t = torch.exp(-nonzero_eigvals * 2 * t.unsqueeze(-1))
        res[:, ~zero_eigval_mask] = (
            (1.0 - Phi2t) # (bs, *dim)
            / 
            (2.0 * nonzero_eigvals) # (, *dim)
        ) # (bs, *dim)
        return res

    def _b(self, yt: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor: 
        """
        Returns the prior drift -cLX_t dt.

        Parameters
        ----------
        yt : Tensor, shape (bs, *dim)
            represents the sample from the probability path
        t : FloatTensor, shape (bs)

        Returns
        -------
        b : Tensor, shape (bs, *dim)
            represents the prior drift -cLX_t dt.
        """
        del t
        return -self.eigenvalues * yt 

    def _bridge_m(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        y1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean m_t: Tensor, shape (bs, *dim)
        """
        t1 = torch.ones_like(t)
        mt = self._m(y0, t=t)
        m1 = self._m(y0, t=t1)
        sigma_tilde_t = self._sigma_tilde(t)
        sigma_tilde_1 = self._sigma_tilde(t1)
        return (
            mt 
            + 
            (
                self._Phi(1.0 - t)
                *
                sigma_tilde_t
                / 
                sigma_tilde_1
            )
            * 
            (y1 - m1)
        )

    def _bridge_kappa(self, t: torch.Tensor) -> torch.Tensor:
        return (self._Phi(1.0 - t) / self._exprel_2lambda)
    
    def _bridge_u_over_kappa(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor, yt: torch.Tensor | None = None) -> torch.Tensor:
        del yt 
        t1 = torch.ones_like(t)
        return y1 - self._m(y0, t1)

    def _bridge_u(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor, yt: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the conditional vector field.

        u_t(y) = exp(-cl(1-t)) / exprel(-2cl) * (y1 - m(1, y0))

        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        y1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        yt : Tensor, shape (bs, *dim)
            represents the sample from the probability path

        Returns
        -------
        u : Tensor, shape (bs, *dim)
            represents the conditional vector field
        """
        return self._bridge_kappa(t) * self._bridge_u_over_kappa(y0, y1, t, yt)

    def compute_mu_t__spectral(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of the probability path. Inputs in spectral coordinates.

        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        y1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: Tensor, shape (bs, *dim)

        """
        return self._bridge_m(y0, y1, t)

    def sample_xt__spectral(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor | None = None):
        """
        Draw a sample from the probability path. Inputs in spectral coordinates.

        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        y1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1) (optional)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        """
        del epsilon
        return self.compute_mu_t__spectral(y0, y1, t)

    def compute_conditional_flow__spectral(self, y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor, yt: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the conditional vector field. Inputs in spectral coordinates.

        u_t(y) = y1 - m(1, y0)
        """
        return self._bridge_u_over_kappa(y0, y1, t, yt)

    def sample_location_and_conditional_flow__spectral(self, y0, y1, t=None, return_noise=False):
        """
        Compute the sample xt and the conditional vector field.

        Parameters
        ----------
        y0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        y1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        yt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field
        (optionally) eps: Tensor, shape (bs, *dim) such that yt = mu_t + sigma_t * epsilon
        """
        if t is None:
            t = torch.rand(y0.shape[0], dtype=y0.dtype, device=y0.device)
        assert len(t) == y0.shape[0], "t has to have batch size dimension"

        yt = self.sample_xt__spectral(y0, y1, t, epsilon=None)
        ut = self.compute_conditional_flow__spectral(y0, y1, t, yt)
        if return_noise:
            raise NotImplementedError("Not implemented")
        else:
            return t, yt, ut

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: Tensor, shape (bs, *dim)

        """
        t = pad_t_like_x(t, x0)
        raise NotImplementedError("Not implemented")

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        """
        del epsilon
        raise NotImplementedError("Not implemented")

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        """
        raise NotImplementedError("Not implemented")

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt and the conditional vector field.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon
        """
        y0, y1 = self.ft.transform(x0), self.ft.transform(x1)

        if return_noise:
            t, yt, ut, eps = self.sample_location_and_conditional_flow__spectral(y0, y1, t, return_noise)
        else:
            t, yt, ut = self.sample_location_and_conditional_flow__spectral(y0, y1, t, return_noise)
        
        xt, vt = self.ft.inverse_transform(yt), self.ft.inverse_transform(ut)

        if return_noise:
            return t, xt, vt, eps
        else:
            return t, xt, vt


class ExactOptimalTransportConditionalTopologicalFlowMatcher(ConditionalTopologicalFlowMatcher):
    """Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, c: float, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, fourier_transform: FourierTransform):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(c, eigenvalues, eigenvectors, fourier_transform)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class TopologicalVectorField(torch.nn.Module):
    def __init__(self, u_over_kappa_fn: torch.nn.Module, fm: ConditionalTopologicalFlowMatcher):
        super().__init__()
        self.u_over_kappa_fn = u_over_kappa_fn
        self.fm = fm 
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor=None, *args, **kwargs) -> torch.Tensor:        
        # standard coordinates computations
        u_over_kappa = self.u_over_kappa_fn(t=t, x=x, y=y, *args, **kwargs)
        u_over_kappa_spectral = self.fm.ft.transform(u_over_kappa)

        # spectral coordinates computations
        x_spectral = self.fm.ft.transform(x)
        kappa = self.fm._bridge_kappa(t=t)
        u_spectral = kappa * u_over_kappa_spectral
        b_spectral = self.fm._b(x_spectral, t)
        dx_spectral = b_spectral + u_spectral 
        dx = self.fm.ft.inverse_transform(dx_spectral)

        return dx