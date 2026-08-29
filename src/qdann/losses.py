"""Yield and domain losses for QDANN.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Equations 5-11 and 16. See
SPEC.md section 4.
"""

import math

import torch
from torch.nn import functional


QUANTILES = (0.1, 0.5, 0.9)


def pinball(y: torch.Tensor, yhat: torch.Tensor, q: float) -> torch.Tensor:
    """Per-sample quantile loss (Koenker and Bassett 1978), Eq. 6.

    Overestimation costs (1-q)|err| and underestimation q|err|, which is the reverse of
    Eq. 6 as printed. Section 3.1, the caption of Fig. 6 and the standard form of the loss
    all agree on this direction and the equation alone does not. AMBIGUITIES.md #1.
    """
    err = y - yhat
    return torch.maximum(q * err, (q - 1) * err)


def quantile_loss(
    y: torch.Tensor,
    yhat: torch.Tensor,
    weights: dict[float, float],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted combination of the three quantile losses, Eq. 7.

    `sample_weights` carries the VAE filter's 1/L_i (Eq. 16), which is printed as an
    unnormalized sum; reducing by a weighted mean keeps the yield term commensurate with
    the domain term of Eq. 5, which lambda = 1 assumes. AMBIGUITIES.md #11.
    """
    per_sample = sum(w * pinball(y=y, yhat=yhat, q=q) for q, w in weights.items())
    return _reduce(per_sample, sample_weights)


def mse_loss(
    y: torch.Tensor, yhat: torch.Tensor, sample_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """The yield loss of the DNN baseline (Section 4.2) and of Fig. 16's ablation arm.

    Reduced exactly as `quantile_loss` is, so that dropping the quantile loss does not also
    silently drop the VAE filter's 1/L_i.
    """
    return _reduce((y - yhat) ** 2, sample_weights)


def _reduce(
    per_sample: torch.Tensor, sample_weights: torch.Tensor | None
) -> torch.Tensor:
    if sample_weights is None:
        return per_sample.mean()
    return (sample_weights * per_sample).sum() / sample_weights.sum()


def update_quantile_weights(
    y: torch.Tensor, yhat: torch.Tensor
) -> dict[float, float]:
    """Rebalance the quantile weights from the source domain's bias, Eqs. 8-11.

    Called once, at epoch M, on the whole source set. The indicator of Eqs. 8-9 is strictly
    positive, so exact ties count toward neither ratio and r_o + r_u <= 1.
    """
    r_o = float((yhat > y).to(torch.float64).mean())
    r_u = float((y > yhat).to(torch.float64).mean())
    over, under = math.exp(r_o**2), math.exp(r_u**2)
    # w_0.5 stays 1: it penalizes over- and underestimation equally
    return {
        0.1: 2 * over / (over + under),
        0.5: 1.0,
        0.9: 2 * under / (over + under),
    }


def domain_loss(logit: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over both domains, Eq. 5, which is printed without its minus.

    Takes a logit rather than a probability for numerical stability; d is 1 for source
    samples and 0 for target ones (Section 3.1).
    """
    return functional.binary_cross_entropy_with_logits(logit, d)
