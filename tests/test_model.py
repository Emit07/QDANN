"""Checks on the QDANN components, run against SPEC.md rather than against the paper.

Two of these guard discrepancies in the publication: test_pinball_asymmetry guards the
swapped branches of Eq. 6 (AMBIGUITIES.md #1) and test_gradient_reversal_flips_only_the
_extractor guards the sign convention of Eq. 3 (AMBIGUITIES.md #10). Both failures train
and converge normally, so nothing else would catch them.
"""

import pytest
import torch
from torch import nn

from qdann.losses import QUANTILES, pinball, quantile_loss, update_quantile_weights
from qdann.model import QDANN


def _linear_count(module: nn.Module) -> int:
    return sum(isinstance(layer, nn.Linear) for layer in module)


def _overfit(seed: int, n: int = 50, steps: int = 400) -> float:
    """Fit a handful of samples with dropout and the adversarial branch off.

    Returns the final loss as a fraction of the loss of predicting the mean, so the
    threshold does not depend on the scale of the synthetic yield.
    """
    torch.manual_seed(seed)
    x = torch.randn(n, 27)
    y = x[:, 0] * 2 - x[:, 1]
    model = QDANN(dropout=0.0, lambda_=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    weights = dict.fromkeys(QUANTILES, 1.0)
    for _ in range(steps):
        loss = quantile_loss(y=y, yhat=model(x)[0], weights=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    null_loss = quantile_loss(y=y, yhat=y.mean().expand_as(y), weights=weights)
    return float(loss.detach() / null_loss)


def test_shapes_and_depth() -> None:
    model = QDANN()
    # Section 3.1's "total depth of ten" is G_f -> G_y (AMBIGUITIES.md #9)
    assert _linear_count(model.features) == 6
    assert _linear_count(model.yield_head) == 4
    assert _linear_count(model.discriminator) == 4

    x = torch.randn(8, 27)
    assert model.features(x).shape == (8, 32)
    yhat, logit = model(x)
    assert yhat.shape == (8,)
    assert logit.shape == (8,)


def test_gradient_reversal_flips_only_the_extractor() -> None:
    """The GRL must negate what reaches G_f and leave G_d's own gradients alone."""
    model = QDANN(dropout=0.0).eval()
    x = torch.randn(16, 27)

    def domain_gradients(lambda_: float) -> dict[str, torch.Tensor]:
        model.lambda_ = lambda_
        model.zero_grad()
        model(x)[1].sum().backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    reversed_, forward = domain_gradients(1.0), domain_gradients(-1.0)
    for name, gradient in reversed_.items():
        if name.startswith("features"):
            assert torch.allclose(gradient, -forward[name], atol=1e-6), name
        elif name.startswith("discriminator"):
            assert torch.allclose(gradient, forward[name], atol=1e-6), name
    assert any(name.startswith("features") for name in reversed_)


@pytest.mark.parametrize(
    ("q", "over_cost", "under_cost"), [(0.1, 9.0, 1.0), (0.5, 5.0, 5.0), (0.9, 1.0, 9.0)]
)
def test_pinball_asymmetry(q: float, over_cost: float, under_cost: float) -> None:
    """Fig. 6: against y = 10, a prediction of 20 costs what the caption says it costs.

    Eq. 6 as printed gives the transpose of this. AMBIGUITIES.md #1.
    """
    y = torch.tensor([10.0])
    assert pinball(y=y, yhat=torch.tensor([20.0]), q=q) == pytest.approx(over_cost)
    assert pinball(y=y, yhat=torch.tensor([0.0]), q=q) == pytest.approx(under_cost)


def test_quantile_weights_track_the_bias() -> None:
    y = torch.full((100,), 10.0)
    over = update_quantile_weights(y=y, yhat=y + 1)
    under = update_quantile_weights(y=y, yhat=y - 1)
    unbiased = update_quantile_weights(y=y, yhat=y + torch.tensor([1.0, -1.0] * 50))

    assert over[0.1] > over[0.9]
    assert under[0.9] > under[0.1]
    assert unbiased[0.1] == pytest.approx(unbiased[0.9])
    for weights in (over, under, unbiased):
        # Eqs. 10-11 share a denominator, so the outer weights always sum to 2
        assert weights[0.1] + weights[0.9] == pytest.approx(2.0)
        assert weights[0.5] == 1.0


def test_quantile_weights_ignore_exact_ties() -> None:
    """The indicator of Eqs. 8-9 is strictly positive, so r_o + r_u <= 1."""
    y = torch.full((100,), 10.0)
    assert update_quantile_weights(y=y, yhat=y) == pytest.approx(
        dict.fromkeys(QUANTILES, 1.0)
    )


def test_overfits_a_small_sample() -> None:
    """Nothing downstream is worth running if 50 samples cannot be memorized."""
    assert _overfit(seed=0) < 0.05


def test_seeds_vary_without_diverging() -> None:
    losses = [_overfit(seed=seed) for seed in range(3)]
    assert len(set(losses)) == 3
    assert max(losses) < 0.05
