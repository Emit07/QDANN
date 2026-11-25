"""QDANN architecture: feature extractor, yield predictor, domain discriminator.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.1. See SPEC.md
section 5 for the architecture statement this is built from.
"""

import itertools

import torch
from torch import nn

# the paper's 0.5 underfits ~860 county rows; picked on an inner split, AMBIGUITIES.md #24
DROPOUT = 0.1


class GradientReversal(torch.autograd.Function):
    """Identity on the forward pass, negated and scaled gradient on the backward pass.

    Ganin et al. (2016). This is how Eq. 3's minus sign is realized: one backward pass
    trains G_d to minimize the domain loss while G_f maximizes it. Applying the minus to
    the loss directly would make G_d maximize its own loss as well. AMBIGUITIES.md #10.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class _Dropout(nn.Dropout):
    """nn.Dropout from a uniform draw: CPU bernoulli_ was 40% of a training step, three
    times what rand_like costs for the same mask."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        return x * (torch.rand_like(x) >= self.p) / (1 - self.p)


def _stack(sizes: list[int], dropout: float, activate_last: bool) -> nn.Sequential:
    """Linear layers over `sizes`, with BatchNorm + ReLU + Dropout between hidden layers.

    Section 3.1 puts normalization and dropout "between hidden layers", so neither follows
    the final Linear of a block.
    """
    layers: list[nn.Module] = []
    for i, (in_features, out_features) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(in_features, out_features))
        if i < len(sizes) - 2:
            layers += [nn.BatchNorm1d(out_features), nn.ReLU(), _Dropout(dropout)]
        elif activate_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class QDANN(nn.Module):
    """Feature extractor G_f feeding a yield predictor G_y and a domain discriminator G_d.

    Layer counts follow Section 3.1's "total depth of ten": G_f is one input layer, four
    hidden layers and one output layer (six Linear); G_y and G_d are one input layer, two
    hidden layers and one output layer (four Linear each). G_f -> G_y is ten.
    AMBIGUITIES.md #9.
    """

    def __init__(
        self,
        n_features: int = 27,
        hidden: int = 64,
        feature_out: int = 32,
        n_feature_hidden: int = 4,
        n_head_hidden: int = 2,
        dropout: float = DROPOUT,
        lambda_: float = 1.0,
        intervals: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_ = lambda_
        # ReLU on the 32-d bottleneck: without it G_f's output layer and each head's input
        # layer compose into a single linear map. AMBIGUITIES.md #12
        self.features = _stack(
            [n_features, *[hidden] * (1 + n_feature_hidden), feature_out],
            dropout=dropout,
            activate_last=True,
        )
        head_sizes = [feature_out, *[hidden] * (1 + n_head_hidden)]
        # `intervals` adds q10 and q90 as outputs 1 and 2 of the last Linear; output 0
        # stays Eq. 7's point estimate. Off, the head and its init are the paper's.
        # AMBIGUITIES.md #23
        self.yield_head = _stack(
            [*head_sizes, 3 if intervals else 1], dropout=dropout, activate_last=False
        )
        # a logit, paired with BCEWithLogitsLoss; Eq. 5 is plain cross-entropy either way
        self.discriminator = _stack(
            [*head_sizes, 1], dropout=dropout, activate_last=False
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Point yield, domain logit, and the (n, 2) q10/q90 outputs ((n, 0) without
        `intervals`)."""
        z = self.features(x)
        reversed_z = GradientReversal.apply(z, self.lambda_)
        out = self.yield_head(z)
        return out[:, 0], self.discriminator(reversed_z).squeeze(-1), out[:, 1:]
