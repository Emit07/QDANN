"""The VAE data filter: which source county-years look like the target domain.

Ma et al., Remote Sensing of Environment 315 (2024) 114427, Section 3.2 and Eqs. 12-16. See
SPEC.md section 6.

Trained on the *target* feature vectors, then run over the *source* ones: a county-year the
VAE cannot reconstruct is a county-year unlike anything in the target domain, and Wang et al.
(2019) is the reason to drop it rather than let it drag the alignment. Survivors are weighted
by 1/L_i (Eq. 16), which `train.fit` passes through to `losses.quantile_loss`.

The VAE is trained on its own, before QDANN, and never sees the yield loss.
AMBIGUITIES.md #21.
"""

import torch
from torch import nn

from qdann.model import _stack


class VAE(nn.Module):
    """Encoder to a Gaussian latent, decoder back to the 27 features. AMBIGUITIES.md #5."""

    def __init__(
        self,
        n_features: int = 27,
        hidden: tuple[int, ...] = (64, 32),
        latent_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = _stack(
            [n_features, *hidden], dropout=dropout, activate_last=True
        )
        self.mu = nn.Linear(hidden[-1], latent_dim)
        self.logvar = nn.Linear(hidden[-1], latent_dim)
        self.decoder = _stack(
            [latent_dim, *reversed(hidden), n_features],
            dropout=dropout,
            activate_last=False,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        # reparameterization: sample in training, take the mean once in eval, so the
        # per-sample error of Eq. 15 is a property of the row and not of the draw
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decoder(z), mu, logvar


def vae_loss(
    x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Eqs. 12-14: squared reconstruction error plus the closed-form KL to N(0, 1)."""
    recon = ((xhat - x) ** 2).sum(dim=1).mean()
    latent = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()
    return recon + latent


def fit(
    x: torch.Tensor,
    epochs: int = 400,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> VAE:
    """Train on target-domain vectors only."""
    torch.manual_seed(seed)
    model = VAE(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        batches = torch.randperm(len(x)).split(batch_size)
        # BatchNorm1d raises on a batch of one, as in train.fit
        if len(batches[-1]) == 1:
            batches = batches[:-1]
        for batch in batches:
            loss = vae_loss(x[batch], *model(x[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def reconstruction_error(model: VAE, x: torch.Tensor) -> torch.Tensor:
    """Eq. 15: one L_i per row."""
    with torch.no_grad():
        xhat = model.eval()(x)[0]
    return ((xhat - x) ** 2).sum(dim=1)


def filter_threshold(errors: torch.Tensor, crop: str) -> float:
    """Section 3.2's threshold table: the mean, or the 80th percentile for winter wheat."""
    if crop == "winter_wheat":
        return float(torch.quantile(errors, 0.8))
    return float(errors.mean())
