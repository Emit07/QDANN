"""Synthetic domain shift with known target labels, to verify QDANN end to end.

Section 3.1 argues the scale transfer is a marginal shift with a stable posterior:
p_s(x) != p_t(x) while p_s(y|x) ~= p_t(y|x). This generator reproduces exactly that and
nothing else -- it is a diagnostic, not a simulation of maize.

Because the generator supplies target labels, this measures what the real pipeline cannot:
whether the adversarial branch recovers accuracy that a source-only model loses.

  uv run python -m qdann.synth
"""

import torch

from qdann.losses import QUANTILES, domain_loss, quantile_loss, update_quantile_weights
from qdann.model import QDANN


N_FEATURES = 27
N_SIGNAL = 6


def make_domains(
    n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a source and a target domain sharing one y|x but differing in p(x).

    Yield depends only on the first N_SIGNAL features, identically in both domains. The
    remaining features are nuisance: in the source they track the yield, so a source-only
    model is rewarded for reading them, and in the target they are shifted and carry
    nothing. Discarding them is what domain-invariance buys.
    """
    generator = torch.Generator().manual_seed(seed)

    def sample(count: int, spurious: bool) -> tuple[torch.Tensor, torch.Tensor]:
        signal = torch.randn(count, N_SIGNAL, generator=generator)
        y = signal @ weights + 0.3 * signal[:, 0] ** 2
        nuisance = torch.randn(count, N_FEATURES - N_SIGNAL, generator=generator)
        nuisance += y.unsqueeze(1) if spurious else 2.0
        return torch.cat([signal, nuisance], dim=1), y

    weights = torch.randn(N_SIGNAL, generator=generator)
    source_x, source_y = sample(count=n, spurious=True)
    target_x, target_y = sample(count=n, spurious=False)
    # standardized on source alone, so the shift survives into the model (AMBIGUITIES.md #4)
    mean, std = source_x.mean(dim=0), source_x.std(dim=0)
    return (source_x - mean) / std, source_y, (target_x - mean) / std, target_y


def fit(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    adversarial: bool,
    epochs: int = 400,
    weight_update_at: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> QDANN:
    """Train QDANN, or the same model with the adversarial branch switched off.

    Both arms see the same mixed source-and-target batches, so BatchNorm statistics are
    identical and the only difference between them is the domain loss and the reversed
    gradient it sends back through G_f.
    """
    torch.manual_seed(seed)
    model = QDANN(n_features=source_x.shape[1], lambda_=1.0 if adversarial else 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = dict.fromkeys(QUANTILES, 1.0)
    for epoch in range(epochs):
        if epoch == weight_update_at:
            model.eval()
            with torch.no_grad():
                weights = update_quantile_weights(y=source_y, yhat=model(source_x)[0])
            model.train()
        for batch in torch.randperm(len(source_x)).split(batch_size):
            targets = target_x[torch.randint(len(target_x), (len(batch),))]
            # one mixed batch, so the discriminator cannot separate the domains on
            # per-domain normalization statistics (AMBIGUITIES.md #6)
            yhat, logit = model(torch.cat([source_x[batch], targets]))
            loss = quantile_loss(
                y=source_y[batch], yhat=yhat[: len(batch)], weights=weights
            )
            if adversarial:
                loss = loss + domain_loss(
                    logit=logit,
                    d=torch.cat([torch.ones(len(batch)), torch.zeros(len(batch))]),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def r_squared(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """Eq. 17."""
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(n: int = 2000, seed: int = 0) -> dict[str, float]:
    source_x, source_y, target_x, target_y = make_domains(n=n, seed=seed)
    scores = {}
    for name, adversarial in (("source_only", False), ("qdann", True)):
        model = fit(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            adversarial=adversarial,
            seed=seed,
        )
        with torch.no_grad():
            scores[name] = r_squared(y=target_y, yhat=model(target_x)[0])
    return scores


def main() -> int:
    scores = evaluate()
    for name, score in scores.items():
        print(f"{name:>12s} target R2 = {score:6.3f}")
    print(f"{'recovered':>12s}          {scores['qdann'] - scores['source_only']:+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
