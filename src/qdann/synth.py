"""Synthetic domain shift with known target labels, to verify QDANN end to end.

Section 3.1 argues the scale transfer is a marginal shift with a stable posterior:
p_s(x) != p_t(x) while p_s(y|x) ~= p_t(y|x). This generator reproduces exactly that and
nothing else -- it is a diagnostic, not a simulation of maize.

Because the generator supplies target labels, this measures what the real pipeline cannot:
whether the adversarial branch recovers accuracy that a source-only model loses.

  uv run python -m qdann.synth
"""

import torch

from qdann import train


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


def evaluate(n: int = 2000, seed: int = 0) -> dict[str, float]:
    source_x, source_y, target_x, target_y = make_domains(n=n, seed=seed)
    scores = {}
    for name, adversarial in (("source_only", False), ("qdann", True)):
        model = train.fit(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            adversarial=adversarial,
            seed=seed,
        )
        with torch.no_grad():
            scores[name] = train.r_squared(y=target_y, yhat=model(target_x)[0])
    return scores


def main() -> int:
    scores = evaluate()
    for name, score in scores.items():
        print(f"{name:>12s} target R2 = {score:6.3f}")
    print(f"{'recovered':>12s}          {scores['qdann'] - scores['source_only']:+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
