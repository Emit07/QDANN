import pytest
import torch

from qdann import vae, weather


def low_rank(n: int = 512, seed: int = 0) -> torch.Tensor:
    """Feature vectors lying on a 3-d subspace, which a latent of 8 can represent exactly."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 3, generator=generator)
    return z @ torch.randn(3, weather.N_FEATURES, generator=generator)


def test_the_kl_term_is_zero_on_the_prior_and_grows_with_the_mean():
    x = torch.zeros(4, 3)
    zeros = torch.zeros(4, 2)
    assert vae.vae_loss(x, x, mu=zeros, logvar=zeros) == pytest.approx(0.0)
    # KL of N(1, 1) from N(0, 1) is 0.5 per latent dimension
    assert vae.vae_loss(x, x, mu=zeros + 1, logvar=zeros) == pytest.approx(1.0)


def test_the_vae_reconstructs_a_low_rank_signal():
    x = low_rank()
    errors = vae.reconstruction_error(vae.fit(x, epochs=200), x)
    assert float(errors.mean()) < 0.1 * float(((x - x.mean(0)) ** 2).sum(dim=1).mean())


def test_a_trailing_batch_of_one_does_not_break_batchnorm():
    vae.fit(torch.randn(257, weather.N_FEATURES), epochs=1)


def test_the_threshold_is_the_mean_except_for_winter_wheat():
    errors = torch.arange(10.0)
    assert vae.filter_threshold(errors, "maize") == pytest.approx(4.5)
    assert vae.filter_threshold(errors, "winter_wheat") == pytest.approx(7.2)
