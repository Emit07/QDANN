import torch

from qdann import baselines, losses, train, weather
from test_train import table


def linear(n: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, weather.N_FEATURES, generator=generator)
    return x, x[:, 0] + 0.5 * x[:, 1]


def test_ridge_fits_a_linear_signal():
    x, y = linear()
    yhat = baselines.ridge(x.numpy(), y.numpy(), x.numpy())
    assert train.r_squared(y, torch.tensor(yhat, dtype=torch.float32)) > 0.99


def test_the_forest_fits_a_product_ridge_structurally_cannot():
    x, _ = linear()
    y = x[:, 0] * x[:, 1]
    predicted = [
        model(x.numpy(), y.numpy(), x.numpy())
        for model in (baselines.random_forest, baselines.ridge)
    ]
    forest, linear_fit = (
        train.r_squared(y, torch.tensor(p, dtype=torch.float32)) for p in predicted
    )
    assert forest > 0.8 > linear_fit


def test_the_mse_branch_carries_the_vae_weights():
    """The ablation runs quantile=False with vae_filter=True, so the weighted reduction of
    Eq. 16 has to survive the swap."""
    y, yhat = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 3.0])
    weights = torch.tensor([3.0, 1.0])
    assert float(losses.mse_loss(y, yhat)) == 5.0
    assert float(losses.mse_loss(y, yhat, sample_weights=weights)) == 3.0


def test_fit_learns_without_the_quantile_loss():
    x, y = linear(n=256)
    # dropout is 0.5 and one epoch is one batch here, so this needs the larger step
    model = train.fit(
        source_x=x, source_y=y, quantile=False, epochs=400, learning_rate=1e-2
    )
    with torch.no_grad():
        assert train.r_squared(y, model(x)[0]) > 0.5


def test_the_baselines_score_on_the_same_fold_as_qdann():
    frame = table()
    scores = baselines.evaluate(frame, epochs=20)
    assert set(scores) == {"ridge", "random_forest", "dnn"}
    assert all(set(s) == {"r2", "rmse", "nrmse"} for s in scores.values())
    # the fold is train.split's own, so the baselines are scored on the counties QDANN
    # held out and not on a second split of their own
    trained, held = train.split(frame, seed=0)
    train_x, test_x = train.standardize(
        trained, held, columns=train.feature_columns(frame)
    )
    test_y = torch.tensor(held[weather.LABEL_COLUMN].to_numpy(), dtype=torch.float32)
    yhat = baselines.ridge(
        train_x.numpy(), trained[weather.LABEL_COLUMN].to_numpy(), test_x.numpy()
    )
    assert scores["ridge"]["rmse"] == train.rmse(
        test_y, torch.tensor(yhat, dtype=torch.float32)
    )
