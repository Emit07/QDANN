import numpy
import pandas
import pytest
import torch

from qdann import train, weather


def table(n_counties: int = 40, years: range = range(2008, 2019)) -> pandas.DataFrame:
    """A county-year frame shaped like the real one: a county effect and a year effect.

    The harmonic columns carry the county, the weather columns carry the year -- each
    redundantly, as the real ones are, so that dropout cannot delete the signal. A model
    reading only the weather reaches the per-year mean and no further.
    """
    generator = numpy.random.default_rng(0)
    rows = []
    county_effect = generator.normal(size=n_counties)
    year_effect = generator.normal(size=len(years))
    n_harmonic = len(weather.HARMONIC_COLUMNS)
    for i in range(n_counties):
        for j, year in enumerate(years):
            features = 0.3 * generator.normal(size=weather.N_FEATURES)
            features[:n_harmonic] += county_effect[i]
            features[n_harmonic:] += year_effect[j]
            rows.append(
                {
                    "fips": f"19{i:03d}",
                    "year": year,
                    "yield_t_ha": 10 + 2 * county_effect[i] + year_effect[j],
                    **dict(zip(columns(), features, strict=True)),
                }
            )
    return pandas.DataFrame(rows)


def columns() -> list[str]:
    return [*weather.HARMONIC_COLUMNS, *weather.weather_columns("maize")]


def test_whole_counties_are_held_out():
    trained, held = train.split(table())
    assert not set(trained["fips"]) & set(held["fips"])
    assert held["fips"].nunique() == 8
    assert len(trained) + len(held) == 440


def test_standardization_reads_the_training_counties_only():
    trained, held = train.split(table())
    held = held.copy()
    held[columns()] += 10.0
    train_x, test_x = train.standardize(trained, held, columns=columns())
    assert float(train_x.mean().abs()) < 1e-5
    # the holdout keeps its offset instead of being centred on its own mean
    assert float(test_x.mean()) > 5.0


def test_a_trailing_batch_of_one_does_not_break_batchnorm():
    x = torch.randn(257, weather.N_FEATURES)
    train.fit(source_x=x, source_y=torch.randn(257), epochs=1, weight_update_at=99)


def test_the_per_year_mean_falls_back_for_an_unseen_year():
    trained, held = train.split(table())
    held = held.copy()
    held.loc[held.index[0], "year"] = 1999
    baseline = train.per_year_mean(trained, held)
    assert float(baseline[0]) == pytest.approx(trained["yield_t_ha"].mean())
    assert not baseline.isnan().any()


def test_the_model_beats_the_per_year_mean_on_a_county_signal():
    """The Phase 5 gate, on data where the answer is known.

    Half the variance is county-to-county, so a model that reads the harmonic features
    must clear the per-year mean; if the wiring leaks the holdout or drops the features,
    it will not.
    """
    scores = train.evaluate(table(), epochs=800)
    assert scores["r2"] > scores["per_year_mean_r2"]
    assert scores["rmse"] < scores["per_year_mean_rmse"]


class _SignDiscriminator:
    """A model whose discriminator reads feature 0's sign, so the metric has an answer."""

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0], x[:, 0]


def test_domain_accuracy_reads_source_as_the_positive_class():
    """1.0 on separable domains, 0.5 on identical ones -- and 0.0 if the labels flip."""
    ones = torch.ones(8, weather.N_FEATURES)
    model = _SignDiscriminator()
    assert train.domain_accuracy(model, source_x=ones, target_x=-ones) == 1.0
    assert train.domain_accuracy(model, source_x=ones * 0, target_x=ones * 0) == 0.5


def test_the_target_is_standardized_on_the_training_counties():
    trained, held = train.split(table())
    target = held.copy()
    target[columns()] += 10.0
    train_x, test_x, target_x = train.standardize(
        trained, held, target, columns=columns()
    )
    assert float(train_x.mean().abs()) < 1e-5
    assert float((target_x - test_x).mean()) > 5.0
