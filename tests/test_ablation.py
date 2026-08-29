from qdann import ablation
from test_train import table


def test_the_table_has_a_row_per_component_and_a_full_model_row():
    """Wiring only: on a synthetic generator the deltas have no direction to assert -- the
    paper's own Fig. 16 has components helping one crop and hurting another."""
    source = table()
    ret = ablation.ablation(source, source.copy(), "maize", epochs=5, seeds=range(2))
    assert list(ret["removed"]) == ["none", *ablation.COMPONENTS]
    assert ret["delta_r2"].notna().all()
    assert ret["delta_r2"].iloc[0] == 0.0
