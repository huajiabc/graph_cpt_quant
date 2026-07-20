import numpy as np
import pandas as pd

from pressure_graph.reports.v2327_multisource_interaction_ridge_oco_selector import (
    _predict_from_operator,
    _prepare_designs,
    _ridge_prediction_operator,
    write_v2327_multisource_interaction_ridge_oco_selector,
)


def test_ridge_operator_learns_simple_relation() -> None:
    train_x = np.arange(10, dtype=float)[:, None]
    predict_x = np.array([[10.0], [11.0]])
    train_design, predict_design = _prepare_designs(
        train_x, predict_x, "linear_ridge"
    )
    operator = _ridge_prediction_operator(train_design, predict_design, 0.01)
    prediction = _predict_from_operator(operator, 2.0 * train_x[:, 0] + 1.0)
    assert np.allclose(prediction, [21.0, 23.0], atol=0.05)


def test_v2327_real_report_reconciles() -> None:
    paths = write_v2327_multisource_interaction_ridge_oco_selector()
    predictions = pd.read_parquet(paths["predictions"])
    summary = pd.read_csv(paths["summary"])
    ablation = pd.read_csv(paths["posthoc_ablation"])
    gates = pd.read_csv(paths["gates"])
    assert len(predictions) == 192
    assert set(predictions["model"]) == {"linear_ridge", "interaction_ridge"}
    assert set(summary["scope"]) == {"validation", "holdout", "oos"}
    assert len(gates) == 11
    assert len(ablation) == 30
    assert not ablation["promotion_eligible"].any()
    assert predictions["predicted_primary_net_return"].notna().all()
