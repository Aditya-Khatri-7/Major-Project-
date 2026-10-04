import json
import math

import numpy as np
import pytest

from eval.metrics import (best_threshold, binary_metrics, bootstrap_ci, choose_thresholds, expected_calibration_error,
                          nan_to_none, selective_metrics, tpr_at_fpr)
from training.common import EarlyStopping, History, monitor_mode, stratified_indices


def test_binary_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05]
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(14 / 15)


def test_perfect_and_single_class_cases():
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1.0
    single = binary_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(single["roc_auc"]) and single["recall"] == 1.0
    assert nan_to_none(single)["roc_auc"] is None
    json.dumps(nan_to_none(single))                                   # valid JSON after NaN removal


def test_ece_zero_for_perfectly_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error([0, 0, 1, 1] * 50, [0.99, 0.99, 0.01, 0.01] * 50) > 0.9


def test_tpr_at_fpr_and_threshold_selection():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p = np.r_[rng.normal(0.3, 0.1, 500), rng.normal(0.7, 0.1, 500)].clip(0, 1)
    assert tpr_at_fpr(y, p, 0.05) > 0.8
    assert 0.4 < best_threshold(y, p, "f1") < 0.6
    th = choose_thresholds(y, p, 0.95, 0.95)
    assert th["t_lo"] <= th["t_star"] <= th["t_hi"] and th["t_lo"] < th["t_hi"]


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 400), 0, 1)
    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(y, p, lambda a, b: float(roc_auc_score(a, b)), n_boot=200)
    assert lo < roc_auc_score(y, p) < hi


def test_selective_metrics():
    out = selective_metrics([1, 0, 1, 0], [1, 0, 0, 1], [False, False, True, True])
    assert out["coverage"] == 0.5 and out["accuracy_on_decided"] == 1.0 and out["accuracy_on_escalated"] == 0.0


def test_early_stopping_saves_only_on_improvement_and_stops_after_patience():
    es = EarlyStopping("val_roc_auc", patience=2, min_delta=0.001)
    assert es.update(0.80, 1) and es.update(0.85, 2)                    # improvements -> save best
    assert not es.update(0.849, 3) and not es.should_stop               # 1 bad epoch
    assert not es.update(0.8505, 4)                                     # below min_delta counts as bad
    assert es.should_stop and es.best == 0.85 and es.best_epoch == 2


def test_early_stopping_recovers_and_handles_loss_and_nan():
    es = EarlyStopping("val_roc_auc", patience=3)
    es.update(0.9, 1); es.update(0.89, 2)
    assert es.update(0.95, 3) and es.num_bad == 0                       # better later result replaces the best
    loss = EarlyStopping("val_loss", patience=2)
    assert monitor_mode("val_loss") == "min" and monitor_mode("val_f1") == "max"
    assert loss.update(0.5, 1) and loss.update(0.4, 2) and not loss.update(float("nan"), 3)
    restored = EarlyStopping("val_loss", patience=2)
    restored.load_state_dict(loss.state_dict())
    assert restored.best == 0.4 and restored.best_epoch == 2


def test_history_columns_and_files(tmp_path):
    h = History()
    h.append({"epoch": 1, "val_loss": 0.5})
    h.append({"epoch": 2, "val_loss": 0.4, "val_f1": 0.9})
    assert h.columns() == {"epoch": [1, 2], "val_loss": [0.5, 0.4], "val_f1": [None, 0.9]}
    h.save(tmp_path / "h.json", tmp_path / "h.csv")
    assert json.loads((tmp_path / "h.json").read_text())["epoch"] == [1, 2]
    assert (tmp_path / "h.csv").read_text().splitlines()[0] == "epoch,val_loss,val_f1"


def test_stratified_indices_preserve_ratio():
    labels = [0] * 900 + [1] * 100
    idx = stratified_indices(labels, 100, seed=0)
    assert len(idx) == 100 and 5 <= sum(labels[i] for i in idx) <= 15
