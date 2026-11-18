"""Tests for model configs, overrides and output-folder logic of src/train_models.py."""
import json

import pytest

from src.train_models import _experiment_tag, apply_params, load_model_config, make_models


def test_repo_configs_load_with_owner():
    models = make_models(42)
    assert "logistic_regression" in models and "gaussian_nb" in models
    family, conf, factory, owner = models["logistic_regression"]
    assert owner and family == "Linear"
    assert factory()[-1].C == 1.0             # StandardScaler + LogisticRegression


def test_config_from_file(tmp_path):
    p = tmp_path / "rf.json"
    p.write_text(json.dumps({"name": "rf", "owner": "X", "family": "Bagging",
                             "estimator": "sklearn.ensemble.RandomForestClassifier",
                             "params": {"n_estimators": 5}}))
    name, (family, conf, factory, owner) = load_model_config(p, seed=7)
    model = factory()
    assert name == "rf" and owner == "X" and model.random_state == 7   # seed injected
    assert "RandomForestClassifier(n_estimators=5)" in conf


def test_only_sklearn_estimators_allowed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"name": "bad", "owner": "X", "family": "?",
                             "estimator": "os.system", "params": {}}))
    with pytest.raises(ValueError):
        load_model_config(p, seed=0)


def test_override_reaches_final_estimator():
    models = apply_params(make_models(42), {"logistic_regression": {"C": 0.1},
                                            "gaussian_nb": {"var_smoothing": 1e-6}})
    assert models["logistic_regression"][2]()[-1].C == 0.1
    assert models["gaussian_nb"][2]().var_smoothing == 1e-6
    assert make_models(42)["logistic_regression"][2]()[-1].C == 1.0   # defaults untouched


def test_bad_names_fail_before_training():
    with pytest.raises(ValueError):
        apply_params(make_models(42), {"logistic_regresion": {"C": 0.1}})
    with pytest.raises(ValueError):
        apply_params(make_models(42), {"logistic_regression": {"not_a_param": 1}})


def test_experiment_tag():
    assert _experiment_tag(["logistic_regression"], {"logistic_regression": {"C": 0.1}}) \
        == "logistic_regression_C0.1"
