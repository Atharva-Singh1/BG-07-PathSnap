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
    assert models["gaussian_nb"][2]()[-1].var_smoothing == 1e-6   # final step of the NB pipeline
    assert make_models(42)["logistic_regression"][2]()[-1].C == 1.0   # defaults untouched


def test_bad_names_fail_before_training():
    with pytest.raises(ValueError):
        apply_params(make_models(42), {"logistic_regresion": {"C": 0.1}})
    with pytest.raises(ValueError):
        apply_params(make_models(42), {"logistic_regression": {"not_a_param": 1}})


def test_experiment_tag():
    assert _experiment_tag(["logistic_regression"], {"logistic_regression": {"C": 0.1}}) \
        == "logistic_regression_C0.1"


def test_power_transform_and_drop_features_via_params():
    import numpy as np
    from src.features import FEATURE_NAMES
    models = apply_params(make_models(42), {"gaussian_nb": {
        "preprocess": "power", "drop_features": ["dist_rank", "dist_gap_m"], "var_smoothing": 1e-6}})
    family, conf, factory, owner = models["gaussian_nb"]
    m = factory()
    assert type(m[-1]).__name__ == "GaussianNB" and m[-1].var_smoothing == 1e-6
    assert "PowerTransformer" in conf and "dist_rank" in conf
    X = np.random.default_rng(0).normal(size=(50, len(FEATURE_NAMES)))
    y = np.r_[np.zeros(25), np.ones(25)]
    m.fit(X, y)
    assert m[-1].theta_.shape[1] == len(FEATURE_NAMES) - 2      # two features dropped
    assert make_models(42)["logistic_regression"][2]()[-1].C == 1.0   # other models unchanged


def test_unknown_feature_or_preprocess_fails_early():
    with pytest.raises(ValueError, match="unknown features"):
        apply_params(make_models(42), {"gaussian_nb": {"drop_features": ["no_such_feature"]}})
    with pytest.raises(ValueError, match="preprocess"):
        apply_params(make_models(42), {"gaussian_nb": {"preprocess": "magic"}})


def test_repo_gaussian_nb_config_uses_power_transform_and_drops_distance_duplicates():
    family, conf, factory, owner = make_models(42)["gaussian_nb"]
    assert "PowerTransformer" in conf and "dist_rank" in conf and "dist_gap_m" in conf
    assert type(factory()[-1]).__name__ == "GaussianNB"
