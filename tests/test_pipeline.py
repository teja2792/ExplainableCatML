import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import load_dataset, split_features_target, TARGETS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def test_dataset_exists_and_has_expected_columns():
    df = load_dataset()
    expected_columns = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + TARGETS)
    assert expected_columns.issubset(set(df.columns))


def test_dataset_no_missing_values():
    df = load_dataset()
    assert df.isnull().sum().sum() == 0


def test_activity_is_non_negative():
    df = load_dataset()
    assert (df["Activity_Rate_Constant_hr"] >= 0).all()


def test_selectivity_within_valid_range():
    df = load_dataset()
    assert (df["Selectivity_Mineralization_pct"] >= 0).all()
    assert (df["Selectivity_Mineralization_pct"] <= 100).all()


def test_green_led_has_highest_mean_selectivity():
    """
    Sanity check against the real literature anchor: green LEDs should
    produce the highest mineralization selectivity (paper reports ~100%),
    followed by amber, then red.
    """
    df = load_dataset()
    means = df.groupby("LED_Color")["Selectivity_Mineralization_pct"].mean()
    assert means["Green"] > means["Amber"] > means["Red"]


def test_split_features_target_excludes_both_targets():
    """
    Regression test for the two-target bug class: selecting features for
    one target must not accidentally leak the other target column in.
    """
    df = load_dataset()
    for target in TARGETS:
        X, y = split_features_target(df, target)
        other_targets = [t for t in TARGETS if t != target]
        for other in other_targets:
            assert other not in X.columns
        assert target not in X.columns