import pandas as pd

from heart_disease.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, clean_data, load_raw


def _raw_row(**overrides):
    base = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        "num": 0,
    }
    base.update(overrides)
    return base


def test_clean_data_binarizes_target():
    raw = pd.DataFrame([_raw_row(num=0), _raw_row(num=1), _raw_row(num=3), _raw_row(num=4)])
    cleaned = clean_data(raw)
    assert list(cleaned[TARGET]) == [0, 1, 1, 1]
    assert "num" not in cleaned.columns


def test_clean_data_imputes_missing_ca_and_thal():
    raw = pd.DataFrame(
        [_raw_row(ca=None, thal=None), _raw_row(ca=1.0, thal=3.0), _raw_row(ca=2.0, thal=7.0)]
    )
    cleaned = clean_data(raw)
    assert cleaned["ca"].isna().sum() == 0
    assert cleaned["thal"].isna().sum() == 0
    # Imputed value should be the mode of the observed values (ties broken ascending).
    assert cleaned.loc[0, "ca"] == 1


def test_clean_data_produces_expected_columns():
    raw = pd.DataFrame([_raw_row(), _raw_row(num=2)])
    cleaned = clean_data(raw)
    expected_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    assert set(cleaned.columns) == expected_cols
    assert cleaned.isna().sum().sum() == 0


def test_load_raw_treats_question_mark_as_missing(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n")
    df = load_raw(csv_path)
    assert df.loc[0, "ca"] != df.loc[0, "ca"]  # NaN check without importing numpy
