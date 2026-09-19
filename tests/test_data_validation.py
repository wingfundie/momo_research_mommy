import pandas as pd
import pytest

from momo_bot.data_validation import infer_frequency, validate_price_frame


def test_infer_frequency_daily():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    assert infer_frequency(idx) == "1d"


def test_infer_frequency_4h():
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")
    assert infer_frequency(idx) == "4h"


def test_strict_frequency_mismatch_raises(tmp_path):
    df = pd.DataFrame({"BTCUSDT": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))

    with pytest.raises(ValueError):
        validate_price_frame(df, path=tmp_path / "prices.pkl", configured_frequency="4h", strict=True)


def test_auto_uses_inferred_frequency(tmp_path):
    df = pd.DataFrame({"BTCUSDT": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    result = validate_price_frame(df, path=tmp_path / "prices.pkl", configured_frequency="auto")

    assert result.inferred_frequency == "1d"
    assert result.effective_frequency == "1d"
