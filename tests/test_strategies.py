import pytest
import pandas as pd
import numpy as np
from strategies.vwap_reversion import VWAPReversion
from strategies.multi_momentum import MultiTimeframeMomentum
from strategies.atr_breakout import ATRTrailingBreakout

@pytest.fixture
def sample_data():
    """Фікстура з тестовими даними."""
    date_rng = pd.date_range(start='2025-02-01', end='2025-02-02', freq='1min')
    np.random.seed(42)
    close = np.cumprod(1 + np.random.randn(len(date_rng)) * 0.001) * 100
    high = close * (1 + np.random.rand(len(date_rng)) * 0.01)
    low = close * (1 - np.random.rand(len(date_rng)) * 0.01)
    volume = np.random.randint(100, 1000, size=len(date_rng))
    
    return pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_rng)

def test_vwap_reversion(sample_data):
    """Тест стратегії VWAP Reversion."""
    strategy = VWAPReversion(sample_data, {'vwap_window': 30, 'deviation_threshold': 0.01})
    signals = strategy.generate_signals()
    
    assert isinstance(signals, pd.DataFrame)
    assert not signals.empty
    assert 'position' in signals.columns
    assert signals['position'].isin([-1, 0, 1]).all()

def test_multi_momentum(sample_data):
    """Тест стратегії Multi-timeframe Momentum."""
    strategy = MultiTimeframeMomentum(sample_data, {
        'rsi_window': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    })
    signals = strategy.generate_signals()
    
    assert isinstance(signals, pd.DataFrame)
    assert not signals.empty
    assert 'position' in signals.columns
    assert signals['position'].isin([-1, 0, 1]).all()

def test_atr_breakout(sample_data):
    """Тест стратегії ATR Trailing Breakout."""
    strategy = ATRTrailingBreakout(sample_data, {
        'atr_window': 14,
        'atr_multiplier': 2.0,
        'lookback_period': 20
    })
    signals = strategy.generate_signals()
    
    assert isinstance(signals, pd.DataFrame)
    assert not signals.empty
    assert all(col in signals.columns for col in ['position', 'stop_loss', 'take_profit'])
    assert signals['position'].isin([-1, 0, 1]).all()