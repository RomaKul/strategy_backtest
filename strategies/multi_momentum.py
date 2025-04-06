# strategies/multi_momentum.py
import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase
from typing import Dict, Any

class MultiTimeframeMomentum(StrategyBase):
    """
    Стратегія комбінування імпульсу з різних таймфреймів (1m + 15m).
    Використовує RSI на 1-хвилинному таймфреймі та MACD на 15-хвилинному.
    """
    
    def __init__(self, price_data: pd.DataFrame, params: Dict[str, Any] = None):
        super().__init__(price_data, params)
        self.rsi_window = self.params.get('rsi_window', 14)
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)
        self.resample_period = self.params.get('resample_period', '15min')
    
    def resample_data(self) -> pd.DataFrame:
        """Ресемплінг даних для вищого таймфрейму."""
        resampled = self.price_data.resample(self.resample_period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return resampled
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.Series:
        """Розраховує MACD для ресемпленого таймфрейму."""
        macd = vbt.MACD.run(
            data['close'],
            fast_window=self.macd_fast,
            slow_window=self.macd_slow,
            signal_window=self.macd_signal
        )
        return macd.macd - macd.signal  # MACD Histogram
    
    def calculate_rsi(self) -> pd.Series:
        """Розраховує RSI для 1-хвилинного таймфрейму."""
        return vbt.RSI.run(self.price_data['close'], window=self.rsi_window).rsi
    
    def generate_signals(self) -> pd.DataFrame:
        """Генерує сигнали на основі мультитаймфреймового аналізу."""
        # 1-хвилинний RSI
        rsi = self.calculate_rsi()
        
        # 15-хвилинний MACD
        resampled_data = self.resample_data()
        macd_signal = self.calculate_macd(resampled_data)
        
        # Вирівнюємо MACD до 1-хвилинного таймфрейму
        macd_signal = macd_signal.reindex(self.price_data.index, method='ffill')
        
        signals = pd.DataFrame(index=self.price_data.index)
        signals['position'] = 0
        
        # Комбіновані умови входу
        buy_condition = (rsi < 30) & (macd_signal > 0)
        sell_condition = (rsi > 70) & (macd_signal < 0)
        
        signals.loc[buy_condition, 'position'] = 1
        signals.loc[sell_condition, 'position'] = -1
        
        # Усунення пропусків
        signals.fillna(0, inplace=True)
        
        self.signals = signals
        return signals
    
    def run_backtest(self, **kwargs) -> pd.DataFrame:
        """Виконує бектест стратегії."""
        if self.signals is None:
            self.generate_signals()
        
        close = self.price_data['close']
        fees = kwargs.get('fees', 0.001)
        slippage = kwargs.get('slippage', 0.0005)
        
        pf = vbt.Portfolio.from_signals(
            close,
            entries=self.signals['position'] == 1,
            exits=self.signals['position'] == -1,
            short_entries=self.signals['position'] == -1,
            short_exits=self.signals['position'] == 1,
            fees=fees,
            slippage=slippage,
            freq='1m'
        )
        
        self.results = pf.stats()
        return pf
    
    def get_metrics(self) -> Dict[str, float]:
        """Повертає метрики продуктивності стратегії."""
        if self.results is None:
            raise ValueError("Спочатку виконайте бектест (run_backtest)")
        
        return {
            'total_return': self.results['Total Return [%]'],
            'sharpe_ratio': self.results['Sharpe Ratio'],
            'max_drawdown': self.results['Max Drawdown [%]'],
            'win_rate': self.results['Win Rate [%]'],
            'expectancy': self.results['Expectancy'],
            'exposure_time': self.results['Avg Winning Trade Duration'] + self.results['Avg Losing Trade Duration']
        }