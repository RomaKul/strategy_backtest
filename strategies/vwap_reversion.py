import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase
from typing import Dict, Any

class VWAPReversion(StrategyBase):
    """
    Стратегія повернення до VWAP (Volume Weighted Average Price).
    Відкриває позиції при значному відхиленні ціни від VWAP з очікуванням повернення.
    """
    
    def __init__(self, price_data: pd.DataFrame, params: Dict[str, Any] = None):
        super().__init__(price_data, params)
        self.vwap_window = self.params.get('vwap_window', 50)
        self.deviation_threshold = self.params.get('deviation_threshold', 0.02)
        self.exit_threshold = self.params.get('exit_threshold', 0.005)
    
    def calculate_vwap(self) -> pd.Series:
        """Розраховує VWAP (Volume Weighted Average Price)."""
        typical_price = (self.price_data['high'] + self.price_data['low'] + self.price_data['close']) / 3
        vwap = (typical_price * self.price_data['volume']).rolling(
            window=self.vwap_window
        ).sum() / self.price_data['volume'].rolling(window=self.vwap_window).sum()
        return vwap
    
    def generate_signals(self) -> pd.DataFrame:
        """Генерує сигнали на основі відхилення від VWAP."""
        close = self.price_data['close']
        vwap = self.calculate_vwap()
        
        # Відхилення від VWAP у відсотках
        deviation = (close - vwap) / vwap
        
        signals = pd.DataFrame(index=close.index)
        signals['position'] = 0
        
        # Умови входу
        signals.loc[deviation > self.deviation_threshold, 'position'] = -1  # Продаж при перекупленості
        signals.loc[deviation < -self.deviation_threshold, 'position'] = 1  # Купівля при перепроданності
        
        # Умови виходу
        signals.loc[(deviation.abs() < self.exit_threshold) & (signals['position'] != 0), 'position'] = 0
        
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