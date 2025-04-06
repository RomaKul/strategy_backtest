import pandas as pd
import numpy as np
import vectorbt as vbt
from strategies.base import StrategyBase
from typing import Dict, Any

class ATRTrailingBreakout(StrategyBase):
    """
    Стратегія з підтягуванням стопів по ATR та вхідними умовами на пробиття.
    Використовує ATR для визначення динамічних стоп-лосів та тейк-профітів.
    """
    
    def __init__(self, price_data: pd.DataFrame, params: Dict[str, Any] = None):
        super().__init__(price_data, params)
        self.atr_window = self.params.get('atr_window', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)
        self.lookback_period = self.params.get('lookback_period', 20)
        self.min_breakout = self.params.get('min_breakout', 0.01)
    
    def calculate_atr(self) -> pd.Series:
        """Розраховує Average True Range (ATR)."""
        return vbt.ATR.run(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close'],
            window=self.atr_window
        ).atr
    
    def generate_signals(self) -> pd.DataFrame:
        """Генерує сигнали на основі ATR та пробоїв."""
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        atr = self.calculate_atr()
        
        signals = pd.DataFrame(index=close.index)
        signals['position'] = 0
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Визначення локальних екстремумів
        rolling_high = high.rolling(window=self.lookback_period).max().shift(1)
        rolling_low = low.rolling(window=self.lookback_period).min().shift(1)
        
        # Умови входу
        breakout_up = (close > rolling_high * (1 + self.min_breakout))
        breakout_down = (close < rolling_low * (1 - self.min_breakout))
        
        # Створення сигналів з динамічними стопами
        in_long = False
        in_short = False
        
        for i in range(1, len(signals)):
            # Вихід по стопу/тейку
            if in_long:
                if close.iloc[i] < signals['stop_loss'].iloc[i-1]:
                    signals.loc[signals.index[i], 'position'] = 0
                    in_long = False
                elif close.iloc[i] > signals['take_profit'].iloc[i-1]:
                    signals.loc[signals.index[i], 'position'] = 0
                    in_long = False
                else:
                    signals.loc[signals.index[i], 'position'] = 1
                    signals.loc[signals.index[i], 'stop_loss'] = max(
                        signals['stop_loss'].iloc[i-1],
                        close.iloc[i] - atr.iloc[i] * self.atr_multiplier
                    )
                    signals.loc[signals.index[i], 'take_profit'] = close.iloc[i] + atr.iloc[i] * self.atr_multiplier * 2
            
            elif in_short:
                if close.iloc[i] > signals['stop_loss'].iloc[i-1]:
                    signals.loc[signals.index[i], 'position'] = 0
                    in_short = False
                elif close.iloc[i] < signals['take_profit'].iloc[i-1]:
                    signals.loc[signals.index[i], 'position'] = 0
                    in_short = False
                else:
                    signals.loc[signals.index[i], 'position'] = -1
                    signals.loc[signals.index[i], 'stop_loss'] = min(
                        signals['stop_loss'].iloc[i-1],
                        close.iloc[i] + atr.iloc[i] * self.atr_multiplier
                    )
                    signals.loc[signals.index[i], 'take_profit'] = close.iloc[i] - atr.iloc[i] * self.atr_multiplier * 2
            
            # Вхід у позиції
            if not in_long and not in_short:
                if breakout_up.iloc[i]:
                    signals.loc[signals.index[i], 'position'] = 1
                    signals.loc[signals.index[i], 'stop_loss'] = close.iloc[i] - atr.iloc[i] * self.atr_multiplier
                    signals.loc[signals.index[i], 'take_profit'] = close.iloc[i] + atr.iloc[i] * self.atr_multiplier * 2
                    in_long = True
                elif breakout_down.iloc[i]:
                    signals.loc[signals.index[i], 'position'] = -1
                    signals.loc[signals.index[i], 'stop_loss'] = close.iloc[i] + atr.iloc[i] * self.atr_multiplier
                    signals.loc[signals.index[i], 'take_profit'] = close.iloc[i] - atr.iloc[i] * self.atr_multiplier * 2
                    in_short = True
        
        self.signals = signals
        return signals
    
    def run_backtest(self, **kwargs) -> pd.DataFrame:
        """Виконує бектест стратегії з динамічними стопами."""
        if self.signals is None:
            self.generate_signals()
        
        close = self.price_data['close']
        fees = kwargs.get('fees', 0.001)
        slippage = kwargs.get('slippage', 0.0005)
        
        # Для цієї стратегії використовуємо більш складний підхід через динамічні стопи
        entries = self.signals['position'].diff() == 1
        exits = self.signals['position'].diff() == -1
        short_entries = self.signals['position'].diff() == -1
        short_exits = self.signals['position'].diff() == 1
        
        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
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