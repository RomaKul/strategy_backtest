import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase
from typing import Dict, Any, Tuple

class VWAPReversion(StrategyBase):
    """
    Стратегія повернення до VWAP (Volume Weighted Average Price).
    Відкриває позиції при значному відхиленні ціни від VWAP з очікуванням повернення.
    Використовує правильні ціни для покупки (bid) та продажу (ask) замість ціни закриття.
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
    
    def get_current_signal_prices(self) -> Tuple[float, float]:
        """
        Returns the current buy and sell trigger prices based on VWAP deviation.
        
        Returns:
            Tuple[float, float]: (buy_price, sell_price) 
            where buy_price is the price that would trigger a buy signal,
            and sell_price is the price that would trigger a sell signal.
        """
        current_vwap = self.calculate_vwap().iloc[-1]
        
        # Calculate price levels that would trigger signals
        buy_trigger_price = current_vwap * (1 - self.deviation_threshold)
        sell_trigger_price = current_vwap * (1 + self.deviation_threshold)
        
        return buy_trigger_price, sell_trigger_price

    def generate_signals(self) -> pd.DataFrame:
        """Генерує сигнали на основі відхилення від VWAP, використовуючи bid/ask ціни."""
        bid = self.price_data['bid']
        ask = self.price_data['ask']
        vwap = self.calculate_vwap()
        
        # Відхилення від VWAP у відсотках для bid (покупка) та ask (продаж)
        buy_deviation = (bid - vwap) / vwap
        sell_deviation = (ask - vwap) / vwap
        
        signals = pd.DataFrame(index=vwap.index)
        signals['position'] = 0
        
        # Умови входу:
        # Купівля, коли bid ціна значно нижче VWAP (перепроданість)
        signals.loc[buy_deviation < -self.deviation_threshold, 'position'] = 1
        
        # Продаж, коли ask ціна значно вище VWAP (перекупленість)
        signals.loc[sell_deviation > self.deviation_threshold, 'position'] = -1
        
        # Умови виходу:
        # Для довгих позицій: коли ask ціна наближається до VWAP
        long_exit_condition = ((ask - vwap) / vwap).abs() < self.exit_threshold
        signals.loc[long_exit_condition & (signals['position'] == 1), 'position'] = 0
        
        # Для коротких позицій: коли bid ціна наближається до VWAP
        short_exit_condition = ((bid - vwap) / vwap).abs() < self.exit_threshold
        signals.loc[short_exit_condition & (signals['position'] == -1), 'position'] = 0
        
        # Усунення пропусків
        signals.fillna(0, inplace=True)
        
        self.signals = signals
        return signals
    
    def run_backtest(self, **kwargs) -> pd.DataFrame:
        """More precise implementation with separate execution prices"""
        if self.signals is None:
            self.generate_signals()
        
        close = self.price_data['close']
        fees = kwargs.get('fees', 0.001)
        slippage = kwargs.get('slippage', 0.0005)
        
        # Create separate price series for each operation type
        entry_prices = self.price_data['ask'].copy()
        exit_prices = self.price_data['bid'].copy()
        
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=self.signals['position'] == 1,
            exits=self.signals['position'] == -1,
            short_entries=self.signals['position'] == -1,
            short_exits=self.signals['position'] == 1,
            fees=fees,
            slippage=slippage,
            # Map each operation to its price series
            price=(
                entry_prices.where(self.signals['position'] == 1)  # Long entries
                .fillna(exit_prices.where(self.signals['position'] == -1))  # Long exits
                .fillna(close)  # Fallback
            )
        )
        
        self.results = pf.stats()
        return pf
