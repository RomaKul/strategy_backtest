import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase
from typing import Dict, Any, Tuple

class BB_RSI_VWAL_Strategy(StrategyBase):
    """
    Long-only strategy using Bollinger Bands, adaptive RSI, and aggregated VWAL.
    Entry signals require:
    1. Price below lower Bollinger Band (oversold)
    2. RSI below adaptive lower boundary
    3. Positive VWAL momentum (aggregated on 5-min timeframe)
    Exit signals occur when:
    1. Price crosses above middle Bollinger Band
    2. RSI crosses above adaptive upper boundary
    """
    
    def __init__(self, price_data: pd.DataFrame, params: Dict[str, Any] = None):
        super().__init__(price_data, params)
        # self.price_data = self.price_data.resample('1h').agg({
        #     'open': 'first',
        #     'high': 'max',
        #     'low': 'min',
        #     'close': 'last',
        #     'volume': 'sum'
        # })

        # # Calculate minimum tick size
        # sorted_prices = np.sort(self.price_data['close'].unique())
        # min_tick = np.min(np.diff(sorted_prices))
        
        # # Handle edge case (if all prices are equal)
        # if np.isnan(min_tick) or min_tick <= 0:
        #     min_tick = 0.000000001  # Default BTC tick size
        
        # # Create bid/ask columns
        # self.price_data = self.price_data.copy()
        # self.price_data['bid'] = self.price_data['close'] - min_tick
        # self.price_data['ask'] = self.price_data['close'] + min_tick

        # Bollinger Bands parameters
        self.bb_window = self.params.get('bb_window', 20)
        self.bb_std = self.params.get('bb_std', 2)
        
        # RSI parameters
        self.rsi_window = self.params.get('rsi_window', 14)
        self.rsi_upper_percentile = self.params.get('rsi_upper_percentile', 0.7)  # 70th percentile
        self.rsi_lower_percentile = self.params.get('rsi_lower_percentile', 0.3)  # 30th percentile
        
        # For adaptive RSI boundaries calculation
        self.lookback_period = self.params.get('lookback_period', 24*60)  # dots for 1 min 24*60 to get day
    
    def calculate_adaptive_rsi_boundaries(self, rsi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate adaptive RSI boundaries based on historical RSI distribution.
        Uses percentiles to determine overbought/oversold levels.
        """
        # Calculate rolling percentiles
        upper_bound = rsi.rolling(window=self.lookback_period).quantile(self.rsi_upper_percentile)  # assuming minute data
        lower_bound = rsi.rolling(window=self.lookback_period).quantile(self.rsi_lower_percentile)
        
        return upper_bound, lower_bound
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on BB, RSI, and VWAL conditions."""
        close = self.price_data['close']
        
        # Calculate Bollinger Bands
        bb = vbt.talib('BBANDS').run(
            close, 
            timeperiod=self.bb_window, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        bb_upper = bb.upperband
        bb_middle = bb.middleband
        bb_lower = bb.lowerband
        
        # Calculate RSI and adaptive boundaries
        rsi = vbt.IndicatorFactory.from_talib("RSI").run(close, timeperiod=self.rsi_window).real
        rsi_upper, rsi_lower = self.calculate_adaptive_rsi_boundaries(rsi)
        # rsi_upper, rsi_lower = 30, 70
        
        signals = pd.DataFrame(index=close.index)
        signals['position'] = 0
        
        # Entry conditions (all must be true):
        # 1. Price below lower Bollinger Band
        # 2. RSI below adaptive lower boundary
        entry_condition = (
            (close < bb_lower) &
            (rsi < rsi_lower)
        )
        # Exit conditions (either condition):
        # 1. Price crosses above middle Bollinger Band
        # 2. RSI crosses above adaptive upper boundary
        exit_condition = (
            (close > bb_upper) & 
            (rsi > rsi_upper)
        )
        
        # Generate signals
        signals.loc[:, 'position'] = 0
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[exit_condition, 'position'] = -1
        
        # Ensure we're always in the market at most one long position
        # signals['position'] = signals['position'].replace(to_replace=0, method='ffill')
        # signals['position'].iloc[0] = 0  # start with no position
        
        self.signals = signals
        return signals
    
    def run_backtest(self, **kwargs) -> pd.DataFrame:
        """Run backtest with precise execution prices (ask for entries, bid for exits)"""
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
            entries=self.signals['position'] == 1,  # position change from 0 to 1
            exits=self.signals['position'] == -1,   # position change from 1 to 0
            fees=fees,
            slippage=slippage,
            price=(
                entry_prices.where(self.signals['position'] == 1)  # entries at ask price
                .fillna(exit_prices.where(self.signals['position'] == -1))  # exits at bid price
                .fillna(close)  # fallback
            )
        )
        
        self.results = pf.stats()
        return pf