# project/core/trading_agent.py
import logging
from typing import Optional, Dict, Any
import pandas as pd
from binance.client import Client
from binance.enums import *
from strategies.base import StrategyBase
import os

class TradingAgent:
    """
    A trading agent that executes trades on Binance based on signals from a given strategy.
    
    Attributes:
        client (binance.Client): Binance API client
        strategy (StrategyBase): Trading strategy to generate signals
        symbol (str): Trading pair symbol (e.g., 'VETBTC')
        timeframe (str): Timeframe for data (e.g., '1m', '5m', '1h')
        logger (logging.Logger): Logger for the agent
        test_mode (bool): Whether to run in test mode (paper trading)
        position (Optional[Dict]): Current position information
    """
    
    def __init__(
        self,
        client: Client,
        strategy: StrategyBase,
        symbol: str,
        timeframe: str,
        test_mode: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the trading agent.
        
        Args:
            client: Binance API client
            strategy: Initialized trading strategy
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            test_mode: Whether to run in test mode (default True)
            logger: Optional logger instance
        """
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.test_mode = test_mode
        self.position = None
        
        # Extract base and quote currencies from symbol (e.g., 'VETBTC' -> 'VET' and 'BTC')
        self.base_currency = symbol[:-3]  # Everything before last 3 chars
        self.quote_currency = symbol[-3:]  # Last 3 chars
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        else:
            self.logger = logger
            
        self.logger.info(f"Initialized TradingAgent for {symbol} with {strategy.__class__.__name__}")
        
    def fetch_market_data(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch market data from Binance.
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with market data
        """
        try:
            candles = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise
            
    def execute_trade(self, signal: int, price: float, quantity: float) -> Optional[Dict]:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal (-1 for sell, 1 for buy, 0 for hold)
            price: Current price
            quantity: Quantity to trade
            
        Returns:
            Dictionary with trade execution details or None if test mode
        """
        if self.test_mode:
            self.logger.info(f"TEST MODE: Would execute {'BUY' if signal == 1 else 'SELL'} order for {quantity} {self.symbol} at {price}")
            return None
            
        try:
            if signal == 1:  # Buy signal (using quote currency - BTC)
                os.system('spd-say "BUY"')

                order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                self.position = {
                    'side': 'long', 
                    'entry_price': price, 
                    'quantity': quantity,
                    'asset': self.base_currency  # We now hold VET
                }
                self.logger.info(f"Executed BUY order: {order}")
                return order
                
            elif signal == -1:  # Sell signal (using base currency - VET)
                os.system('spd-say "SELL"')
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                self.position = None  # We've sold our VET position
                self.logger.info(f"Executed SELL order: {order}")
                return order
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            raise
            
    def calculate_quantity(self, signal: int, price: float, risk_pct: float = 0.01) -> float:
        """
        Calculate trade quantity based on risk percentage and account balance.
        Considers whether we're buying (using quote currency) or selling (using base currency).
        
        Args:
            signal: Trading signal (-1 for sell, 1 for buy)
            price: Current price of the asset
            risk_pct: Percentage of balance to risk (default 1%)
            
        Returns:
            Quantity to trade
        """
        try:
            if self.test_mode:
                if signal == 1:  # Buying - use BTC balance
                    balance = 0.1  # Default test BTC balance
                else:  # Selling - use VET balance
                    balance = 1000  # Default test VET balance
            else:
                if signal == 1:  # Buying - need quote currency (BTC)
                    balance = float(self.client.get_asset_balance(asset=self.quote_currency)['free'])
                else:  # Selling - need base currency (VET)
                    balance = float(self.client.get_asset_balance(asset=self.base_currency)['free'])
            
            if signal == 1:  # Buying - calculate how much VET we can buy with our BTC
                risk_amount = balance * risk_pct
                quantity = risk_amount / price
            else:  # Selling - we can only sell the VET we have
                quantity = balance * risk_pct  # Sell a percentage of our VET holdings
            
            # Get trading pair info for quantity precision
            symbol_info = self.client.get_symbol_info(self.symbol)
            step_size = float([f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
            # min_qty = float([f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
            
            # Round quantity to proper step size and ensure it meets minimum
            quantity = round(quantity - (quantity % step_size), 8)
            
            # Also ensure we're not trying to trade more than we have
            if not self.test_mode:
                if signal == 1:  # Buying - check BTC balance
                    max_possible = balance / price
                else:  # Selling - check VET balance
                    max_possible = balance
                    
                quantity = min(quantity, max_possible)
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity: {e}")
            raise
            
    def run(self, risk_pct: float = 0.01) -> None:
        """
        Main trading loop that fetches data, generates signals, and executes trades.
        
        Args:
            risk_pct: Percentage of balance to risk per trade (default 1%)
        """
        try:
            # Fetch market data
            data = self.fetch_market_data()
            
            # Update strategy with latest data
            self.strategy.price_data = data
            
            # Generate signals
            signals = self.strategy.generate_signals()
            latest_signal = signals['position'].iloc[-1]
            
            # Skip if no signal (0)
            if latest_signal == 0:
                self.logger.info("No trading signal")
                return
                
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Calculate quantity based on signal direction
            quantity = self.calculate_quantity(
                signal=latest_signal,
                price=current_price,
                risk_pct=risk_pct
            )
            
            # Check if we have enough balance
            if quantity <= 0:
                self.logger.warning(f"Insufficient balance for {self.base_currency if latest_signal == -1 else self.quote_currency}")
                return
                
            # Execute trade
            self.execute_trade(
                signal=latest_signal,
                price=current_price,
                quantity=quantity
            )
            
        except Exception as e:
            os.system('spd-say "ERROR"')
            self.logger.error(f"Error in trading loop: {e}")
            raise