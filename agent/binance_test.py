import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from binance.client import Client
from binance.enums import *
from strategies.base import StrategyBase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import time
import math

class TradingAgent:
    """Enhanced trading agent with configurable order placement logic."""
    
    def __init__(
        self,
        client: Client,
        strategy: StrategyBase,
        symbol: str,
        timeframe: str,
        test_mode: bool = True,
        order_timeout: int = 300,
        risk_pct: float = 0.01,
        safety_buffer_pct: float = 0.05,  # 5% buffer from signal price
        logger: Optional[logging.Logger] = None
    ):
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.test_mode = test_mode
        self.order_timeout = order_timeout
        self.risk_pct = risk_pct
        self.safety_buffer_pct = safety_buffer_pct
        self.position = None
        self.pending_orders = {}
        self.base_currency = symbol[:-3]
        self.quote_currency = symbol[-3:]
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        else:
            self.logger = logger
            
        self.logger.info(f"Initialized TradingAgent for {symbol}")

    def fetch_market_data(self, limit: int = 51) -> pd.DataFrame:
        # Initialize Binance client

        """Fetch OHLCV market data from Binance."""
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

    def get_balance(self, asset: str) -> float:
        """Get available balance for an asset."""
        if self.test_mode:
            return 1000 if asset == self.quote_currency else 0.1
        try:
            return float(self.client.get_asset_balance(asset)['free'])
        except Exception as e:
            self.logger.error(f"Error getting {asset} balance: {e}")
            raise

    def calculate_quantity(self, price: float, is_buy: bool) -> float:
        """Calculate precise trade quantity based on risk percentage."""
        try:
            if is_buy:
                balance = self.get_balance(self.quote_currency)
                quantity = (balance * self.risk_pct) / price
            else:
                quantity = self.get_balance(self.base_currency) * self.risk_pct

            # Get lot size precision
            symbol_info = self.client.get_symbol_info(self.symbol)
            step_size = float([f['stepSize'] for f in symbol_info['filters'] 
                             if f['filterType'] == 'LOT_SIZE'][0])
            
            # Round down to nearest step size
            quantity = math.floor(quantity / step_size) * step_size
            
            return round(quantity, 8)
            
        except Exception as e:
            self.logger.error(f"Quantity calculation error: {e}")
            raise

    def determine_order_price(self, signal_price: float, is_buy: bool) -> float:
        """
        Determine order price with safety buffer.
        For buys: place at or below signal price
        For sells: place at or above signal price
        """
        price_precision = 8
        
        if is_buy:
            # Place buy order at or below signal price with buffer
            adjusted_price = signal_price * (1 - self.safety_buffer_pct)
            return round(adjusted_price, price_precision)
        else:
            # Place sell order at or above signal price with buffer
            adjusted_price = signal_price * (1 + self.safety_buffer_pct)
            return round(adjusted_price, price_precision)

    def place_limit_order(
        self, 
        side: str, 
        price: float, 
        quantity: float,
        time_in_force: str = TIME_IN_FORCE_GTC
    ) -> Optional[Dict]:
        
        """Place a limit order at specified price."""
        try:
            price = "{0:.8f}".format(price)
            self.logger.info(f"Placing {side} order for {quantity} {self.symbol} @ {price}")
            
            if self.test_mode:
                self.logger.info("TEST MODE: Order not actually placed")
                return {
                    'orderId': 'test_order',
                    'status': 'NEW',
                    'price': str(price),
                    'side': side,
                    'origQty': str(quantity)
                }
                
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=time_in_force,
                quantity=quantity,
                price=str(price)
            )
            
            # Track pending order
            self.pending_orders[order['orderId']] = {
                'side': side,
                'price': price,
                'quantity': quantity,
                'timestamp': time.time()
            }
            
            return order
            
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            raise

    def cancel_pending_orders(self) -> None:
        """Cancel all pending orders."""
        if self.test_mode:
            self.logger.info("TEST MODE: Would cancel all pending orders")
            self.pending_orders = {}
            return
            
        for order_id in list(self.pending_orders.keys()):
            try:
                self.client.cancel_order(
                    symbol=self.symbol,
                    orderId=order_id
                )
                self.logger.info(f"Cancelled pending order {order_id}")
                del self.pending_orders[order_id]
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
                # Remove from tracking even if cancellation failed
                del self.pending_orders[order_id]

    def check_pending_orders(self) -> bool:
        """
        Check status of pending orders.
        Returns True if any orders were filled, False otherwise.
        """
        if not self.pending_orders:
            return False
            
        any_filled = False
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        
        for order_id in list(self.pending_orders.keys()):
            try:
                if self.test_mode:
                    # Simulate order fill in test mode
                    order_info = {
                        'status': 'FILLED',
                        'price': self.pending_orders[order_id]['price'],
                        'executedQty': self.pending_orders[order_id]['quantity'],
                        'side': self.pending_orders[order_id]['side']
                    }
                else:
                    order_info = self.client.get_order(
                        symbol=self.symbol,
                        orderId=order_id
                    )
                
                if order_info['status'] == 'FILLED':
                    any_filled = True
                    fill_price = float(order_info['price'])
                    executed_qty = float(order_info['executedQty'])
                    side = order_info['side']
                    
                    if side == SIDE_BUY:
                        self.position = {
                            'side': 'long',
                            'entry_price': fill_price,
                            'quantity': executed_qty,
                            'asset': self.base_currency
                        }
                    else:
                        self.position = None
                        
                    self.logger.info(f"Pending {side} order filled at {fill_price}")
                    del self.pending_orders[order_id]

                # Check if price has moved against us significantly
                elif time.time() - self.pending_orders[order_id]['timestamp'] > self.order_timeout:
                    original_price = float(self.pending_orders[order_id]['price'])
                    price_diff_pct = abs(current_price - original_price) / original_price * 100
                    
                    if price_diff_pct > 10.0:  # 10% price movement against us
                        self.logger.info(f"Price moved {price_diff_pct:.2f}% against pending order")
                        self.client.cancel_order(
                            symbol=self.symbol,
                            orderId=order_id
                        )
                        del self.pending_orders[order_id]                    
                        
            except Exception as e:
                self.logger.error(f"Error checking order {order_id}: {e}")
                # Remove from tracking if we can't check status
                # del self.pending_orders[order_id]
                
        return any_filled

    def execute_trade(self, signal: int, signal_price: float) -> Optional[Dict]:
        """Execute trade based on signal price."""
        try:
            
            is_buy = signal == 1
            side = SIDE_BUY if is_buy else SIDE_SELL
            
            # Calculate order price with safety buffer
            order_price = self.determine_order_price(signal_price, is_buy)
            
            # Calculate quantity based on risk percentage
            quantity = self.calculate_quantity(order_price, is_buy)
            
            if quantity * signal_price < 0.0001:
                self.logger.warning("Invalid quantity")
                return None
                
            # Place the order
            order = self.place_limit_order(side, order_price, quantity)
            
            if self.test_mode:
                return None
                
            return order
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            raise

    def run(self) -> None:
        """Main trading loop with order monitoring."""
        try:
            self.check_pending_orders()
            
            data = self.fetch_market_data()
            self.strategy.price_data = data
            
            # Get signal prices from strategy
            buy_price, sell_price = self.strategy.get_current_signal_prices()
            
            # Get current balances
            base_balance = self.get_balance(self.base_currency)
            quote_balance = self.get_balance(self.quote_currency)
            
            # Determine which orders to place based on balances and signals
            if buy_price is not None and quote_balance > 0:
                # We have quote currency and a buy signal
                self.execute_trade(1, buy_price)
            if sell_price is not None and base_balance > 0:
                # We have base currency and a sell signal
                self.execute_trade(-1, sell_price)
            # else:
            #     self.logger.debug("No valid trading opportunity")
            
        except Exception as e:
            self.logger.error(f"Trading error: {e}")
            raise