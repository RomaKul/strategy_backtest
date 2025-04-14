import numpy as np

class TradingAgent:
    def __init__(self, window_size=14, percentage_threshold=7, order_offset=0.01):
        """
        Initialize the trading agent.
        
        Parameters:
        - window_size: Size of the moving average window (default 14)
        - percentage_threshold: Percentage below/above MA to place orders (default 7%)
        - order_offset: How far to place limit orders from current price (default 0.01%)
        """
        self.window_size = window_size
        self.percentage_threshold = percentage_threshold
        self.order_offset = order_offset
        self.price_history = []
        self.last_filled_price = None
        self.last_action = None  # 'buy' or 'sell'
        self.current_orders = {'buy': None, 'sell': None}
        
    def update_price(self, new_price):
        """
        Update the agent with a new price point.
        
        Parameters:
        - new_price: The latest market price
        """
        self.price_history.append(new_price)
        
        # Keep only the window_size most recent prices
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
            
        # Calculate moving average if we have enough data
        if len(self.price_history) >= self.window_size:
            self._update_orders(new_price)
    
    def _update_orders(self, current_price):
        """
        Update orders based on current price and strategy.
        """
        ma_value = np.median(self.price_history)
        
        # Calculate threshold prices
        buy_threshold = ma_value * (1 - self.percentage_threshold / 100)
        sell_threshold = ma_value * (1 + self.percentage_threshold / 100)
        
        # Cancel existing orders if they're no longer valid
        self._cancel_invalid_orders(current_price, buy_threshold, sell_threshold)
        
        # Place new orders if conditions are met
        if current_price <= buy_threshold and not self.current_orders['buy']:
            if self.last_action != 'buy' or (self.last_filled_price is None or current_price < self.last_filled_price):
                buy_price = current_price * (1 - self.order_offset / 100)
                self._place_order('buy', buy_price)
                
        elif current_price >= sell_threshold and not self.current_orders['sell']:
            if self.last_action != 'sell' or (self.last_filled_price is None or current_price > self.last_filled_price):
                sell_price = current_price * (1 + self.order_offset / 100)
                self._place_order('sell', sell_price)
    
    def _place_order(self, action, price):
        """
        Place a new order (simulated).
        """
        self.current_orders[action] = price
        print(f"Placed {action} limit order at {price:.4f}")
    
    def _cancel_invalid_orders(self, current_price, buy_threshold, sell_threshold):
        """
        Cancel orders that are no longer valid.
        """
        for action in ['buy', 'sell']:
            if self.current_orders[action]:
                # Cancel buy orders if price is above buy threshold
                if action == 'buy' and current_price > buy_threshold:
                    self.current_orders[action] = None
                    print(f"Cancelled buy order as price rose above buy threshold")
                
                # Cancel sell orders if price is below sell threshold
                elif action == 'sell' and current_price < sell_threshold:
                    self.current_orders[action] = None
                    print(f"Cancelled sell order as price fell below sell threshold")
    
    def order_filled(self, action, filled_price):
        """
        Notify the agent that an order was filled.
        
        Parameters:
        - action: 'buy' or 'sell'
        - filled_price: Price at which the order was filled
        """
        print(f"{action.capitalize()} order filled at {filled_price:.4f}")
        
        # Record the filled order
        self.last_action = action
        self.last_filled_price = filled_price
        self.current_orders[action] = None
        
        # Immediately place opposite order with offset
        opposite_action = 'sell' if action == 'buy' else 'buy'
        offset = self.order_offset / 100
        
        if opposite_action == 'sell':
            new_price = filled_price * (1 + offset)
        else:
            new_price = filled_price * (1 - offset)
            
        self._place_order(opposite_action, new_price)

# Example usage
if __name__ == "__main__":
    agent = TradingAgent()
    
    # Simulate price updates
    prices = [100, 101, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]
    
    for price in prices:
        print(f"\nNew price: {price}")
        agent.update_price(price)
        
        # Simulate order fills (in a real scenario, this would come from exchange callbacks)
        if price <= 90 and agent.current_orders['buy'] and price <= agent.current_orders['buy']:
            agent.order_filled('buy', price)
        elif price >= 110 and agent.current_orders['sell'] and price >= agent.current_orders['sell']:
            agent.order_filled('sell', price)