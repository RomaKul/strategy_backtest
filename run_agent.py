from binance.client import Client
from strategies.vwap_reversion import VWAPReversion  # or any other strategy
from agent.binance import TradingAgent
import time

# Initialize Binance client
api_key = '0FhZdWeLefwxmQYVzi8pT1hQaR6lvC0NRn35oWiG5bJV1LutsMJgbHnK5ZJrZbQK'
api_secret = 'SheFNtiCQKhphhWCACRKQWdaTbXKyv2ZxzhgmLyvfXQQNqo8iOBmLxoBVw2nlIGu'
client = Client(api_key, api_secret)

# Initialize strategy
strategy = VWAPReversion(1)

# Create trading agent
agent = TradingAgent(
    client=client,
    strategy=strategy,
    symbol='VETBTC',
    timeframe='1m',
    test_mode=False  # Set to True for paper trading
)

# Run the agent in a loop
while True:
    agent.run(risk_pct=1)  # % risk per trade
    time.sleep(30)  # Wait for next candle