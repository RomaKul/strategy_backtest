from binance.client import Client
from agent.binance_BB import TradingAgent
import time
import sys

# Initialize Binance client
with open('config.txt', 'r') as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

client = Client(api_key, api_secret, {"timeout": 30})

# Create trading agent
agent = TradingAgent(
    client=None,
    strategy=None,
    symbol='SOLBTC',
    timeframe='1h',
    risk_pct=0.99,
    test_mode=False  # Set to True for paper trading
)

# Run the agent in a loop
while True:
    try:
        agent.run()  # % risk per trade
        time.sleep(10)  # Wait for next candle
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Exiting gracefully...")
        agent.cancel_pending_orders()
        sys.exit(0)  # Exit with status code 0 (success)
