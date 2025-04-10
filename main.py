import warnings
warnings.filterwarnings("ignore")

from core.data_loader import DataLoader
import os
import sys
from core.backtester import Backtester
from strategies.atr_breakout import ATRTrailingBreakout
from strategies.multi_momentum import MultiTimeframeMomentum
from strategies.vwap_reversion import VWAPReversion
from strategies.bb_rsi import BB_RSI_VWAL_Strategy
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def main():
    # Налаштування логування
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Завантаження даних
    logger.info("Завантаження даних...")
    loader = DataLoader()    
    
    # Завантаження OHLCV даних (або з кешу, або з API)
    data_file = 'btc_1m_apr25_askbid.parquet'
    if not os.path.exists(os.path.join('data', data_file)):

        # Отримання топ-100 пар
        # top_pairs = loader.get_top_pairs(limit=100)
        # logger.info(f"Отримано {len(top_pairs)} торгових пар")
        top_pairs = ['ETHBTC', 'VETBTC', 'VIDTBTC', 'SOLBTC']
        data = {}
        for pair in tqdm(top_pairs):
            df = loader.fetch_historical_bid_ask(pair, '1m', start_date='2025-04-01', end_date='2025-05-01')
            data[pair.replace('/', '')] = df
        
        # Збереження даних
        loader.save_data(data, data_file)
    else:
        # Завантаження з кешу
        data = loader.load_data(data_file)
    
    # Keep only these pairs
    pairs_to_keep = ['ETHBTC', 'VETBTC', 'VIDTBTC', 'SOLBTC']
    data = {k: v for k, v in data.items() if k in pairs_to_keep}
    # Ініціалізація бектестера
    backtester = Backtester(data, results_dir='results'+datetime.today().strftime('%Y-%m-%d'))
    
    # Параметри стратегій
    strategies = [
        # (ATRTrailingBreakout, {'atr_window': 14, 'atr_multiplier': 2.0, 'lookback_period': 20}),
        # (MultiTimeframeMomentum, {'rsi_window': 14, 'bb_window': 20}),
        # (VWAPReversion, {'vwap_window': 50, 'deviation_threshold': 0.02}),
        (BB_RSI_VWAL_Strategy, {})
    ]
    
    # Запуск бектесту для кожної стратегії
    metrics_agg = pd.DataFrame()

    for strategy_class, params in strategies:
        logger.info(f"Запуск бектесту для {strategy_class.__name__}")
        results = backtester.run_strategy(strategy_class, params, slippage=0)
        metrics = backtester.aggregate_metrics(results, strategy_class.__name__)

        metrics_agg = pd.concat([metrics_agg, metrics], ignore_index=True)
        
        logger.info(f"Результати для {strategy_class.__name__}:")
        logger.info(f"Середня доходність: {metrics['total_return'].mean():.2f}%")
        logger.info(f"Середній Sharpe Ratio: {metrics['sharpe_ratio'].mean():.2f}")
    
    # Save all metrics to CSV
    metrics_agg.to_csv(os.path.join(backtester.results_dir, 'all_metrics_vwap_new.csv'), index=False)

    # Generate combined heatmap for all strategies
    backtester.plot_metrics_heatmap(metrics_agg)

    logger.info("Бектест завершено. Результати збережено у папці results.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Exiting gracefully...")
        sys.exit(0)  # Exit with status code 0 (success)