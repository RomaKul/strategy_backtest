import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from typing import List, Dict
import ccxt

class DataLoader:
    """
    Клас для завантаження та обробки OHLCV даних з Binance.
    Реалізує кешування даних у форматі Parquet.
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.exchange = ccxt.binance()
        os.makedirs(data_dir, exist_ok=True)
    
    def get_top_pairs(self, base: str = 'BTC', limit: int = 100) -> List[str]:
        """
        Отримує список найліквідніших торгових пар.
        
        Args:
            base: Базова валюта (за замовчуванням BTC)
            limit: Кількість пар для повернення
            
        Returns:
            Список символів торгових пар
        """
        markets = self.exchange.load_markets()
        btc_pairs = [s for s in markets if s.endswith(f'/{base}')]
        
        # Сортування за об'ємом торгів
        volumes = []
        for pair in btc_pairs:
            try:
                ticker = self.exchange.fetch_ticker(pair)
                volumes.append((pair, ticker['quoteVolume']))
            except:
                continue
        
        volumes.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in volumes[:limit]]
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                   start_date: str = '2025-02-01', end_date: str = '2025-03-01') -> pd.DataFrame:
        """
        Завантажує OHLCV дані для конкретної пари.
        
        Args:
            symbol: Торгова пара (наприклад, 'ETH/BTC')
            timeframe: Таймфрейм даних (1m, 5m тощо)
            start_date: Початкова дата у форматі 'YYYY-MM-DD'
            end_date: Кінцева дата у форматі 'YYYY-MM-DD'
            
        Returns:
            DataFrame з OHLCV даними
        """
        since = self.exchange.parse8601(f'{start_date}T00:00:00Z')
        end = self.exchange.parse8601(f'{end_date}T00:00:00Z')
        
        all_ohlcv = []
        current_since = since
        
        while current_since < end:
            data = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=current_since,
                limit=1000
            )
            if not data:
                break
                
            all_ohlcv.extend(data)
            current_since = data[-1][0] + 1
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Define the end date
        # end_date = pd.to_datetime(end_date)
        # Filter rows where timestamp <= end_date
        df = df[df.index < end_date]
        df.dropna(inplace=True)
        
        return df
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str) -> None:
        """
        Зберігає дані у форматі Parquet.
        
        Args:
            data: Словник з DataFrame для кожної пари
            filename: Ім'я файлу для збереження
        """
        # Об'єднання даних в один DataFrame з мультііндексом
        combined = pd.concat(data.values(), axis=1, keys=data.keys())
        combined.to_parquet(
            os.path.join(self.data_dir, filename), 
            engine='pyarrow', 
            compression='snappy'
        )
    
    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """
        Завантажує дані з Parquet файлу.
        
        Args:
            filename: Ім'я файлу для завантаження
            
        Returns:
            Словник з DataFrame для кожної пари
        """
        df = pd.read_parquet(os.path.join(self.data_dir, filename))
        df.dropna(inplace=True)
        return {symbol: df[symbol] for symbol in df.columns.levels[0]}