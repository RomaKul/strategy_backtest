from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class StrategyBase(ABC):
    """
    Абстрактний базовий клас для торгових стратегій.
    Усі стратегії повинні наслідувати цей клас.
    """
    
    def __init__(self, price_data: pd.DataFrame, params: Dict[str, Any] = None):
        """
        Ініціалізація стратегії.
        
        Args:
            price_data: DataFrame з OHLCV даними
            params: Словник з параметрами стратегії
        """
        self.price_data = price_data
        self.params = params or {}
        self.signals = None
        self.results = None
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Генерує торгові сигнали на основі цінових даних.
        
        Returns:
            DataFrame з торговими сигналами
        """
        pass
    
    @abstractmethod
    def run_backtest(self, **kwargs) -> pd.DataFrame:
        """
        Виконує бектест стратегії.
        
        Returns:
            DataFrame з результатами бектесту
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Обчислює метрики продуктивності стратегії.
        
        Returns:
            Словник з метриками
        """
        pass
    
    def plot_results(self):
        """
        Візуалізує результати бектесту.
        """
        if self.results is None:
            raise ValueError("Спочатку виконайте бектест (run_backtest)")
        
        # Тут може бути реалізація візуалізації з використанням Plotly або Matplotlib
        pass