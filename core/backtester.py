import pandas as pd
from typing import Dict, List
import vectorbt as vbt
from strategies.base import StrategyBase
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Backtester:
    """
    Клас для виконання бектесту торгових стратегій.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], results_dir: str = 'results'):
        """
        Ініціалізація бектестера.
        
        Args:
            data: Словник з OHLCV даними для кожної пари
            results_dir: Директорія для збереження результатів
        """
        self.data = data
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'screenshots'), exist_ok=True)
    
    def run_strategy(self, strategy_class: type, strategy_params: Dict = None, 
                    fees: float = 0.001, slippage: float = 0.0005) -> Dict[str, Dict]:
        """
        Запускає стратегію для всіх пар.
        
        Args:
            strategy_class: Клас стратегії (нащадок StrategyBase)
            strategy_params: Параметри стратегії
            fees: Рівень комісій
            slippage: Рівень сліппейджу
            
        Returns:
            Словник з результатами для кожної пари
        """
        all_results = {}
        
        for symbol, price_data in tqdm(self.data.items(), desc="Processing symbols"):
            try:
                strategy = strategy_class(price_data, strategy_params)
                strategy.generate_signals()
                pf = strategy.run_backtest(fees=fees, slippage=slippage)
                
                # Збереження результатів
                all_results[symbol] = {
                    'metrics': strategy.get_metrics(),
                    'portfolio': pf
                }
                
                # Збереження графіків
                self._save_plots(pf, symbol, strategy_class.__name__)
                
            except Exception as e:
                print(f"Помилка при бектесті {symbol}: {str(e)}")
                continue
                
        return all_results
    
    def _save_plots(self, portfolio, symbol: str, strategy_name: str) -> None:
        """
        Зберігає графіки результатів бектесту.
        Використовує Plotly для збереження графіків.
        """
        import plotly.io as pio
        
        # Створюємо директорію для збереження, якщо її немає
        os.makedirs(os.path.join(self.results_dir, f'screenshots_{strategy_name}'), exist_ok=True)
        
        # Equity curve
        fig = portfolio.plot(subplots=['orders', 'trade_pnl', 'cum_returns'])
        
        # Додатково зберігаємо як статичне зображення (PNG)
        png_filename = os.path.join(
            self.results_dir,
            f'screenshots_{strategy_name}',
            f'{symbol}_equity.png'
        )
        fig.write_image(png_filename, scale=2)
    
    def aggregate_metrics(self, all_results: Dict[str, Dict], strategy_name: str) -> pd.DataFrame:
        """
        Агрегує метрики по всіх парах.
        
        Args:
            all_results: Результати бектесту для всіх пар
            
        Returns:
            DataFrame з агрегованими метриками
        """
        metrics_list = []
        
        for symbol, result in all_results.items():
            metrics = result['metrics']
            metrics['symbol'] = symbol
            metrics['strategy_name'] = strategy_name
            metrics_list.append(metrics)
            
        df = pd.DataFrame(metrics_list)
        
        return df
    
    def plot_metrics_heatmap(self, metrics_df: pd.DataFrame) -> None:
        """
        Створює heatmap продуктивності стратегій.
        """
        pivot = metrics_df.pivot_table(
            index='symbol',
            columns='strategy_name',
            values='total_return',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='RdYlGn')
        plt.title('Productivity Heatmap by Symbol')
        plt.savefig(
            os.path.join(self.results_dir, 'screenshots', 'performance_heatmap.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()