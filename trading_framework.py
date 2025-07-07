# -*- coding: utf-8 -*-
"""
Final Quantitative Trading Framework with Telegram Integration

This script is designed to be run automatically via GitHub Actions.
It performs a full backtest and sends the performance summary and plots
to a specified Telegram chat.

It accepts the Telegram Bot Token and Chat ID as command-line arguments
to ensure credentials are not hardcoded.
"""

# @title 1. Setup and Imports
import sys
import logging
import argparse
import requests
from io import BytesIO
from typing import Dict, List
from dataclasses import dataclass
import warnings

# --- Standard Library Imports ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    from scipy.optimize import minimize
except ImportError:
    print("Error: Required packages are not installed. Please install them using requirements.txt")
    sys.exit(1)


# --- Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

@dataclass
class TradingConfig:
    """Configuration class for trading parameters"""
    universe: List[str] = None
    benchmark: str = 'SPY'
    start_date: str = '2020-01-01' # Using a shorter period for faster execution in automation
    end_date: str = '2023-12-31'
    risk_free_rate: float = 0.02
    rebalance_frequency: str = 'M'
    transaction_cost: float = 0.001

    def __post_init__(self):
        if self.universe is None:
            self.universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT', 'XOM']

# --- The rest of the classes (DataManager, RegimeDetector, SignalGenerator, etc.) remain the same ---
# (The full code for these classes is omitted here for brevity but should be included in your file)
# You can copy them from the previous "Refined" version of the script.
# The key changes are in the PerformanceAnalyzer and the main execution block.

class DataManager:
    """Manages fetching and processing of market data."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data = {}
    
    def fetch_data(self):
        logger.info(f"Fetching data for {len(self.config.universe)} assets...")
        all_tickers = self.config.universe + [self.config.benchmark]
        raw_data = yf.download(all_tickers, start=self.config.start_date, end=self.config.end_date, progress=False, threads=True)
        prices = raw_data['Adj Close']
        
        # Validation and cleaning
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=True)
        
        self.data['prices'] = prices
        self.data['returns'] = prices.pct_change().dropna()
        logger.info("Data fetching completed.")
        return self.data

class SignalGenerator:
    """Generates trading signals from market data."""
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def generate_all_signals(self):
        prices = self.data_manager.data['prices'][self.config.universe]
        returns = self.data_manager.data['returns'][self.config.universe]
        
        # Momentum signals
        momentum_signals = pd.DataFrame(index=prices.index)
        for period in [20, 60]:
            mean_ret = returns.rolling(period).mean()
            vol_ret = returns.rolling(period).std()
            momentum_signals[f'mom_{period}'] = (mean_ret / vol_ret).rank(axis=1, pct=True)
        
        return {'momentum': momentum_signals}

class EnhancedAlphaEnsemble:
    """ML ensemble model to predict alpha."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, signal_dict, returns):
        features = signal_dict['momentum']
        target = returns.shift(-21).stack() # 21-day forward returns
        features_stacked = features.stack()
        combined = pd.concat([features_stacked, target], axis=1).dropna()
        combined.columns = list(features_stacked.columns) + ['target']
        return combined.drop('target', axis=1), combined['target']

    def train(self, X, y):
        logger.info("Training ML model...")
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        logger.info("Model training complete.")

    def predict(self, X):
        if not self.is_trained: raise RuntimeError("Model not trained.")
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict(X_scaled), index=X.index)

class RiskManager:
    """Manages portfolio optimization."""
    def __init__(self, config: TradingConfig):
        self.config = config

    def optimize_portfolio(self, expected_returns, cov_matrix):
        num_assets = len(expected_returns)
        def objective(weights):
            ret = np.sum(expected_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(ret - self.config.risk_free_rate) / vol
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1./num_assets] * num_assets)
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=expected_returns.index)

class Backtester:
    """Runs the backtest simulation."""
    def __init__(self, config, data_manager, alpha_model, risk_manager):
        self.config, self.data_manager, self.alpha_model, self.risk_manager = config, data_manager, alpha_model, risk_manager

    def run_backtest(self):
        logger.info("Starting backtest...")
        returns = self.data_manager.data['returns'][self.config.universe]
        
        signals = SignalGenerator(self.data_manager).generate_all_signals()
        X, y = self.alpha_model.prepare_features(signals, returns)
        self.alpha_model.train(X, y)
        predictions = self.alpha_model.predict(X)
        expected_returns = predictions.unstack()
        
        rebalance_dates = returns.resample(self.config.rebalance_frequency).first().index
        weights_history = pd.DataFrame(index=returns.index, columns=self.config.universe)
        
        for i in range(len(rebalance_dates) - 1):
            rebal_date = rebalance_dates[i]
            if rebal_date not in expected_returns.index: continue
            
            current_expected_returns = expected_returns.loc[rebal_date]
            cov_matrix = returns.loc[:rebal_date].tail(60).cov()
            optimal_weights = self.risk_manager.optimize_portfolio(current_expected_returns, cov_matrix)
            weights_history.loc[rebal_date:rebalance_dates[i+1]] = optimal_weights.values

        weights_history = weights_history.ffill().dropna()
        portfolio_returns = (returns * weights_history.shift(1)).sum(axis=1)
        turnover = weights_history.diff().abs().sum(axis=1)
        transaction_costs = turnover * self.config.transaction_cost
        
        logger.info("Backtest completed.")
        return (portfolio_returns - transaction_costs).dropna()

class PerformanceAnalyzer:
    """Analyzes performance and creates reports."""
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, config: TradingConfig):
        self.returns = portfolio_returns
        self.benchmark = benchmark_returns.loc[portfolio_returns.index]
        self.config = config
        self.metrics = {}

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculates all key performance metrics."""
        self.metrics['annual_return'] = self.returns.mean() * 252
        self.metrics['annual_volatility'] = self.returns.std() * np.sqrt(252)
        self.metrics['sharpe_ratio'] = (self.metrics['annual_return'] - self.config.risk_free_rate) / self.metrics['annual_volatility']
        
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        self.metrics['max_drawdown'] = drawdown.min()
        
        X = sm.add_constant(self.benchmark)
        model = sm.OLS(self.returns, X).fit()
        self.metrics['beta'] = model.params[1]
        self.metrics['alpha'] = model.params[0] * 252
        return self.metrics

    def create_performance_report(self) -> str:
        """Creates a formatted text summary of the performance."""
        report = "📈 **Quantitative Strategy Performance Report** 📉\n\n"
        report += f"*{self.returns.index.min().strftime('%Y-%m-%d')} to {self.returns.index.max().strftime('%Y-%m-%d')}*\n\n"
        report += "--- Key Metrics ---\n"
        report += f"Annual Return:       {self.metrics['annual_return']:.2%}\n"
        report += f"Annual Volatility:   {self.metrics['annual_volatility']:.2%}\n"
        report += f"Sharpe Ratio:        {self.metrics['sharpe_ratio']:.2f}\n"
        report += f"Max Drawdown:        {self.metrics['max_drawdown']:.2%}\n"
        report += f"Beta vs {self.config.benchmark}:           {self.metrics['beta']:.2f}\n"
        report += f"Annual Alpha:        {self.metrics['alpha']:.2%}\n"
        return report

    def create_performance_plot(self) -> BytesIO:
        """Creates the performance plot and returns it as an in-memory file."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Cumulative Returns
        (1 + self.returns).cumprod().plot(ax=axes[0], label='Portfolio', lw=2)
        (1 + self.benchmark).cumprod().plot(ax=axes[0], label=f'Benchmark ({self.config.benchmark})', linestyle='--', lw=2)
        axes[0].set_title('Performance: Cumulative Returns', fontsize=16)
        axes[0].set_ylabel('Growth of $1')
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        drawdown.plot(ax=axes[1], kind='area', color='red', alpha=0.3)
        axes[1].set_title('Portfolio Drawdown', fontsize=16)
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot to an in-memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig) # Close the figure to free memory
        return buf

def send_telegram_message(token: str, chat_id: str, message: str, plot_buffer: BytesIO = None):
    """Sends a message and an optional plot to a Telegram chat."""
    logger.info("Sending report to Telegram...")
    try:
        if plot_buffer:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            files = {'photo': ('performance_summary.png', plot_buffer, 'image/png')}
            data = {'chat_id': chat_id, 'caption': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, files=files, data=data, timeout=20)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data, timeout=10)
        
        response.raise_for_status()
        logger.info("Telegram message sent successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram message: {e}")
        # Log the response content for debugging if available
        if e.response is not None:
            logger.error(f"Telegram API Response: {e.response.text}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run quantitative trading backtest and send report.")
    parser.add_argument('--telegram-token', required=True, help="Telegram Bot API Token")
    parser.add_argument('--telegram-chat-id', required=True, help="Telegram Chat ID")
    args = parser.parse_args()

    try:
        # 1. Configuration
        config = TradingConfig()

        # 2. Core Logic
        data_manager = DataManager(config)
        market_data = data_manager.fetch_data()

        alpha_model = EnhancedAlphaEnsemble(config)
        risk_manager = RiskManager(config)

        backtester = Backtester(config, data_manager, alpha_model, risk_manager)
        portfolio_returns = backtester.run_backtest()

        # 3. Performance Analysis & Reporting
        benchmark_returns = market_data['returns'][config.benchmark]
        analyzer = PerformanceAnalyzer(portfolio_returns, benchmark_returns, config)
        
        analyzer.calculate_all_metrics()
        report_message = analyzer.create_performance_report()
        plot_image = analyzer.create_performance_plot()

        # 4. Send Notification
        send_telegram_message(args.telegram_token, args.telegram_chat_id, report_message, plot_image)

    except Exception as e:
        error_message = f"❌ **CRITICAL ERROR** ❌\n\nAn error occurred during the script execution:\n\n`{e}`"
        logger.critical(error_message, exc_info=True)
        send_telegram_message(args.telegram_token, args.telegram_chat_id, error_message)

if __name__ == '__main__':
    main()
