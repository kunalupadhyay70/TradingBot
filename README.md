# 🤖 TradingBot - Advanced Cryptocurrency Trading Bot

**A production-grade trading bot with machine learning, technical analysis, and professional risk management.**

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Indicators](#indicators)
- [Performance](#performance)
- [Risk Management](#risk-management)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### 🤖 **Machine Learning**
- **LightGBM** binary classifier for price direction prediction
- Feature scaling and normalization
- Time series cross-validation (no look-ahead bias)
- Early stopping to prevent overfitting
- Feature importance analysis

### 📊 **27+ Technical Indicators**
- **Trend**: EMA (9/21/50/200), SMA-50
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Volatility**: ATR, Bollinger Bands, Historical Volatility
- **Volume**: OBV, Money Flow Index (MFI), Volume Ratio
- **Price Action**: VWAP, Returns (multi-period)

### 📈 **Advanced Backtesting**
- Realistic commission (0.1%) and slippage (0.05%)
- Position sizing with Kelly Criterion variant
- Dynamic stop loss and take profit
- Comprehensive performance metrics:
  - Sharpe Ratio, Max Drawdown, Win Rate
  - Profit Factor, Precision, Recall, F1

### 🛡️ **Professional Risk Management**
- Per-trade risk limits (1% of capital)
- Maximum drawdown protection
- Position sizing optimization
- Overtrading guards
- Real-time P&L tracking

### 📡 **Real-Time Trading**
- Binance WebSocket integration
- Order book analysis
- Live feature computation
- Graceful error handling

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Binance account with API keys (optional, for live trading)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/kunalupadhyay70/TradingBot.git
cd TradingBot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import lightgbm, pandas, numpy; print('✅ All packages installed')"
