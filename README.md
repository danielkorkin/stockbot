# Stock Bot - Real-Time Stock & Crypto Trading Simulator

A Discord bot that simulates trading of stocks and cryptocurrencies using real-time market data. Users can buy/sell stocks and crypto using virtual currency and track their portfolio performance.

## Features

-   **Real-Time Data**: Live stock prices including pre-market and after-hours trading
-   **Crypto Trading**: Buy and sell major cryptocurrencies with real-time prices
-   **Portfolio Management**: Track your stock and crypto holdings and net worth
-   **Market Overview**: View top 50 stocks by market cap
-   **Price Charts**: Interactive price charts with multiple timeframes and styles
-   **Transaction History**: Track all your trades across stocks and crypto
-   **Leaderboard**: Compete with other users
-   **Automatic Orders**: Set up automatic buy/sell orders at target prices

## Commands

### Account & Portfolio

-   `/balance [user]` - Check your cash balance, stock portfolio, crypto holdings, and total net worth
-   `/history [user]` - View your transaction history for both stocks and crypto
-   `/leaderboard` - See top traders by net worth

### Stock Trading

-   `/stock buy <ticker> <shares>` - Buy shares of a stock
-   `/stock sell <ticker> <shares>` - Sell shares of a stock
-   `/stock buyat <ticker> <shares> <price>` - Set up automatic buy order
-   `/stock sellat <ticker> <shares> <price>` - Set up automatic sell order
-   `/stock pending` - View your pending buy/sell orders

### Crypto Trading

-   `/crypto buy <ticker> <amount>` - Buy cryptocurrency (e.g., BTC, ETH)
-   `/crypto sell <ticker> <amount>` - Sell cryptocurrency
-   `/crypto lookup <ticker>` - Get detailed crypto info
-   `/crypto chart <ticker> <period>` - View crypto price charts

### Market Information

-   `/stock info <ticker>` - View detailed stock info including pre/post market prices
-   `/stock chart <ticker> <period> <type>` - View customizable stock price charts
    -   Periods: 1min, 5min, 1h, 12h, 1d, 5d, 1mo, 1y, ytd, 5y, max
    -   Chart Types: line, candle, mountain, baseline, etc.
-   `/stock top` - Browse top 50 stocks by market cap
-   `/stock today` - Get market overview with indices and top movers
-   `/stock news <ticker>` - Get latest news for a stock

### Price Alerts

-   `/stock alert <ticker> <price>` - Set price alert (use negative price for below target)
-   `/stock alerts [ticker] [clear]` - View or clear your price alerts

### Trading Strategy

-   `/stock strategy <ticker> <period> <amount>` - Test trading strategies with backtesting

## Setup Guide

### Prerequisites

-   Docker installed
-   MongoDB database (Cloud Atlas or self-hosted)
-   Discord Bot Token

### Environment Setup

1. Copy `.env.example` to `.env`:

2. Fill in your environment variables in `.env`:

```env
OWNER_ID=your_discord_id
TEST_GUILD_ID=your_test_server_id
MONGO_URI=your_mongodb_connection_string
BOT_TOKEN=your_discord_bot_token
MONGO_DB_NAME=StockBotDB
ENVIRONMENT=development  # or production
```

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t stockbot .
```

2. Run the container:

```bash
docker run -d --name stockbot stockbot
```

### Manual Deployment

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the bot:

```bash
python main.py
```
