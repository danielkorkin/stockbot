# Stock Bot - Real-Time Stock Trading Simulator

A Discord bot that simulates stock trading using real-time market data from Yahoo Finance. Users can buy/sell stocks using virtual currency and track their portfolio performance.

## Features

- **Real-Time Data**: Live stock prices including pre-market and after-hours trading
- **Portfolio Management**: Track your holdings and net worth
- **Market Overview**: View top 50 stocks by market cap
- **Price Charts**: Interactive price charts with multiple timeframes
- **Transaction History**: Track all your trades
- **Leaderboard**: Compete with other users

## Commands

### Trading Commands
- `/balance` - Check your cash balance and net worth
- `/buy <ticker> <shares>` - Buy shares of a stock
- `/sell <ticker> <shares>` - Sell shares of a stock
- `/portfolio` - View your current stock holdings

### Market Information
- `/stock <ticker>` - View detailed stock info including pre/post market prices
- `/chart <ticker> <period>` - View price chart with customizable timeframes
  - Periods: 1min, 5min, 1h, 12h, 1d, 5d, 1mo, 1y, ytd, 5y, max
- `/top` - Browse top 50 stocks by market cap (paginated)

### User Stats
- `/history` - View your transaction history
- `/leaderboard` - See top traders by net worth

## Setup Guide

### Prerequisites
- Docker installed
- MongoDB database (Cloud Atlas or self-hosted)
- Discord Bot Token

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