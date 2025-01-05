# Stock Bot - Discord Trading Simulator

A Discord bot that simulates stock market trading with virtual currency. Users can buy/sell stocks, track their portfolio, and compete on the leaderboard.

## Features

- **Trading**: Buy and sell stocks with virtual currency
- **Portfolio Management**: Track your holdings and net worth
- **Market Overview**: View stocks by industry with current prices
- **Price Charts**: Visual price history for each stock
- **News & Events**: Events that affect stock prices
- **Leaderboard**: Compete with other users
- **Price Alerts**: Get notified when stocks hit target prices

## Commands

### Basic Commands
- `/balance` - Check your cash balance
- `/portfolio` - View your stock holdings
- `/market` - Browse available stocks by industry
- `/leaderboard` - See top traders by net worth
- `/stock <ticker>` - View detailed stock info
- `/stock_chart <ticker> <timeframe>` - View price history chart

### Trading Commands
- `/buy <ticker> <amount>` - Buy shares
- `/sell <ticker> <amount>` - Sell shares
- `/alert <ticker> <price>` - Set price alert

### Admin Commands (Owner Only)
- `/create_stock` - Add new stock
- `/update_stock` - Modify stock details
- `/publish_event` - Create market-moving events
- `/plateau` - Freeze stock price
- `/set_target_price` - Gradually move stock to target price

## Deployment Guide

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