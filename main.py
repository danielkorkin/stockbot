import ast
import asyncio
import datetime
import datetime as dt
import io
import math
import os

# Replace pynescript import with re
import re

# Add after existing imports
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import aiohttp  # Add this to the imports at the top
import discord
import matplotlib.pyplot as plt
import motor.motor_asyncio
import pandas as pd
import pandas_ta as ta  # Add to requirements.txt
import pytz
import yfinance as yf

# Add these imports at the top
from discord import TextStyle, app_commands, ui
from discord.ext import commands
from dotenv import load_dotenv

# Add to imports section
from pycoingecko import CoinGeckoAPI

load_dotenv()

###########################
# Configuration Constants #
###########################

OWNER_ID = int(os.getenv("OWNER_ID", "123456789012345678"))
TEST_GUILD_ID = int(os.getenv("TEST_GUILD_ID", "123456789012345678"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "StockBotDB")
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 10000.0))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Add new constants after existing configuration constants
CRYPTO_MIN_TRADE = Decimal("0.01")
COINGECKO = CoinGeckoAPI()

# Add after configuration constants
MAJOR_CRYPTO_SYMBOLS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "USDC": "usd-coin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "TRX": "tron",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "TON": "the-open-network",
    "DAI": "dai",
    "SHIB": "shiba-inu",
    "UNI": "uniswap",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
}

# Add after existing constants
PENDING_ORDERS_COLLECTION = "pending_orders"


# Add new helper functions before the PaginatorView class
# Replace get_coin_id function
@lru_cache(maxsize=100)
def get_coin_id(symbol: str) -> Optional[str]:
    """Get CoinGecko coin ID from symbol (only for major cryptocurrencies)"""
    upper_symbol = symbol.upper()
    if (upper_symbol := symbol.upper()) in MAJOR_CRYPTO_SYMBOLS:
        return MAJOR_CRYPTO_SYMBOLS[upper_symbol]
    return None


async def get_crypto_price(ticker: str) -> Optional[float]:
    """Get current price for a cryptocurrency using CoinGecko"""
    try:
        coin_id = get_coin_id(ticker)
        if not coin_id:
            print(f"Could not find coin ID for {ticker}")
            return None

        # Add debug logging
        print(f"Getting price for coin_id: {coin_id}")

        price_data = COINGECKO.get_price(
            ids=coin_id,
            vs_currencies="usd",
            include_market_cap=True,
            include_24hr_vol=True,
            include_24hr_change=True,
        )

        print(f"Raw price data: {price_data}")  # Debug log

        if coin_id in price_data and "usd" in price_data[coin_id]:
            price = float(price_data[coin_id]["usd"])
            print(f"Parsed price for {ticker}: ${price}")  # Debug log
            return price

        print(f"No price data found for {ticker}")  # Debug log
        return None
    except Exception as e:
        print(f"Error getting crypto price for {ticker}: {e}")
        return None


async def get_crypto_info(ticker: str) -> Optional[Dict]:
    """Get detailed information about a cryptocurrency using CoinGecko"""
    try:
        coin_id = get_coin_id(ticker)
        if not coin_id:
            print(f"Could not find coin ID for {ticker}")
            return None

        # Add debug logging
        print(f"Getting info for coin_id: {coin_id}")

        data = COINGECKO.get_coin_by_id(
            coin_id,
            localization=False,
            tickers=False,
            market_data=True,
            community_data=False,
            developer_data=False,
            sparkline=False,
        )

        market_data = data.get("market_data", {})
        current_price = market_data.get("current_price", {}).get("usd")

        if not current_price:
            print(f"No price found in market data for {ticker}")
            return None

        price = float(current_price)
        if price <= 0:
            print(f"Invalid price ({price}) for {ticker}")
            return None

        return {
            "name": data.get("name", ticker.upper()),
            "symbol": data.get("symbol", "").upper(),
            "price": price,
            "market_cap": float(market_data.get("market_cap", {}).get("usd", 0)),
            "volume_24h": float(market_data.get("total_volume", {}).get("usd", 0)),
            "circulating_supply": float(market_data.get("circulating_supply", 0)),
            "total_supply": float(market_data.get("total_supply", 0) or 0),
            "price_change_24h": float(
                market_data.get("price_change_percentage_24h", 0)
            ),
            "ath": float(market_data.get("ath", {}).get("usd", 0)),
            "ath_date": market_data.get("ath_date", {}).get("usd", ""),
        }
    except Exception as e:
        print(f"Error getting crypto info for {ticker}: {e}")
        return None


async def get_stock_history(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)

        # Update period and interval mappings for better short-term data
        period_map = {
            "1mo": "1mo",
            "3mo": "3mo",
            "6mo": "6mo",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y",
            "max": "max",
        }

        interval_map = {
            "1mo": "1d",  # Daily data for 1 month
            "3mo": "1d",  # Daily data for 3 months
            "6mo": "1d",  # Daily data for 6 months
            "1y": "1d",  # Daily data for 1 year
            "2y": "1d",  # Daily data for 2 years
            "5y": "1wk",  # Weekly data for 5 years
            "max": "1mo",  # Monthly data for max period
        }

        yf_period = period_map.get(period, "1mo")
        yf_interval = interval_map.get(period, "1d")

        print(
            f"Fetching data for {ticker} with period: {yf_period}, interval: {yf_interval}"
        )

        # Get historical data
        history = stock.history(period=yf_period, interval=yf_interval, prepost=True)

        if history.empty:
            print(f"No data returned for {ticker}")
            return None

        # Print data info for debugging
        print(f"Fetched {len(history)} data points")
        print(f"Date range: {history.index[0]} to {history.index[-1]}")

        return history

    except Exception as e:
        print(f"Error getting history for {ticker}: {e}")
        return None


# Update the get_stock_news function
async def get_stock_news(ticker: str) -> List[Dict]:
    """Get the latest news for a stock from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news  # Access news directly as a property
        if not news:
            return []

        # Format the news items with proper field validation and timestamps
        formatted_news = []
        for item in news[:5]:  # Get latest 5 news items
            # Validate required fields with defaults
            title = item.get("title", "").strip()
            if not title:
                continue  # Skip items without titles

            # Get summary and publisher with defaults
            summary = item.get("summary", "").strip() or "No summary available"
            publisher = item.get("publisher", "Yahoo Finance").strip()

            # Convert timestamp to datetime with validation
            timestamp = item.get("providerPublishTime", 0)
            if timestamp:
                published = datetime.datetime.fromtimestamp(timestamp)
            else:
                published = datetime.datetime.utcnow()  # Fallback to current time

            # Add only if we have valid title
            formatted_news.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "link": item.get("link", ""),
                    "published": published,
                    "summary": summary,
                }
            )

        return formatted_news
    except Exception as e:
        print(f"Error getting news for {ticker}: {e}")
        return []


async def add_pending_order(
    bot, user_id: int, order_type: str, ticker: str, quantity: float, price: float
) -> bool:
    """Add a pending buy/sell order"""
    try:
        # Verify user has sufficient funds/stocks before adding order
        user_data = await get_user_data(bot.user_collection, user_id)

        if order_type == "buy":
            total_cost = price * quantity
            if user_data["balance"] < total_cost:
                return False
        elif order_type == "sell":
            if (
                ticker.upper() not in user_data["portfolio"]
                or user_data["portfolio"][ticker.upper()] < quantity
            ):
                return False

        order = {
            "user_id": user_id,
            "type": order_type,
            "ticker": ticker.upper(),
            "quantity": quantity,
            "price": price,
            "created_at": datetime.datetime.utcnow(),
        }

        await bot.db[PENDING_ORDERS_COLLECTION].insert_one(order)
        return True
    except Exception as e:
        print(f"Error adding pending order: {e}")
        return False


async def remove_pending_order(bot, order_id: str) -> bool:
    """Remove a pending order by its ID"""
    try:
        result = await bot.db[PENDING_ORDERS_COLLECTION].delete_one({"_id": order_id})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error removing pending order: {e}")
        return False


async def get_user_pending_orders(bot, user_id: int) -> List[Dict]:
    """Get all pending orders for a user"""
    try:
        cursor = bot.db[PENDING_ORDERS_COLLECTION].find({"user_id": user_id})
        return await cursor.to_list(length=None)
    except Exception as e:
        print(f"Error getting pending orders: {e}")
        return []


class PaginatorView(discord.ui.View):
    def __init__(
        self,
        items: list,
        items_per_page: int,
        embed_factory,
        author_id: int,
        max_pages: Optional[int] = None,
    ):
        super().__init__(timeout=None)
        self.items = items
        self.items_per_page = items_per_page
        self.embed_factory = embed_factory
        self.author_id = author_id
        self.current_page = 0
        total_full_pages = math.ceil(len(items) / items_per_page)
        self.total_pages = (
            min(total_full_pages, max_pages) if max_pages else total_full_pages
        )
        self.update_button_states()

    def update_button_states(self):
        # Disable Prev button on first page
        self.previous_button.disabled = self.current_page == 0
        # Disable Next button on last page
        self.next_button.disabled = self.current_page >= self.total_pages - 1

    async def send_first_page(
        self, interaction: discord.Interaction, is_followup: bool = False
    ):
        embed = self.build_embed()
        self.update_button_states()
        if is_followup:
            await interaction.send(embed=embed, view=self)
        else:
            await interaction.response.send_message(embed=embed, view=self)

    def build_embed(self) -> discord.Embed:
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        subset = self.items[start_idx:end_idx]
        return self.embed_factory(self.current_page, self.total_pages, subset)

    async def update_message(self, interaction: discord.Interaction):
        embed = self.build_embed()
        self.update_button_states()
        await interaction.response.edit_message(embed=embed, view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "You cannot interact with this paginator.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Prev", style=discord.ButtonStyle.blurple)
    async def previous_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if self.current_page > 0:
            self.current_page -= 1
            await self.update_message(interaction)
        else:
            await interaction.response.defer()

    @discord.ui.button(label="Next", style=discord.ButtonStyle.blurple)
    async def next_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            await self.update_message(interaction)
        else:
            await interaction.response.defer()

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.red)
    async def stop_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        for child in self.children:
            child.disabled = True
        await self.update_message(interaction)
        self.stop()


class StockBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = False
        super().__init__(command_prefix="!", intents=intents)
        self.db = None
        self.user_collection = None
        self.alert_manager = AlertManager()

    async def setup_hook(self):
        try:
            # Add commands to the command tree first
            self.tree.add_command(stock_group)
            self.tree.add_command(crypto_group)

            if ENVIRONMENT.lower() == "development":
                # Sync to test guild
                print("Development environment detected, syncing to test guild...")
                self.tree.copy_global_to(guild=discord.Object(id=TEST_GUILD_ID))
                await self.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
                # Clear global commands in development
                await self.tree.sync()
                print("Commands synced to test guild and cleared globally")
            else:
                # Production: Clear test guild commands and sync globally
                print("Production environment detected, syncing globally...")
                await self.tree.sync()
                # Clear test guild specific commands
                self.tree.clear_commands(guild=discord.Object(id=TEST_GUILD_ID))
                await self.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
                print("Commands synced globally and cleared from test guild")

            print(
                f"Registered commands: {[cmd.name for cmd in self.tree.get_commands()]}"
            )

        except Exception as e:
            print(f"Error syncing commands: {e}")
            raise  # Re-raise the exception for better error tracking

        # Setup MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = client[MONGO_DB_NAME]
        self.user_collection = self.db["users"]

        self.alert_manager.start_monitoring(self)

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching, name="the Stock Market"
            )
        )


class AlertManager:
    def __init__(self):
        self.alerts: Dict[str, Set[Tuple[int, discord.TextChannel, float]]] = {}
        self.is_running = False
        self.bot = None  # Add this line to store bot reference

    def start_monitoring(self, bot):
        """Initialize monitoring with bot reference"""
        if not self.is_running:
            self.is_running = True
            self.bot = bot  # Store bot reference
            self.bot.loop.create_task(self.check_alerts())

    def add_alert(
        self,
        ticker: str,
        user_id: int,
        channel: discord.TextChannel,
        target_price: float,
    ):
        if ticker not in self.alerts:
            self.alerts[ticker] = set()
        self.alerts[ticker].add((user_id, channel, target_price))

    def remove_alerts(
        self, ticker: str, user_id: int
    ) -> List[Tuple[discord.TextChannel, float]]:
        if ticker not in self.alerts:
            return []

        # Find all alerts for this user/ticker combination
        user_alerts = [
            (ch, price) for (uid, ch, price) in self.alerts[ticker] if uid == user_id
        ]
        # Remove them from the set
        self.alerts[ticker] = {
            alert for alert in self.alerts[ticker] if alert[0] != user_id
        }
        # Clean up empty tickers
        if not self.alerts[ticker]:
            del self.alerts[ticker]
        return user_alerts

    async def check_alerts(self):
        while True:
            try:
                if (
                    self.bot is None
                ):  # Change from 'if not self.bot' to explicit None check
                    print("Bot reference not set in AlertManager")
                    return

                # Check alerts
                for ticker in list(self.alerts.keys()):
                    if not self.alerts.get(ticker):
                        continue

                    price = await get_stock_price(ticker)
                    if price is None:
                        continue

                    triggered = set()
                    for user_id, channel, target_price in self.alerts[ticker]:
                        if (target_price >= 0 and price >= target_price) or (
                            target_price < 0 and price <= abs(target_price)
                        ):
                            triggered.add((user_id, channel, target_price))
                            await channel.send(
                                f"<@{user_id}> Alert triggered for {ticker}!\n"
                                f"Target price: ${abs(target_price):,.2f} {'(above)' if target_price >= 0 else '(below)'}\n"
                                f"Current price: ${price:,.2f}"
                            )

                    if triggered:
                        self.alerts[ticker] -= triggered
                        if not self.alerts[ticker]:
                            del self.alerts[ticker]

                # Check pending orders - Fix database object comparison
                if (
                    self.bot.db is not None
                ):  # Change from 'if self.bot.db' to explicit None check
                    async for order in self.bot.db[PENDING_ORDERS_COLLECTION].find():
                        price = await get_stock_price(order["ticker"])
                        if price is None:
                            continue

                        if order["type"] == "buy" and price <= order["price"]:
                            await self.execute_buy_order(order, price)
                        elif order["type"] == "sell" and price >= order["price"]:
                            await self.execute_sell_order(order, price)

            except Exception as e:
                print(f"Error checking alerts/orders: {e}")

            await asyncio.sleep(60)

    async def execute_buy_order(self, order: Dict, current_price: float):
        """Execute a pending buy order"""
        if not self.bot:  # Add safety check
            return

        try:
            user_data = await get_user_data(self.bot.user_collection, order["user_id"])
            total_cost = current_price * order["quantity"]

            if user_data["balance"] >= total_cost:
                # Execute the buy
                new_balance = user_data["balance"] - total_cost
                portfolio = user_data["portfolio"]
                portfolio[order["ticker"]] = (
                    portfolio.get(order["ticker"], 0) + order["quantity"]
                )

                transaction = {
                    "type": "buy",
                    "ticker": order["ticker"],
                    "shares": order["quantity"],
                    "price": current_price,
                    "total": total_cost,
                    "timestamp": datetime.datetime.utcnow(),
                }

                await self.bot.user_collection.update_one(
                    {"_id": order["user_id"]},
                    {
                        "$set": {"balance": new_balance, "portfolio": portfolio},
                        "$push": {"transactions": transaction},
                    },
                )

                # Remove the pending order
                await self.bot.db[PENDING_ORDERS_COLLECTION].delete_one(
                    {"_id": order["_id"]}
                )

                # Notify the user
                user = await self.bot.fetch_user(order["user_id"])
                if user:
                    embed = discord.Embed(
                        title="Buy Order Executed",
                        description=f"Bought {order['quantity']} shares of {order['ticker']} at ${current_price:,.2f}",
                        color=discord.Color.green(),
                    )
                    embed.add_field(name="Total Cost", value=f"${total_cost:,.2f}")
                    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}")
                    await user.send(embed=embed)

        except Exception as e:
            print(f"Error executing buy order: {e}")

    async def execute_sell_order(self, order: Dict, current_price: float):
        """Execute a pending sell order"""
        if not self.bot:  # Add safety check
            return

        try:
            user_data = await get_user_data(self.bot.user_collection, order["user_id"])
            portfolio = user_data["portfolio"]

            if (
                order["ticker"] in portfolio
                and portfolio[order["ticker"]] >= order["quantity"]
            ):
                # Execute the sell
                total_value = current_price * order["quantity"]
                new_balance = user_data["balance"] + total_value

                portfolio[order["ticker"]] -= order["quantity"]
                if portfolio[order["ticker"]] == 0:
                    del portfolio[order["ticker"]]

                transaction = {
                    "type": "sell",
                    "ticker": order["ticker"],
                    "shares": order["quantity"],
                    "price": current_price,
                    "total": total_value,
                    "timestamp": datetime.datetime.utcnow(),
                }

                await self.bot.user_collection.update_one(
                    {"_id": order["user_id"]},
                    {
                        "$set": {"balance": new_balance, "portfolio": portfolio},
                        "$push": {"transactions": transaction},
                    },
                )

                # Remove the pending order
                await self.bot.db[PENDING_ORDERS_COLLECTION].delete_one(
                    {"_id": order["_id"]}
                )

                # Notify the user
                user = await self.bot.fetch_user(order["user_id"])
                if user:
                    embed = discord.Embed(
                        title="Sell Order Executed",
                        description=f"Sold {order['quantity']} shares of {order['ticker']} at ${current_price:,.2f}",
                        color=discord.Color.green(),
                    )
                    embed.add_field(name="Total Value", value=f"${total_value:,.2f}")
                    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}")
                    await user.send(embed=embed)

        except Exception as e:
            print(f"Error executing sell order: {e}")


async def get_stock_price(ticker: str) -> Optional[float]:
    """Get the most recent price available, including after-hours trading."""
    try:
        info = await get_stock_info(ticker)
        if info:
            return info["price"]  # This already includes pre/post market prices
        return None
    except Exception as e:
        print(f"Error getting latest price for {ticker}: {e}")
        return None


async def get_stock_info(ticker: str) -> Optional[Dict]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get regular market price with better validation
        current_price = None
        market_status = "Regular Hours"

        # Try current price first
        if info.get("currentPrice") and float(info["currentPrice"]) > 0:
            current_price = float(info["currentPrice"])

        # Then try regular market price
        elif info.get("regularMarketPrice") and float(info["regularMarketPrice"]) > 0:
            current_price = float(info["regularMarketPrice"])

        # Finally try previous close
        elif info.get("previousClose") and float(info["previousClose"]) > 0:
            current_price = float(info["previousClose"])
            market_status = "Previous Close"

        if not current_price or current_price <= 0:
            print(f"No valid price found for {ticker}")
            return None

        # Handle market cap more carefully
        market_cap = None
        if info.get("marketCap"):
            market_cap = float(info["marketCap"])
        elif info.get("totalMarketCap"):
            market_cap = float(info["totalMarketCap"])

        # If no market cap, try to calculate it
        if not market_cap and info.get("sharesOutstanding"):
            shares = float(info["sharesOutstanding"])
            market_cap = current_price * shares

        # If still no market cap, use a reasonable default
        if not market_cap:
            print(f"Using price-based market cap estimation for {ticker}")
            market_cap = current_price * 1000000  # Assume 1M shares as fallback

        # Debug logging
        print(f"Stock info for {ticker}:")
        print(f"Price: ${current_price:.2f}")
        print(f"Market Cap: ${market_cap:,.2f}")
        print(f"Status: {market_status}")

        return {
            "name": info.get("longName") or info.get("shortName", "Unknown"),
            "price": current_price,
            "market_status": market_status,
            "market_cap": market_cap,
            "regular_price": float(info.get("regularMarketPrice", 0)) or current_price,
            "pre_market": float(info.get("preMarketPrice", 0)) or None,
            "post_market": float(info.get("postMarketPrice", 0)) or None,
            "pe_ratio": float(info.get("forwardPE", 0))
            or float(info.get("trailingPE", 0))
            or 0.0,
            "dividend_yield": float(info.get("dividendYield", 0) * 100)
            if info.get("dividendYield")
            else 0.0,
            "sector": info.get("sector", "Unknown"),
            "volume": int(info.get("volume", 0)) or int(info.get("averageVolume", 0)),
        }

    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        print(f"Raw info data: {info}")
        return None


# Modify get_user_data function to include crypto
async def get_user_data(user_collection, user_id: int):
    user_data = await user_collection.find_one({"_id": user_id})
    if not user_data:
        user_data = {
            "_id": user_id,
            "balance": INITIAL_BALANCE,
            "portfolio": {},
            "crypto": {},  # Add crypto portfolio
            "transactions": [],
        }
        await user_collection.insert_one(user_data)
    elif "crypto" not in user_data:  # Add crypto field if missing
        user_data["crypto"] = {}
        await user_collection.update_one({"_id": user_id}, {"$set": {"crypto": {}}})
    return user_data


# Modify calculate_portfolio_value to include crypto
async def calculate_portfolio_value(
    portfolio: Dict[str, int], crypto: Dict[str, float] = None
) -> tuple[float, dict, float]:
    """Calculate total portfolio value including stocks and crypto"""
    total = 0.0
    price_info = {}
    crypto_value = 0.0

    # Calculate stock portfolio value
    for ticker, shares in portfolio.items():
        info = await get_stock_info(ticker)
        if info and info["price"] > 0:
            value = info["price"] * shares
            total += value
            price_info[ticker] = {
                "price": info["price"],
                "market_status": info["market_status"],
                "total_value": value,
            }

    # Calculate crypto portfolio value
    if crypto:
        for ticker, amount in crypto.items():
            price = await get_crypto_price(ticker)
            if price:
                value = price * amount
                crypto_value += value
                price_info[f"{ticker}-USD"] = {
                    "price": price,
                    "market_status": "24/7",
                    "total_value": value,
                }

    return round(total, 2), price_info, round(crypto_value, 2)


async def get_top_stocks() -> List[Dict]:  # Change this line
    try:
        # Get top 50 stocks using yfinance and pandas
        dow_tickers = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NVDA",
            "BRK-B",
            "JPM",
            "V",
            "PG",
            "JNJ",
            "XOM",
            "MA",
            "UNH",
            "HD",
            "CVX",
            "MRK",
            "LLY",
            "KO",
            "PEP",
            "ABBV",
            "AVGO",
            "BAC",
            "PFE",
            "CSCO",
            "TMO",
            "MCD",
            "ABT",
            "CRM",
            "COST",
            "DIS",
            "WMT",
            "ACN",
            "DHR",
            "VZ",
            "NEE",
            "LIN",
            "TXN",
            "ADBE",
            "PM",
            "CMCSA",
            "NKE",
            "WFC",
            "BMY",
            "RTX",
            "UPS",
            "HON",
            "T",
            "ORCL",
            "QCOM",
        ]

        stocks = []
        for ticker in dow_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Get price with fallbacks
                price = (
                    info.get("regularMarketPrice")
                    or info.get("currentPrice")
                    or info.get("previousClose")
                )

                # Get market cap with fallback
                market_cap = info.get("marketCap") or info.get("totalMarketCap")

                if price and market_cap:
                    stocks.append(
                        {
                            "name": info.get("longName")
                            or info.get("shortName", "Unknown"),
                            "ticker": ticker,
                            "price": float(price),
                            "market_cap": float(market_cap),
                        }
                    )
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # Sort by market cap
        stocks.sort(key=lambda x: x["market_cap"], reverse=True)
        return stocks[:50]  # Return top 50
    except Exception as e:
        print(f"Error in get_top_stocks: {e}")
        return []


# Define the stock group and its commands
stock_group = app_commands.Group(name="stock", description="Stock related commands")


@stock_group.command(name="alert", description="Set a price alert for a stock")
@app_commands.describe(
    ticker="Stock ticker symbol",
    price="Target price (positive for above, negative for below)",
)
async def alert_command(interaction: discord.Interaction, ticker: str, price: float):
    """Set a price alert. Use positive price for above alerts, negative for below alerts."""
    ticker = ticker.upper()

    # Verify the ticker exists
    current_price = await get_stock_price(ticker)
    if current_price is None:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    # Add the alert
    bot.alert_manager.add_alert(ticker, interaction.user.id, interaction.channel, price)

    alert_type = "above" if price >= 0 else "below"
    embed = discord.Embed(
        title="Price Alert Set",
        description=f"You will be notified when {ticker} goes {alert_type} ${abs(price):,.2f}",
        color=discord.Color.green(),
    )
    embed.add_field(name="Current Price", value=f"${current_price:,.2f}", inline=True)
    embed.add_field(
        name="Target Price", value=f"${abs(price):,.2f} ({alert_type})", inline=True
    )

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="alerts", description="View or clear your price alerts")
@app_commands.describe(
    ticker="Stock ticker to clear alerts for (optional)",
    clear="Whether to clear the alerts for the specified ticker",
)
async def alerts_command(
    interaction: discord.Interaction, ticker: Optional[str] = None, clear: bool = False
):
    """View or clear your price alerts."""
    if ticker and clear:
        ticker = ticker.upper()
        removed = bot.alert_manager.remove_alerts(ticker, interaction.user.id)
        if removed:
            await interaction.response.send_message(
                f"Cleared {len(removed)} alert(s) for {ticker}."
            )
        else:
            await interaction.response.send_message(
                f"No alerts found for {ticker}.", ephemeral=True
            )
        return

    # Show all alerts
    embed = discord.Embed(title="Your Price Alerts", color=discord.Color.blue())
    alerts_found = False

    for t, alerts in bot.alert_manager.alerts.items():
        user_alerts = [a for a in alerts if a[0] == interaction.user.id]
        if user_alerts:
            alerts_found = True
            alert_texts = []
            for _, _, price in user_alerts:
                alert_type = "above" if price >= 0 else "below"
                alert_texts.append(f"${abs(price):,.2f} ({alert_type})")
            embed.add_field(name=t, value="\n".join(alert_texts), inline=False)

    if not alerts_found:
        embed.description = "You have no active price alerts."

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="info", description="Get information about a stock")
@app_commands.describe(ticker="Ticker symbol")
async def stock_info_command(interaction: discord.Interaction, ticker: str):
    # This is the renamed original 'stock' command
    info = await get_stock_info(ticker.upper())
    if not info:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    embed = discord.Embed(
        title=f"{info['name']} ({ticker.upper()})",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.utcnow(),  # Add timestamp
    )

    # Always show the most recent price first
    embed.add_field(
        name="Current Price",
        value=f"${info['price']:,.2f} ({info['market_status']})",
        inline=False,
    )

    # Show other prices if available
    if info["market_status"] != "Regular Hours" and info["regular_price"]:
        embed.add_field(
            name="Regular Hours", value=f"${info['regular_price']:,.2f}", inline=True
        )

    embed.add_field(name="Market Cap", value=f"${info['market_cap']:,.2f}", inline=True)
    embed.add_field(name="P/E Ratio", value=f"{info['pe_ratio']:.2f}", inline=True)
    embed.add_field(
        name="Dividend Yield", value=f"{info['dividend_yield']:.2f}%", inline=True
    )
    embed.add_field(name="Sector", value=info["sector"], inline=True)
    embed.add_field(name="Volume", value=f"{info['volume']:,}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="buy", description="Buy shares of a stock")
@app_commands.describe(ticker="Ticker symbol", shares="Number of shares to buy")
async def stock_buy_command(interaction: discord.Interaction, ticker: str, shares: int):
    # Move existing buy command code here
    if shares <= 0:
        await interaction.response.send_message(
            "Please enter a positive number of shares.", ephemeral=True
        )
        return

    price = await get_stock_price(ticker.upper())
    if not price:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    total_cost = price * shares
    user_data = await get_user_data(bot.user_collection, interaction.user.id)

    if user_data["balance"] < total_cost:
        await interaction.response.send_message("Insufficient funds.", ephemeral=True)
        return

    new_balance = user_data["balance"] - total_cost
    portfolio = user_data["portfolio"]
    portfolio[ticker.upper()] = portfolio.get(ticker.upper(), 0) + shares

    transaction = {
        "type": "buy",
        "ticker": ticker.upper(),
        "shares": shares,
        "price": price,
        "total": total_cost,
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": transaction},
        },
    )

    embed = discord.Embed(title="Purchase Successful", color=discord.Color.green())
    embed.add_field(name="Stock", value=ticker.upper(), inline=True)
    embed.add_field(name="Shares", value=str(shares), inline=True)
    embed.add_field(name="Price/Share", value=f"${price:,.2f}", inline=True)
    embed.add_field(name="Total Cost", value=f"${total_cost:,.2f}", inline=True)
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="sell", description="Sell shares of a stock")
@app_commands.describe(ticker="Ticker symbol", shares="Number of shares to sell")
async def stock_sell_command(
    interaction: discord.Interaction, ticker: str, shares: int
):
    # Move existing sell command code here
    if shares <= 0:
        await interaction.response.send_message(
            "Please enter a positive number of shares.", ephemeral=True
        )
        return

    price = await get_stock_price(ticker.upper())
    if not price:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    portfolio = user_data["portfolio"]

    if ticker.upper() not in portfolio or portfolio[ticker.upper()] < shares:
        await interaction.response.send_message("Insufficient shares.", ephemeral=True)
        return

    total_value = price * shares
    new_balance = user_data["balance"] + total_value

    portfolio[ticker.upper()] -= shares
    if portfolio[ticker.upper()] == 0:
        del portfolio[ticker.upper()]

    transaction = {
        "type": "sell",
        "ticker": ticker.upper(),
        "shares": shares,
        "price": price,
        "total": total_value,
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": transaction},
        },
    )

    embed = discord.Embed(title="Sale Successful", color=discord.Color.green())
    embed.add_field(name="Stock", value=ticker.upper(), inline=True)
    embed.add_field(name="Shares", value=str(shares), inline=True)
    embed.add_field(name="Price/Share", value=f"${price:,.2f}", inline=True)
    embed.add_field(name="Total Value", value=f"${total_value:,.2f}", inline=True)
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="chart", description="Show price chart for a stock")
@app_commands.describe(
    ticker="Ticker symbol",
    period="Time period (1min,5min,1h,12h,1d,5d,1mo,1y,ytd,5y,max)",
    chart_type="Chart visualization type",
)
@app_commands.choices(
    period=[
        app_commands.Choice(name=p, value=p)
        for p in [
            "1min",
            "5min",
            "1h",
            "12h",
            "1d",
            "5d",
            "1mo",
            "1y",
            "ytd",
            "5y",
            "max",
        ]
    ],
    chart_type=[
        app_commands.Choice(name=t, value=t)
        for t in [
            "line",
            "step",
            "mountain",
            "baseline",
            "candle",
            "bar",
            "hlc",
            "wave",
            "scatter",
            "histogram",
            "range",
        ]
    ],
)
async def stock_chart_command(
    interaction: discord.Interaction, ticker: str, period: str, chart_type: str = "line"
):
    # Move existing chart command code here
    await interaction.response.defer()

    history = await get_stock_history(ticker.upper(), period)
    if history is None:
        await interaction.followup.send("Unable to fetch stock data.", ephemeral=True)
        return

    # Create the chart
    plt.figure(figsize=(10, 6))

    # Plot based on chart type
    if chart_type == "line":
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "step":
        plt.step(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "mountain":
        plt.fill_between(history.index, history["Close"], alpha=0.3, color="blue")
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "baseline":
        baseline = history["Close"].mean()
        plt.fill_between(
            history.index,
            history["Close"],
            baseline,
            where=(history["Close"] >= baseline),
            color="green",
            alpha=0.3,
        )
        plt.fill_between(
            history.index,
            history["Close"],
            baseline,
            where=(history["Close"] < baseline),
            color="red",
            alpha=0.3,
        )
        plt.axhline(y=baseline, color="gray", linestyle="--")

    elif chart_type == "candle":
        import matplotlib.dates as mdates
        from mplfinance.original_flavor import candlestick_ohlc

        # Convert date to numerical format for candlestick
        history["Date"] = mdates.date2num(history.index.to_pydatetime())
        ohlc = history[["Date", "Open", "High", "Low", "Close"]].values
        candlestick_ohlc(plt.gca(), ohlc, width=0.6, colorup="green", colordown="red")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    elif chart_type == "bar":
        # Plot OHLC bars
        plt.vlines(history.index, history["Low"], history["High"], color="black")
        plt.hlines(
            history["Open"],
            history.index,
            [i - 0.2 for i in range(len(history.index))],
            color="black",
        )
        plt.hlines(
            history["Close"],
            history.index,
            [i + 0.2 for i in range(len(history.index))],
            color="black",
        )

    elif chart_type == "hlc":
        plt.vlines(history.index, history["Low"], history["High"], color="black")
        plt.hlines(
            history["Close"],
            history.index,
            [i + 0.2 for i in range(len(history.index))],
            color="blue",
        )

    elif chart_type == "wave":
        plt.fill_between(
            history.index, history["High"], history["Low"], alpha=0.3, color="blue"
        )
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "scatter":
        plt.scatter(
            history.index,
            history["Close"],
            c=history["Close"],
            cmap="viridis",
            alpha=0.6,
        )

    elif chart_type == "histogram":
        plt.hist(history["Close"], bins=50, alpha=0.6, color="blue")
        plt.gca().set_ylabel("Frequency")

    elif chart_type == "range":
        # Plot price range channel
        upper = history["High"].rolling(window=20).max()
        lower = history["Low"].rolling(window=20).min()
        plt.fill_between(history.index, upper, lower, alpha=0.3, color="gray")
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    plt.title(f"{ticker.upper()} {chart_type.title()} Chart ({period})")
    plt.xlabel("Date/Time")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if chart_type != "histogram":
        current_price = history["Close"].iloc[-1]
        plt.scatter(history.index[-1], current_price, color="green", s=100, zorder=5)

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close()

    # Get current stock info and create embed
    stock_info = await get_stock_info(ticker.upper())
    price_change = history["Close"].iloc[-1] - history["Close"].iloc[0]
    price_change_pct = (price_change / history["Close"].iloc[0]) * 100

    embed = discord.Embed(
        title=f"{ticker.upper()} {chart_type.title()} Chart", color=discord.Color.blue()
    )

    # Add fields to embed
    current_price = history["Close"].iloc[-1]
    price_text = f"${current_price:,.2f} ({stock_info['market_status']})"
    embed.add_field(name="Current Price", value=price_text, inline=True)

    if stock_info["pre_market"]:
        embed.add_field(
            name="Pre-Market", value=f"${stock_info['pre_market']:,.2f}", inline=True
        )
    if stock_info["post_market"]:
        embed.add_field(
            name="After-Hours", value=f"${stock_info['post_market']:,.2f}", inline=True
        )

    embed.add_field(
        name="Change",
        value=f"${price_change:,.2f} ({price_change_pct:,.2f}%)",
        inline=True,
    )
    embed.add_field(name="Period", value=period, inline=True)
    embed.add_field(name="Chart Type", value=chart_type.title(), inline=True)

    # Send the chart
    file = discord.File(buf, filename=f"{ticker}_chart.png")
    embed.set_image(url=f"attachment://{ticker}_chart.png")
    await interaction.followup.send(file=file, embed=embed)


# Update the news command's embed factory
@stock_group.command(name="news", description="Show latest news for a stock")
@app_commands.describe(ticker="Ticker symbol")
async def stock_news_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()

    news_items = await get_stock_news(ticker.upper())
    if not news_items:
        await interaction.followup.send("No news found for this stock.", ephemeral=True)
        return

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        news_item = subset[0]  # Since we're showing 1 item per page
        embed = discord.Embed(
            title=f"{ticker.upper()} News ({page_idx + 1}/{total_pages})",
            description=news_item["title"],
            color=discord.Color.blue(),
            url=news_item["link"],
            timestamp=news_item["published"],  # Add timestamp to embed
        )

        if news_item["summary"]:
            # Truncate summary if too long and add ellipsis
            summary = news_item["summary"][:1000] + (
                "..." if len(news_item["summary"]) > 1000 else ""
            )
            embed.add_field(
                name="Summary",
                value=summary,
                inline=False,
            )

        if news_item["publisher"]:
            embed.add_field(name="Source", value=news_item["publisher"], inline=True)

        return embed

    view = PaginatorView(news_items, 1, embed_factory, interaction.user.id)
    await view.send_first_page(interaction.followup, is_followup=True)


@stock_group.command(name="strategy", description="Test a Pine trading strategy")
@app_commands.describe(
    ticker="Stock ticker symbol",
    period="Time period for backtest",
    initial_amount="Initial investment amount",
    algorithm_url="Optional: URL to raw algorithm code (e.g., GitHub raw link)",
)
@app_commands.choices(
    period=[
        app_commands.Choice(name=p, value=p)
        for p in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    ]
)
async def stock_strategy_command(
    interaction: discord.Interaction,
    ticker: str,
    period: str,
    initial_amount: float,
    algorithm_url: str = None,
):
    # Move existing strategy command code here
    view = PineStrategyView(ticker.upper(), period, initial_amount)

    if algorithm_url:
        # Verify URL format
        if not algorithm_url.startswith(("http://", "https://")):
            await interaction.response.send_message(
                "Invalid URL format. Please provide a valid HTTP(S) URL.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        algorithm_code = await fetch_algorithm_code(algorithm_url)
        if algorithm_code is None:
            await interaction.followup.send(
                "Failed to fetch valid algorithm code. Please ensure the URL points to a raw Pine script.",
                ephemeral=True,
            )
            return

        view.pine_code = algorithm_code
        await view.execute_strategy(interaction)
    else:
        # Show the modal for manual code input
        modal = PineStrategyModal()
        modal.view = view
        await interaction.response.send_modal(modal)


@stock_group.command(
    name="top", description="Show top 50 stocks by market cap (10 pages, 5 per page)"
)
async def stock_top_command(interaction: discord.Interaction):
    # Move existing top command code here
    await interaction.response.defer()

    stocks = await get_top_stocks()
    if not stocks:
        await interaction.followup.send("Unable to fetch stock data.", ephemeral=True)
        return

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        embed = discord.Embed(
            title=f"Top Stocks by Market Cap (Page {page_idx + 1}/{total_pages})",
            color=discord.Color.blue(),
        )

        for i, stock in enumerate(subset, start=page_idx * 5 + 1):
            market_cap_b = stock["market_cap"] / 1_000_000_000  # Convert to billions
            embed.add_field(
                name=f"#{i}. {stock['ticker']}",
                value=f"**{stock['name']}**\nPrice: ${stock['price']:,.2f}\nMarket Cap: ${market_cap_b:.1f}B",
                inline=False,
            )

        return embed

    view = PaginatorView(stocks, 5, embed_factory, interaction.user.id, max_pages=10)
    await view.send_first_page(interaction.followup, is_followup=True)


@stock_group.command(name="lookup", description="Get information about a stock")
@app_commands.describe(ticker="Ticker symbol")
async def stock_lookup_command(interaction: discord.Interaction, ticker: str):
    info = await get_stock_info(ticker.upper())
    if not info:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    embed = discord.Embed(
        title=f"{info['name']} ({ticker.upper()})",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.utcnow(),
    )

    # Always show the most recent price first
    embed.add_field(
        name="Current Price",
        value=f"${info['price']:,.2f} ({info['market_status']})",
        inline=False,
    )

    # Show other prices if available
    if info["market_status"] != "Regular Hours" and info["regular_price"]:
        embed.add_field(
            name="Regular Hours", value=f"${info['regular_price']:,.2f}", inline=True
        )

    embed.add_field(name="Market Cap", value=f"${info['market_cap']:,.2f}", inline=True)
    embed.add_field(name="P/E Ratio", value=f"{info['pe_ratio']:.2f}", inline=True)
    embed.add_field(
        name="Dividend Yield", value=f"{info['dividend_yield']:.2f}%", inline=True
    )
    embed.add_field(name="Sector", value=info["sector"], inline=True)
    embed.add_field(name="Volume", value=f"{info['volume']:,}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="portfolio", description="Show portfolio holdings")
@app_commands.describe(user="Optional: View another user's portfolio")
async def stock_portfolio_command(
    interaction: discord.Interaction, user: discord.Member = None
):
    """View portfolio with pagination (5 stocks per page)"""
    target_user = user or interaction.user
    user_data = await get_user_data(bot.user_collection, target_user.id)
    portfolio = user_data["portfolio"]

    if not portfolio:
        message = (
            "Your portfolio is empty."
            if user is None
            else f"{target_user.display_name}'s portfolio is empty."
        )
        await interaction.response.send_message(message, ephemeral=True)
        return

    # Convert portfolio to list of (ticker, shares) for pagination
    portfolio_items = list(portfolio.items())
    portfolio_value, price_info = await calculate_portfolio_value(portfolio)

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        embed = discord.Embed(
            title=f"Portfolio - {target_user.display_name}", color=discord.Color.blue()
        )

        page_total = 0.0
        for ticker, shares in subset:
            if ticker in price_info:
                info = price_info[ticker]
                value = info["total_value"]
                page_total += value

                embed.add_field(
                    name=ticker,
                    value=f"{shares} shares @ ${info['price']:,.2f} each ({info['market_status']})\n"
                    f"Value: ${value:,.2f}",
                    inline=False,
                )

        embed.add_field(
            name="Portfolio Value",
            value=f"Page Total: ${page_total:,.2f}\nTotal Value: ${portfolio_value:,.2f}",
            inline=False,
        )

        embed.set_footer(text=f"Page {page_idx + 1}/{total_pages}")
        return embed

    view = PaginatorView(portfolio_items, 5, embed_factory, interaction.user.id)
    await view.send_first_page(interaction)


@stock_group.command(
    name="buyat", description="Set up an automatic buy order at a specific price"
)
@app_commands.describe(
    ticker="Stock ticker symbol",
    quantity="Number of shares to buy",
    price="Target price to buy at",
)
async def buyat_command(
    interaction: discord.Interaction, ticker: str, quantity: int, price: float
):
    """Create a pending buy order"""
    if quantity <= 0 or price <= 0:
        await interaction.response.send_message(
            "Please enter positive values.", ephemeral=True
        )
        return

    # Verify the ticker exists
    current_price = await get_stock_price(ticker.upper())
    if current_price is None:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    # Add the pending order
    success = await add_pending_order(
        bot, interaction.user.id, "buy", ticker, quantity, price
    )

    if not success:
        await interaction.response.send_message(
            "Insufficient funds for this order. Make sure you have enough balance.",
            ephemeral=True,
        )
        return

    total_cost = price * quantity
    embed = discord.Embed(
        title="Buy Order Created",
        description=f"Will buy {ticker.upper()} when price reaches ${price:,.2f}",
        color=discord.Color.green(),
    )
    embed.add_field(name="Quantity", value=str(quantity), inline=True)
    embed.add_field(name="Total Cost", value=f"${total_cost:,.2f}", inline=True)
    embed.add_field(name="Current Price", value=f"${current_price:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(
    name="sellat", description="Set up an automatic sell order at a specific price"
)
@app_commands.describe(
    ticker="Stock ticker symbol",
    quantity="Number of shares to sell",
    price="Target price to sell at",
)
async def sellat_command(
    interaction: discord.Interaction, ticker: str, quantity: int, price: float
):
    """Create a pending sell order"""
    if quantity <= 0 or price <= 0:
        await interaction.response.send_message(
            "Please enter positive values.", ephemeral=True
        )
        return

    # Verify the ticker exists and user has sufficient shares
    current_price = await get_stock_price(ticker.upper())
    if current_price is None:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    # Add the pending order
    success = await add_pending_order(
        bot, interaction.user.id, "sell", ticker, quantity, price
    )

    if not success:
        await interaction.response.send_message(
            "Insufficient shares for this order. Make sure you own enough shares.",
            ephemeral=True,
        )
        return

    total_value = price * quantity
    embed = discord.Embed(
        title="Sell Order Created",
        description=f"Will sell {ticker.upper()} when price reaches ${price:,.2f}",
        color=discord.Color.green(),
    )
    embed.add_field(name="Quantity", value=str(quantity), inline=True)
    embed.add_field(name="Total Value", value=f"${total_value:,.2f}", inline=True)
    embed.add_field(name="Current Price", value=f"${current_price:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@stock_group.command(name="pending", description="View your pending buy/sell orders")
async def pending_orders_command(interaction: discord.Interaction):
    """View pending orders with pagination"""
    await interaction.response.defer()

    orders = await get_user_pending_orders(bot, interaction.user.id)
    if not orders:
        await interaction.followup.send("You have no pending orders.", ephemeral=True)
        return

    class OrderPaginatorView(PaginatorView):
        async def cancel_order(
            self, button: discord.ui.Button, interaction: discord.Interaction
        ):
            order = self.items[self.current_page]
            success = await remove_pending_order(bot, order["_id"])
            if success:
                self.items.pop(self.current_page)
                if not self.items:
                    await interaction.response.edit_message(
                        content="No more pending orders.", embed=None, view=None
                    )
                    return
                self.total_pages = max(
                    1, math.ceil(len(self.items) / self.items_per_page)
                )
                await self.update_message(interaction)
            else:
                await interaction.response.send_message(
                    "Failed to cancel order.", ephemeral=True
                )

        @discord.ui.button(label="Cancel Order", style=discord.ButtonStyle.red)
        async def cancel_button(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self.cancel_order(button, interaction)

    def create_order_embed(
        page_idx: int, total_pages: int, subset: list
    ) -> discord.Embed:
        order = subset[0]  # Show one order per page
        order_type = order["type"].capitalize()
        ticker = order["ticker"]
        embed = discord.Embed(
            title=f"Pending Orders ({page_idx + 1}/{total_pages})",
            color=discord.Color.blue(),
        )

        total = order["price"] * order["quantity"]
        embed.add_field(name="Type", value=order_type, inline=True)
        embed.add_field(name="Stock", value=ticker, inline=True)
        embed.add_field(name="Quantity", value=str(order["quantity"]), inline=True)
        embed.add_field(
            name="Target Price", value=f"${order['price']:,.2f}", inline=True
        )
        embed.add_field(
            name=f"Total {'Cost' if order_type == 'Buy' else 'Value'}",
            value=f"${total:,.2f}",
            inline=True,
        )
        embed.add_field(
            name="Created",
            value=order["created_at"].strftime("%Y-%m-%d %H:%M UTC"),
            inline=True,
        )

        return embed

    view = OrderPaginatorView(orders, 1, create_order_embed, interaction.user.id)
    await view.send_first_page(interaction.followup, is_followup=True)


async def get_market_hours() -> Dict:
    """Get market hours and status for today"""
    et_tz = pytz.timezone("US/Eastern")
    now = dt.datetime.now(et_tz)

    # Market schedule (ET)
    pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    after_market_end = now.replace(hour=20, minute=0, second=0, microsecond=0)

    # Check if it's a weekday
    is_weekday = now.weekday() < 5

    # Determine current market phase
    if not is_weekday:
        status = "Closed (Weekend)"
    elif now < pre_market_start:
        status = "Closed (Pre-market starts at 4:00 AM ET)"
    elif pre_market_start <= now < market_open:
        status = "Pre-market Trading"
    elif market_open <= now < market_close:
        status = "Regular Trading Hours"
    elif market_close <= now < after_market_end:
        status = "After-hours Trading"
    else:
        status = "Closed"

    return {
        "status": status,
        "pre_market": "4:00 AM - 9:30 AM ET",
        "regular": "9:30 AM - 4:00 PM ET",
        "after_hours": "4:00 PM - 8:00 PM ET",
        "is_open": market_open <= now < market_close and is_weekday,
    }


async def get_major_indices() -> Dict:
    """Get major market indices performance"""
    try:
        indices = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}

        results = {}
        for symbol, name in indices.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("regularMarketPrice", 0)
            prev_close = info.get("previousClose", 0)
            if price and prev_close:
                change = price - prev_close
                change_pct = (change / prev_close) * 100
                results[name] = {
                    "price": price,
                    "change": change,
                    "change_pct": change_pct,
                }
        return results
    except Exception as e:
        print(f"Error fetching indices: {e}")
        return {}


async def get_top_movers(limit: int = 5) -> Dict[str, List]:
    """Get top gainers and losers for the day"""
    try:
        stocks = await get_top_stocks()
        if not stocks:
            return {"gainers": [], "losers": []}

        # Get current prices and calculate daily changes
        movers = []
        for stock in stocks:
            ticker = stock["ticker"]
            info = await get_stock_info(ticker)
            if info:
                price = info["price"]
                prev_close = info.get("previous_close", price)
                if prev_close:
                    change_pct = ((price - prev_close) / prev_close) * 100
                    movers.append(
                        {
                            "ticker": ticker,
                            "name": info["name"],
                            "price": price,
                            "change_pct": change_pct,
                        }
                    )

        # Sort and get top gainers/losers
        movers.sort(key=lambda x: x["change_pct"], reverse=True)
        return {"gainers": movers[:limit], "losers": movers[-limit:][::-1]}
    except Exception as e:
        print(f"Error getting top movers: {e}")
        return {"gainers": [], "losers": []}


@stock_group.command(name="today", description="Get today's market overview")
async def stock_today_command(interaction: discord.Interaction):
    """Show today's market status and trends"""
    await interaction.response.defer()

    # Get market hours first to check if it's weekend
    market_hours = await get_market_hours()

    # Create embed
    embed = discord.Embed(
        title="Market Overview",
        description=f"Market Status: {market_hours['status']}",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.utcnow(),
    )

    # If it's weekend, only show market hours
    if "Weekend" in market_hours["status"]:
        embed.add_field(
            name="Trading Hours (ET)",
            value=f"Pre-market: {market_hours['pre_market']}\n"
            f"Regular: {market_hours['regular']}\n"
            f"After-hours: {market_hours['after_hours']}",
            inline=False,
        )
        await interaction.followup.send(embed=embed)
        return

    # Get market data for weekdays
    indices = await get_major_indices()
    movers = await get_top_movers(5)

    # Add market performance section if we have data
    sp500_data = indices.get("S&P 500", {})
    dow_data = indices.get("Dow Jones", {})
    nasdaq_data = indices.get("NASDAQ", {})

    if sp500_data or dow_data or nasdaq_data:
        market_summary = ""

        if sp500_data:
            change = sp500_data["change"]
            pct = sp500_data["change_pct"]
            emoji = "🟢" if change >= 0 else "🔴"
            market_summary += (
                f"{emoji} **S&P 500**: {sp500_data['price']:,.2f}\n"
                f"    Change: {'+' if change >= 0 else ''}{change:,.2f} points ({pct:+.2f}%)\n"
            )

        if dow_data:
            change = dow_data["change"]
            pct = dow_data["change_pct"]
            emoji = "🟢" if change >= 0 else "🔴"
            market_summary += (
                f"{emoji} **Dow Jones**: {dow_data['price']:,.2f}\n"
                f"    Change: {'+' if change >= 0 else ''}{change:,.2f} points ({pct:+.2f}%)\n"
            )

        if nasdaq_data:
            change = nasdaq_data["change"]
            pct = nasdaq_data["change_pct"]
            emoji = "🟢" if change >= 0 else "🔴"
            market_summary += (
                f"{emoji} **NASDAQ**: {nasdaq_data['price']:,.2f}\n"
                f"    Change: {'+' if change >= 0 else ''}{change:,.2f} points ({pct:+.2f}%)\n"
            )

        if market_summary:
            embed.add_field(
                name="Market Performance", value=market_summary, inline=False
            )

    # Add trading hours field
    embed.add_field(
        name="Trading Hours (ET)",
        value=f"Pre-market: {market_hours['pre_market']}\n"
        f"Regular: {market_hours['regular']}\n"
        f"After-hours: {market_hours['after_hours']}",
        inline=False,
    )

    # Add gainers/losers only on weekdays and if we have data
    if movers["gainers"]:
        gainers_text = ""
        for stock in movers["gainers"]:
            gainers_text += f"**{stock['ticker']}**: +{stock['change_pct']:.2f}% (${stock['price']:,.2f})\n"
        embed.add_field(name="Top Gainers", value=gainers_text, inline=False)

    if movers["losers"]:
        losers_text = ""
        for stock in movers["losers"]:
            losers_text += f"**{stock['ticker']}**: {stock['change_pct']:.2f}% (${stock['price']:,.2f})\n"
        embed.add_field(name="Top Losers", value=losers_text, inline=False)

    # Add timestamp
    embed.set_footer(
        text=f"Values updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    await interaction.followup.send(embed=embed)


bot = StockBot()


@bot.tree.command(name="leaderboard", description="Show the top users by net worth.")
async def leaderboard_command(interaction: discord.Interaction):
    users = bot.user_collection.find({})
    leaderboard = []

    async for user in users:
        try:
            member = await interaction.guild.fetch_member(user["_id"])
            if member:
                # Get detailed portfolio value with real-time prices
                portfolio_value, _, crypto_value = await calculate_portfolio_value(
                    user["portfolio"],
                    user.get("crypto", {}),  # Add crypto portfolio
                )
                total_value = (
                    user["balance"] + portfolio_value + crypto_value
                )  # Include crypto value
                leaderboard.append((member, total_value))
        except discord.NotFound:
            continue

    if not leaderboard:
        await interaction.response.send_message("No users found.", ephemeral=True)
        return

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    embed = discord.Embed(title="Leaderboard", color=discord.Color.gold())

    for rank, (member, net_worth) in enumerate(leaderboard[:10], start=1):
        embed.add_field(
            name=f"{rank}. {member.display_name}",
            value=f"Net Worth: ${net_worth:,.2f}",
            inline=False,
        )

    # Add timestamp to show when values were last updated
    embed.set_footer(
        text=f"Values updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="resync", description="Resync all slash commands (Owner Only)")
async def resync_command(interaction: discord.Interaction):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    await interaction.response.defer()

    try:
        if ENVIRONMENT.lower() == "development":
            bot.tree.copy_global_to(guild=discord.Object(id=TEST_GUILD_ID))
            await bot.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
            await interaction.followup.send(
                "Slash commands resynced to test guild successfully."
            )
        else:
            await bot.tree.sync()
            await interaction.followup.send(
                "Slash commands resynced globally successfully."
            )
    except Exception as e:
        await interaction.followup.send(
            f"Error resyncing commands: {str(e)}", ephemeral=True
        )


class PineStrategyModal(ui.Modal, title="Pine Strategy Input"):
    pine_code = ui.TextInput(
        label="Pine Strategy Code",
        style=TextStyle.paragraph,
        placeholder="Paste your Pine strategy code here...",
        required=True,
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.process_strategy(interaction)

    async def process_strategy(self, interaction: discord.Interaction):
        try:
            # Store the code for access in the callback
            self.view.pine_code = self.pine_code.value
            await self.view.execute_strategy(interaction)
        except Exception as e:
            await interaction.followup.send(
                f"Error processing strategy: {str(e)}", ephemeral=True
            )


class PineStrategyView(discord.ui.View):
    def __init__(self, ticker: str, period: str, initial_amount: float):
        super().__init__()
        self.ticker = ticker
        self.period = period
        self.initial_amount = initial_amount
        self.pine_code = None

    async def execute_strategy(self, interaction: discord.Interaction):
        try:
            print(f"Starting strategy execution for {self.ticker} over {self.period}")

            history = await get_stock_history(self.ticker, self.period)
            if history is None or history.empty:
                await interaction.followup.send(
                    f"Unable to fetch sufficient stock data for {self.ticker} over {self.period}. "
                    "Try a different time period.",
                    ephemeral=True,
                )
                return

            print(f"Got {len(history)} historical data points")

            # Process strategy
            signals = await self.process_pine_strategy(history)
            if signals is None:
                await interaction.followup.send(
                    "Unable to generate trading signals. Ensure you have sufficient data points.",
                    ephemeral=True,
                )
                return

            # Calculate returns and generate chart
            final_amount, percent_gain = self.calculate_returns(history, signals)
            buf = await self.create_strategy_chart(history, signals)

            # Create detailed embed
            embed = discord.Embed(
                title=f"Strategy Test Results for {self.ticker}",
                color=discord.Color.blue(),
            )

            # Summary stats
            embed.add_field(
                name="Initial Investment",
                value=f"${self.initial_amount:,.2f}",
                inline=True,
            )
            embed.add_field(
                name="Final Amount", value=f"${final_amount:,.2f}", inline=True
            )
            embed.add_field(
                name="Total Return", value=f"{percent_gain:,.2f}%", inline=True
            )

            # Trading stats
            buy_signals = len(signals[signals["buy_signal"]].index)
            sell_signals = len(signals[signals["sell_signal"]].index)
            embed.add_field(name="Buy Signals", value=str(buy_signals), inline=True)
            embed.add_field(name="Sell Signals", value=str(sell_signals), inline=True)
            embed.add_field(name="Period", value=self.period, inline=True)

            # Send results
            file = discord.File(buf, filename="strategy_chart.png")
            embed.set_image(url="attachment://strategy_chart.png")
            await interaction.followup.send(embed=embed, file=file)

        except Exception as e:
            print(f"Strategy execution error: {e}")
            await interaction.followup.send(
                f"Error executing strategy: {str(e)}", ephemeral=True
            )

    async def process_pine_strategy(self, history: pd.DataFrame) -> pd.DataFrame:
        try:
            df = history.copy()

            # Ensure we have enough data points for meaningful analysis
            min_required_points = 30  # At least 30 data points for reliable signals
            if len(df) < min_required_points:
                print(f"Insufficient data points: {len(df)} < {min_required_points}")
                return None

            # Rest of the existing process_pine_strategy code...
            # ...existing code...

            # Ensure Close column exists and has valid data
            if "Close" not in df.columns or df["Close"].empty:
                print("No valid price data found")
                return None

            # Convert Close prices to numeric and handle NaN values
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df["Close"] = df["Close"].ffill().bfill()

            # Debug log
            print(f"Data shape: {df.shape}")
            print(f"Close price range: {df['Close'].min()} - {df['Close'].max()}")
            print(f"Sample of Close prices: {df['Close'].head()}")

            # Extract MA length
            ma_length_match = re.search(
                r"maLength\s*=\s*input\.int\s*\(\s*(\d+)", self.pine_code
            )
            ma_length = int(ma_length_match.group(1)) if ma_length_match else 14
            print(f"Using MA length: {ma_length}")

            # Calculate Moving Average with fallback options
            try:
                # Try pandas_ta first
                ma_series = ta.sma(df["Close"], length=ma_length)

                # If pandas_ta fails, use pandas rolling mean
                if (
                    ma_series is None
                    or isinstance(ma_series, pd.DataFrame)
                    and ma_series.empty
                ):
                    print("Falling back to pandas rolling mean")
                    ma_series = (
                        df["Close"].rolling(window=ma_length, min_periods=1).mean()
                    )

                # Convert to pandas Series if needed
                if isinstance(ma_series, pd.DataFrame):
                    ma_series = ma_series.iloc[:, 0]

                # Handle NaN values in MA
                df["ma"] = ma_series
                df["ma"] = df["ma"].ffill().bfill()

                print(f"MA calculation successful")
                print(f"MA range: {df['ma'].min()} - {df['ma'].max()}")
                print(f"Sample of MA values: {df['ma'].head()}")

            except Exception as e:
                print(f"Error in MA calculation: {e}")
                print(f"Data types - Close: {df['Close'].dtype}")
                return None

            # Additional validation
            if df["ma"].isna().any():
                print("Warning: MA contains NaN values after calculation")
                df["ma"] = df["ma"].ffill().bfill()

            # Generate signals with validation
            df["buy_signal"] = False
            df["sell_signal"] = False

            valid_mask = df["Close"].notna() & df["ma"].notna()
            if not valid_mask.any():
                print("No valid data points for signal generation")
                return None

            # Calculate signals on valid data only
            df.loc[valid_mask, "buy_signal"] = (df["Close"] > df["ma"]) & (
                df["Close"].shift(1) <= df["ma"].shift(1)
            )

            df.loc[valid_mask, "sell_signal"] = (df["Close"] < df["ma"]) & (
                df["Close"].shift(1) >= df["ma"].shift(1)
            )

            print(
                f"Signal generation complete. Buy signals: {df['buy_signal'].sum()}, Sell signals: {df['sell_signal'].sum()}"
            )
            return df

        except Exception as e:
            print(f"Error processing Pine strategy: {e}")
            return None

    async def create_strategy_chart(
        self, history: pd.DataFrame, signals: pd.DataFrame
    ) -> io.BytesIO:
        plt.figure(figsize=(12, 6))

        # Plot price and MA
        plt.plot(
            history.index, history["Close"], label="Price", color="blue", alpha=0.8
        )
        plt.plot(history.index, signals["ma"], label="MA", color="orange", alpha=0.6)

        # Plot buy/sell signals
        buy_points = signals[signals["buy_signal"]].index
        sell_points = signals[signals["sell_signal"]].index

        if not buy_points.empty:
            plt.scatter(
                buy_points,
                signals.loc[buy_points, "Close"],
                color="green",
                marker="^",
                s=100,
                label="Buy Signal",
            )
        if not sell_points.empty:
            plt.scatter(
                sell_points,
                signals.loc[sell_points, "Close"],
                color="red",
                marker="v",
                s=100,
                label="Sell Signal",
            )

        plt.title(f"{self.ticker} Strategy Backtest - {self.period}")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Format date axis
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close()

        return buf

    def calculate_returns(
        self, history: pd.DataFrame, signals: pd.DataFrame
    ) -> tuple[float, float]:
        current_amount = self.initial_amount
        current_shares = 0
        trades = []

        for i in range(len(signals)):
            price = signals["Close"].iloc[i]

            # Buy signal
            if signals["buy_signal"].iloc[i] and current_shares == 0:
                current_shares = current_amount / price
                current_amount = 0
                trades.append(("BUY", price, current_shares))

            # Sell signal
            elif signals["sell_signal"].iloc[i] and current_shares > 0:
                current_amount = current_shares * price
                current_shares = 0
                trades.append(("SELL", price, current_amount))

        # Close any remaining position
        if current_shares > 0:
            current_amount = current_shares * signals["Close"].iloc[-1]
            trades.append(("FINAL", signals["Close"].iloc[-1], current_amount))

        percent_gain = (
            (current_amount - self.initial_amount) / self.initial_amount
        ) * 100

        # Print trades for debugging
        print(f"Strategy Trades for {self.ticker}:")
        for trade_type, price, amount in trades:
            print(f"{trade_type}: Price=${price:.2f}, Amount=${amount:.2f}")

        return round(current_amount, 2), round(percent_gain, 2)


async def fetch_algorithm_code(url: str) -> Optional[str]:
    """Fetch and validate algorithm code from URL."""
    try:
        async with aiohttp.ClientSession() as session:
            # Add headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None

                # Get content type and try to handle various formats
                content_type = response.headers.get("content-type", "").lower()

                # Allow more content types that might contain raw code
                valid_content_types = [
                    "text/plain",
                    "text/markdown",
                    "application/x-httpd-php",
                    "text/html",
                    "application/octet-stream",  # GitHub raw files sometimes use this
                ]

                # Special handling for GitHub raw URLs
                if "githubusercontent.com" in url:
                    content = await response.text()
                    return content if content.strip() else None

                # Check if content type is acceptable
                if not any(t in content_type for t in valid_content_types):
                    return None

                content = await response.text()

                # Basic validation of content
                if not content.strip():
                    return None

                # Check if content looks like code (contains specific keywords)
                code_indicators = ["//@version=", "indicator(", "strategy(", "study("]
                if not any(indicator in content for indicator in code_indicators):
                    return None

                return content

    except Exception as e:
        print(f"Error fetching algorithm code: {e}")
        return None


# Add crypto command group
crypto_group = app_commands.Group(
    name="crypto", description="Cryptocurrency related commands"
)


# Add after the imports section
def format_crypto_price(price: float) -> str:
    """Format crypto price with appropriate precision based on price range"""
    if price < 1:
        return f"${price:.8f}"  # 8 digits for sub-dollar prices
    elif price < 10:
        return f"${price:.4f}"  # 4 digits for under $10
    else:
        return f"${price:,.2f}"  # 2 digits for $10+


@crypto_group.command(name="buy", description="Buy cryptocurrency")
@app_commands.describe(
    ticker="Crypto ticker symbol (e.g. BTC, ETH)", amount="Amount to buy (minimum 0.01)"
)
async def crypto_buy_command(
    interaction: discord.Interaction, ticker: str, amount: float
):
    if amount < float(CRYPTO_MIN_TRADE):
        await interaction.response.send_message(
            f"Minimum trade amount is {CRYPTO_MIN_TRADE} units.", ephemeral=True
        )
        return

    price = await get_crypto_price(ticker.upper())
    if not price:
        await interaction.response.send_message(
            "Invalid cryptocurrency symbol.", ephemeral=True
        )
        return

    total_cost = price * amount
    user_data = await get_user_data(bot.user_collection, interaction.user.id)

    if user_data["balance"] < total_cost:
        await interaction.response.send_message("Insufficient funds.", ephemeral=True)
        return

    new_balance = user_data["balance"] - total_cost
    crypto = user_data.get("crypto", {})
    crypto[ticker.upper()] = crypto.get(ticker.upper(), 0) + amount

    transaction = {
        "type": "crypto_buy",
        "ticker": ticker.upper(),
        "amount": amount,
        "price": price,
        "total": total_cost,
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "crypto": crypto},
            "$push": {"transactions": transaction},
        },
    )

    embed = discord.Embed(
        title="Crypto Purchase Successful", color=discord.Color.green()
    )
    embed.add_field(name="Cryptocurrency", value=ticker.upper(), inline=True)
    embed.add_field(name="Amount", value=str(amount), inline=True)
    embed.add_field(name="Price/Unit", value=format_crypto_price(price), inline=True)
    embed.add_field(name="Total Cost", value=f"${total_cost:,.2f}", inline=True)
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@crypto_group.command(name="sell", description="Sell cryptocurrency")
@app_commands.describe(
    ticker="Crypto ticker symbol (e.g. BTC, ETH)",
    amount="Amount to sell (minimum 0.01)",
)
async def crypto_sell_command(
    interaction: discord.Interaction, ticker: str, amount: float
):
    if amount < float(CRYPTO_MIN_TRADE):
        await interaction.response.send_message(
            f"Minimum trade amount is {CRYPTO_MIN_TRADE} units.", ephemeral=True
        )
        return

    price = await get_crypto_price(ticker.upper())
    if not price:
        await interaction.response.send_message(
            "Invalid cryptocurrency symbol.", ephemeral=True
        )
        return

    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    crypto = user_data.get("crypto", {})

    if ticker.upper() not in crypto or crypto[ticker.upper()] < amount:
        await interaction.response.send_message(
            "Insufficient cryptocurrency balance.", ephemeral=True
        )
        return

    total_value = price * amount
    new_balance = user_data["balance"] + total_value

    crypto[ticker.upper()] -= amount
    if crypto[ticker.upper()] == 0:
        del crypto[ticker.upper()]

    transaction = {
        "type": "crypto_sell",
        "ticker": ticker.upper(),
        "amount": amount,
        "price": price,
        "total": total_value,
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "crypto": crypto},
            "$push": {"transactions": transaction},
        },
    )

    embed = discord.Embed(title="Crypto Sale Successful", color=discord.Color.green())
    embed.add_field(name="Cryptocurrency", value=ticker.upper(), inline=True)
    embed.add_field(name="Amount", value=str(amount), inline=True)
    embed.add_field(name="Price/Unit", value=format_crypto_price(price), inline=True)
    embed.add_field(name="Total Value", value=f"${total_value:,.2f}", inline=True)
    embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)

    await interaction.response.send_message(embed=embed)


@crypto_group.command(name="chart", description="Show price chart for a cryptocurrency")
@app_commands.describe(
    ticker="Crypto ticker symbol (e.g. BTC, ETH)",
    period="Time period for chart",
    chart_type="Chart visualization type",
)
@app_commands.choices(
    period=[
        app_commands.Choice(name=p, value=p)
        for p in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    ],
    chart_type=[
        app_commands.Choice(name=t, value=t)
        for t in ["line", "candle", "mountain", "baseline"]
    ],
)
async def crypto_chart_command(
    interaction: discord.Interaction, ticker: str, period: str, chart_type: str = "line"
):
    await interaction.response.defer()

    # Convert crypto ticker to yfinance format
    yf_ticker = f"{ticker.upper()}-USD"

    # Get historical data using the existing get_stock_history function
    history = await get_stock_history(yf_ticker, period)
    if history is None:
        await interaction.followup.send(
            "Unable to fetch cryptocurrency data. Make sure you're using a valid symbol (e.g. BTC, ETH).",
            ephemeral=True,
        )
        return

    # Create chart using existing chart creation logic
    plt.figure(figsize=(10, 6))

    if chart_type == "line":
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "candle":
        import matplotlib.dates as mdates
        from mplfinance.original_flavor import candlestick_ohlc

        history["Date"] = mdates.date2num(history.index.to_pydatetime())
        ohlc = history[["Date", "Open", "High", "Low", "Close"]].values
        candlestick_ohlc(plt.gca(), ohlc, width=0.6, colorup="green", colordown="red")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    elif chart_type == "mountain":
        plt.fill_between(history.index, history["Close"], alpha=0.3, color="blue")
        plt.plot(history.index, history["Close"], color="blue", alpha=0.8)

    elif chart_type == "baseline":
        baseline = history["Close"].mean()
        plt.fill_between(
            history.index,
            history["Close"],
            baseline,
            where=(history["Close"] >= baseline),
            color="green",
            alpha=0.3,
        )
        plt.fill_between(
            history.index,
            history["Close"],
            baseline,
            where=(history["Close"] < baseline),
            color="red",
            alpha=0.3,
        )
        plt.axhline(y=baseline, color="gray", linestyle="--")

    plt.title(f"{ticker.upper()} {chart_type.title()} Chart ({period})")
    plt.xlabel("Date/Time")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add current price point
    current_price = history["Close"].iloc[-1]
    plt.scatter(history.index[-1], current_price, color="green", s=100, zorder=5)

    plt.tight_layout()

    # Save chart to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close()

    # Create embed with price information
    embed = discord.Embed(
        title=f"{ticker.upper()} Price Chart", color=discord.Color.blue()
    )

    # Get current price info
    price_change = history["Close"].iloc[-1] - history["Close"].iloc[0]
    price_change_pct = (price_change / history["Close"].iloc[0]) * 100

    embed.add_field(
        name="Current Price", value=format_crypto_price(current_price), inline=True
    )
    embed.add_field(
        name="Change",
        value=f"{format_crypto_price(price_change)} ({price_change_pct:,.2f}%)",
        inline=True,
    )
    embed.add_field(name="Period", value=period, inline=True)
    embed.add_field(name="Chart Type", value=chart_type.title(), inline=True)

    # Send chart
    file = discord.File(buf, filename=f"{ticker}_chart.png")
    embed.set_image(url=f"attachment://{ticker}_chart.png")
    await interaction.followup.send(file=file, embed=embed)


@crypto_group.command(
    name="lookup", description="Get information about a cryptocurrency"
)
@app_commands.describe(ticker="Crypto ticker symbol (e.g. BTC, ETH)")
async def crypto_lookup_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()  # Add this line to prevent timeout

    info = await get_crypto_info(ticker.upper())
    if not info:
        await interaction.followup.send(  # Change to followup
            "Invalid or unsupported cryptocurrency symbol. Only major cryptocurrencies are supported.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=f"{info['name']} ({info['symbol']})",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.utcnow(),
    )

    # Add more debug information
    print(f"Processing crypto info for {ticker}:")
    print(f"Name: {info['name']}")
    print(f"Symbol: {info['symbol']}")
    print(f"Price: ${info['price']}")

    embed.add_field(
        name="Current Price",
        value=format_crypto_price(info["price"]),
        inline=False,
    )

    # Add 24h change if available
    if info["price_change_24h"]:
        embed.add_field(
            name="24h Change",
            value=f"{info['price_change_24h']:+.2f}%",
            inline=True,
        )

    embed.add_field(
        name="Market Cap",
        value=f"${info['market_cap']:,.2f}",
        inline=True,
    )
    embed.add_field(
        name="24h Volume",
        value=f"${info['volume_24h']:,.2f}",
        inline=True,
    )
    embed.add_field(
        name="Circulating Supply",
        value=f"{info['circulating_supply']:,.0f} {info['symbol']}",
        inline=True,
    )
    if info["total_supply"]:
        embed.add_field(
            name="Total Supply",
            value=f"{info['total_supply']:,.0f} {info['symbol']}",
            inline=True,
        )

    await interaction.followup.send(embed=embed)  # Change to followup


@bot.tree.command(name="history", description="Show your trading history")
@app_commands.describe(user="Optional: View another user's history")
async def history_command(
    interaction: discord.Interaction, user: discord.Member = None
):
    """View trading history with pagination (5 trades per page)"""
    target_user = user or interaction.user
    user_data = await get_user_data(bot.user_collection, target_user.id)
    transactions = user_data["transactions"]

    if not transactions:
        await interaction.response.send_message(
            f"{'You have' if user is None else f'{target_user.display_name} has'} no transaction history.",
            ephemeral=True,
        )
        return

    transactions.sort(key=lambda x: x["timestamp"], reverse=True)

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        embed = discord.Embed(
            title=f"Transaction History - {target_user.display_name}",
            color=discord.Color.purple(),
        )

        for tx in subset:
            timestamp = tx["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")

            if tx["type"] in ["crypto_buy", "crypto_sell"]:
                action = "Buy" if tx["type"] == "crypto_buy" else "Sell"
                embed.add_field(
                    name=f"Crypto {action} {tx['amount']} {tx['ticker']}",
                    value=f"Price: {format_crypto_price(tx['price'])}\nTotal: ${tx['total']:,.2f}\nDate: {timestamp}",
                    inline=False,
                )
            else:  # Stock transactions
                embed.add_field(
                    name=f"{tx['type'].capitalize()} {tx['shares']} {tx['ticker']}",
                    value=f"Price: ${tx['price']:,.2f}\nTotal: ${tx['total']:,.2f}\nDate: {timestamp}",
                    inline=False,
                )

        embed.set_footer(text=f"Page {page_idx + 1}/{total_pages}")
        return embed

    view = PaginatorView(transactions, 5, embed_factory, interaction.user.id)
    await view.send_first_page(interaction)


@bot.tree.command(name="balance", description="Show your cash balance and net worth")
@app_commands.describe(user="Optional: View another user's balance")
async def balance_command(
    interaction: discord.Interaction, user: discord.Member = None
):
    """Show user's balance including crypto holdings"""
    target_user = user or interaction.user
    user_data = await get_user_data(bot.user_collection, target_user.id)
    balance = round(user_data["balance"], 2)

    # Get detailed portfolio values
    portfolio_value, price_info, crypto_value = await calculate_portfolio_value(
        user_data["portfolio"], user_data.get("crypto", {})
    )
    total_value = round(balance + portfolio_value + crypto_value, 2)

    embed = discord.Embed(
        title=f"Balance Sheet - {target_user.display_name}", color=discord.Color.green()
    )
    embed.add_field(name="Cash Balance", value=f"${balance:,.2f}", inline=True)
    embed.add_field(
        name="Stock Portfolio", value=f"${portfolio_value:,.2f}", inline=True
    )
    embed.add_field(
        name="Crypto Portfolio", value=format_crypto_price(crypto_value), inline=True
    )
    embed.add_field(name="Total Net Worth", value=f"${total_value:,.2f}", inline=False)

    # Add timestamp to show when values were last updated
    embed.set_footer(
        text=f"Values updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    await interaction.response.send_message(embed=embed)


class SellAllConfirmationView(discord.ui.View):
    def __init__(self, user_data: dict, total_value: float, portfolio_details: str):
        super().__init__()
        self.user_data = user_data
        self.total_value = total_value
        self.portfolio_details = portfolio_details

    @discord.ui.button(label="I Confirm", style=discord.ButtonStyle.danger)
    async def confirm_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        modal = SellAllConfirmModal(self.user_data, self.total_value)
        await interaction.response.send_modal(modal)


class SellAllConfirmModal(discord.ui.Modal, title="Confirm Sell All"):
    confirmation = discord.ui.TextInput(
        label="Type 'CONFIRM' to sell everything", placeholder="CONFIRM", required=True
    )

    def __init__(self, user_data: dict, total_value: float):
        super().__init__()
        self.user_data = user_data
        self.total_value = total_value

    async def on_submit(self, interaction: discord.Interaction):
        if self.confirmation.value != "CONFIRM":
            await interaction.response.send_message(
                "Sell all cancelled - incorrect confirmation.", ephemeral=True
            )
            return

        # Execute sell all
        new_balance = self.user_data["balance"] + self.total_value

        # Create transaction records
        transactions = []
        # Add stock transactions
        for ticker, shares in self.user_data["portfolio"].items():
            price = (await get_stock_price(ticker)) or 0
            transactions.append(
                {
                    "type": "sell",
                    "ticker": ticker,
                    "shares": shares,
                    "price": price,
                    "total": price * shares,
                    "timestamp": datetime.datetime.utcnow(),
                }
            )

        # Add crypto transactions
        for ticker, amount in self.user_data.get("crypto", {}).items():
            price = (await get_crypto_price(ticker)) or 0
            transactions.append(
                {
                    "type": "crypto_sell",
                    "ticker": ticker,
                    "amount": amount,
                    "price": price,
                    "total": price * amount,
                    "timestamp": datetime.datetime.utcnow(),
                }
            )

        # Update database
        await bot.user_collection.update_one(
            {"_id": interaction.user.id},
            {
                "$set": {"balance": new_balance, "portfolio": {}, "crypto": {}},
                "$push": {"transactions": {"$each": transactions}},
            },
        )

        embed = discord.Embed(
            title="Portfolio Liquidated",
            description="All stocks and cryptocurrencies have been sold.",
            color=discord.Color.green(),
        )
        embed.add_field(
            name="Total Value Received", value=f"${self.total_value:,.2f}", inline=True
        )
        embed.add_field(name="New Balance", value=f"${new_balance:,.2f}", inline=True)

        await interaction.response.send_message(embed=embed)


# Remove the existing portfolio command from stock_group and add as root command
@bot.tree.command(
    name="portfolio",
    description="Show your complete portfolio including stocks and crypto",
)
@app_commands.describe(user="Optional: View another user's portfolio")
async def portfolio_command(
    interaction: discord.Interaction, user: discord.Member = None
):
    """View complete portfolio with pagination"""
    target_user = user or interaction.user
    user_data = await get_user_data(bot.user_collection, target_user.id)

    if not user_data["portfolio"] and not user_data.get("crypto", {}):
        message = (
            "Your portfolio is empty."
            if user is None
            else f"{target_user.display_name}'s portfolio is empty."
        )
        await interaction.response.send_message(message, ephemeral=True)
        return

    portfolio_value, price_info, crypto_value = await calculate_portfolio_value(
        user_data["portfolio"], user_data.get("crypto", {})
    )

    # Combine stocks and crypto into one list for pagination
    portfolio_items = []

    # Add stocks
    for ticker, shares in user_data["portfolio"].items():
        if ticker in price_info:
            portfolio_items.append(
                {
                    "type": "stock",
                    "ticker": ticker,
                    "amount": shares,
                    "price_info": price_info[ticker],
                }
            )

    # Add crypto
    for ticker, amount in user_data.get("crypto", {}).items():
        if f"{ticker}-USD" in price_info:
            portfolio_items.append(
                {
                    "type": "crypto",
                    "ticker": ticker,
                    "amount": amount,
                    "price_info": price_info[f"{ticker}-USD"],
                }
            )

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        embed = discord.Embed(
            title=f"Portfolio - {target_user.display_name}", color=discord.Color.blue()
        )

        page_total = 0.0
        for item in subset:
            value = item["price_info"]["total_value"]
            page_total += value

            if item["type"] == "stock":
                embed.add_field(
                    name=f"Stock: {item['ticker']}",
                    value=f"{item['amount']} shares @ ${item['price_info']['price']:,.2f} each\n"
                    f"Value: ${value:,.2f}",
                    inline=False,
                )
            else:  # crypto
                embed.add_field(
                    name=f"Crypto: {item['ticker']}",
                    value=f"{item['amount']} tokens @ {format_crypto_price(item['price_info']['price'])} each\n"
                    f"Value: {format_crypto_price(value)}",
                    inline=False,
                )

        total_assets = portfolio_value + crypto_value
        embed.add_field(
            name="Portfolio Summary",
            value=f"Page Total: {format_crypto_price(page_total)}\n"
            f"Stocks Value: ${portfolio_value:,.2f}\n"
            f"Crypto Value: {format_crypto_price(crypto_value)}\n"
            f"Total Assets: {format_crypto_price(total_assets)}\n"
            f"Cash Balance: ${user_data['balance']:,.2f}\n"
            f"Net Worth: ${(total_assets + user_data['balance']):,.2f}",
            inline=False,
        )

        embed.set_footer(text=f"Page {page_idx + 1}/{total_pages}")
        return embed

    view = PaginatorView(portfolio_items, 5, embed_factory, interaction.user.id)
    await view.send_first_page(interaction)


@bot.tree.command(name="sell_all", description="Sell all stocks and cryptocurrencies")
async def sell_all_command(interaction: discord.Interaction):
    """Sell entire portfolio with confirmation"""
    user_data = await get_user_data(bot.user_collection, interaction.user.id)

    if not user_data["portfolio"] and not user_data.get("crypto", {}):
        await interaction.response.send_message(
            "You have no assets to sell.", ephemeral=True
        )
        return

    # Calculate current values and prepare portfolio details
    portfolio_value, price_info, crypto_value = await calculate_portfolio_value(
        user_data["portfolio"], user_data.get("crypto", {})
    )

    total_value = portfolio_value + crypto_value
    details = []

    # Add stock details
    for ticker, shares in user_data["portfolio"].items():
        if ticker in price_info:
            details.append(
                f"Stock: {ticker}\n"
                f"Quantity: {shares} shares\n"
                f"Current Price: ${price_info[ticker]['price']:,.2f}\n"
                f"Total Value: ${price_info[ticker]['total_value']:,.2f}\n"
            )

    # Add crypto details
    for ticker, amount in user_data.get("crypto", {}).items():
        key = f"{ticker}-USD"
        if key in price_info:
            details.append(
                f"Crypto: {ticker}\n"
                f"Amount: {amount}\n"
                f"Current Price: {format_crypto_price(price_info[key]['price'])}\n"
                f"Total Value: ${price_info[key]['total_value']:,.2f}\n"
            )

    embed = discord.Embed(
        title="⚠️ Confirm Sell All Assets",
        description="You are about to sell all your stocks and cryptocurrencies.",
        color=discord.Color.yellow(),
    )

    embed.add_field(
        name="Portfolio Summary",
        value=f"Stocks Value: ${portfolio_value:,.2f}\n"
        f"Crypto Value: {format_crypto_price(crypto_value)}\n"
        f"Total Sale Value: {format_crypto_price(total_value)}\n"
        f"Current Balance: ${user_data['balance']:,.2f}\n"
        f"Balance After Sale: {format_crypto_price(user_data['balance'] + total_value)}",
        inline=False,
    )

    # Add asset details
    for detail in details:
        embed.add_field(name="Asset Details", value=detail, inline=True)

    view = SellAllConfirmationView(user_data, total_value, "\n".join(details))
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


def main():
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
