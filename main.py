import asyncio
import datetime
import io
import math
import os
from typing import Dict, List, Optional

import discord
import matplotlib.pyplot as plt
import motor.motor_asyncio
import pandas as pd
import yfinance as yf
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

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

    async def send_first_page(
        self, interaction: discord.Interaction, is_followup: bool = False
    ):
        embed = self.build_embed()
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

    async def setup_hook(self):
        try:
            if ENVIRONMENT.lower() == "development":
                self.tree.copy_global_to(guild=discord.Object(id=TEST_GUILD_ID))
                await self.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
            else:
                await self.tree.sync()
        except Exception as e:
            print(f"Error syncing commands: {e}")

        # Setup MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = client[MONGO_DB_NAME]
        self.user_collection = self.db["users"]

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching, name="the Stock Market"
            )
        )


async def get_stock_price(ticker: str) -> Optional[float]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get regular market price first
        price = info.get("regularMarketPrice")

        # Check for pre-market price
        pre_market = info.get("preMarketPrice")
        if pre_market:
            return float(pre_market)

        # Check for after-hours price
        post_market = info.get("postMarketPrice")
        if post_market:
            return float(post_market)

        # Fallback to regular price or previous close
        return float(price or info.get("previousClose", 0))
    except Exception as e:
        print(f"Error getting price for {ticker}: {e}")
        return None


async def get_stock_info(ticker: str) -> Optional[Dict]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get all market prices
        regular_price = info.get("regularMarketPrice", 0.0)
        pre_market = info.get("preMarketPrice")
        post_market = info.get("postMarketPrice")

        # Determine current price based on market hours
        current_price = regular_price
        market_status = "Regular Hours"

        if pre_market:
            current_price = pre_market
            market_status = "Pre-Market"
        elif post_market:
            current_price = post_market
            market_status = "After-Hours"

        # Get market cap with fallback
        market_cap = info.get("marketCap") or info.get("totalMarketCap")

        # Other data...
        pe_ratio = info.get("forwardPE") or info.get("trailingPE") or 0.0
        div_yield = (
            info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
        )
        if div_yield:
            div_yield = div_yield * 100

        return {
            "name": info.get("longName") or info.get("shortName", "Unknown"),
            "price": float(current_price),
            "market_status": market_status,
            "regular_price": float(regular_price) if regular_price else 0.0,
            "pre_market": float(pre_market) if pre_market else None,
            "post_market": float(post_market) if post_market else None,
            "market_cap": float(market_cap) if market_cap else 0,
            "pe_ratio": float(pe_ratio),
            "dividend_yield": float(div_yield),
            "sector": info.get("sector", "Unknown"),
            "volume": info.get("volume") or info.get("averageVolume", 0),
        }
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return None


async def get_user_data(user_collection, user_id: int):
    user_data = await user_collection.find_one({"_id": user_id})
    if not user_data:
        user_data = {
            "_id": user_id,
            "balance": INITIAL_BALANCE,
            "portfolio": {},
            "transactions": [],
        }
        await user_collection.insert_one(user_data)
    return user_data


async def calculate_portfolio_value(portfolio: Dict[str, int]) -> float:
    total = 0.0
    for ticker, shares in portfolio.items():
        price = await get_stock_price(ticker)
        if price:
            total += price * shares
    return round(total, 2)


async def get_top_stocks() -> List[Dict]:
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


bot = StockBot()


@bot.tree.command(name="balance", description="Show your cash balance.")
async def balance_command(interaction: discord.Interaction):
    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    balance = round(user_data["balance"], 2)
    portfolio_value = await calculate_portfolio_value(user_data["portfolio"])
    total_value = round(balance + portfolio_value, 2)

    embed = discord.Embed(title="Balance Sheet", color=discord.Color.green())
    embed.add_field(name="Cash Balance", value=f"${balance:,.2f}", inline=True)
    embed.add_field(
        name="Portfolio Value", value=f"${portfolio_value:,.2f}", inline=True
    )
    embed.add_field(name="Total Net Worth", value=f"${total_value:,.2f}", inline=False)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="buy", description="Buy shares of a stock.")
@app_commands.describe(ticker="Ticker symbol", shares="Number of shares to buy")
async def buy_command(interaction: discord.Interaction, ticker: str, shares: int):
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


@bot.tree.command(name="sell", description="Sell shares of a stock.")
@app_commands.describe(ticker="Ticker symbol", shares="Number of shares to sell")
async def sell_command(interaction: discord.Interaction, ticker: str, shares: int):
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


@bot.tree.command(name="portfolio", description="Show your portfolio.")
async def portfolio_command(interaction: discord.Interaction):
    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    portfolio = user_data["portfolio"]

    if not portfolio:
        await interaction.response.send_message(
            "Your portfolio is empty.", ephemeral=True
        )
        return

    embed = discord.Embed(title="Your Portfolio", color=discord.Color.blue())
    total_value = 0.0
    for ticker, shares in portfolio.items():
        price = await get_stock_price(ticker)
        if price:
            value = price * shares
            total_value += value
            embed.add_field(
                name=ticker,
                value=f"{shares} shares @ ${price:,.2f} each\nValue: ${value:,.2f}",
                inline=False,
            )

    embed.add_field(
        name="Total Portfolio Value", value=f"${total_value:,.2f}", inline=False
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="stock", description="Get information about a stock.")
@app_commands.describe(ticker="Ticker symbol")
async def stock_command(interaction: discord.Interaction, ticker: str):
    info = await get_stock_info(ticker.upper())
    if not info:
        await interaction.response.send_message(
            "Invalid ticker symbol.", ephemeral=True
        )
        return

    embed = discord.Embed(
        title=f"{info['name']} ({ticker.upper()})", color=discord.Color.blue()
    )

    # Price field with market status
    price_text = f"${info['price']:,.2f} ({info['market_status']})"
    if info["market_status"] != "Regular Hours":
        price_text += f"\nRegular Hours: ${info['regular_price']:,.2f}"
    embed.add_field(name="Price", value=price_text, inline=True)

    # Add pre/post market prices if available
    if info["pre_market"]:
        embed.add_field(
            name="Pre-Market", value=f"${info['pre_market']:,.2f}", inline=True
        )
    if info["post_market"]:
        embed.add_field(
            name="After-Hours", value=f"${info['post_market']:,.2f}", inline=True
        )

    embed.add_field(name="Market Cap", value=f"${info['market_cap']:,.2f}", inline=True)
    embed.add_field(name="P/E Ratio", value=f"{info['pe_ratio']:.2f}", inline=True)
    embed.add_field(
        name="Dividend Yield", value=f"{info['dividend_yield']:.2f}%", inline=True
    )
    embed.add_field(name="Sector", value=info["sector"], inline=True)
    embed.add_field(name="Volume", value=f"{info['volume']:,}", inline=True)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="history", description="Show your transaction history.")
async def history_command(interaction: discord.Interaction):
    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    transactions = user_data["transactions"]

    if not transactions:
        await interaction.response.send_message(
            "You have no transaction history.", ephemeral=True
        )
        return

    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
    embed = discord.Embed(title="Transaction History", color=discord.Color.purple())
    for tx in transactions:
        timestamp = tx["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        embed.add_field(
            name=f"{tx['type'].capitalize()} {tx['shares']} {tx['ticker']}",
            value=f"Price: ${tx['price']:,.2f}\nTotal: ${tx['total']:,.2f}\nDate: {timestamp}",
            inline=False,
        )

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="leaderboard", description="Show the top users by net worth.")
async def leaderboard_command(interaction: discord.Interaction):
    users = bot.user_collection.find({})
    leaderboard = []
    async for user in users:
        portfolio_value = await calculate_portfolio_value(user["portfolio"])
        total_value = user["balance"] + portfolio_value
        leaderboard.append((user["_id"], total_value))

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    embed = discord.Embed(title="Leaderboard", color=discord.Color.gold())
    for rank, (user_id, net_worth) in enumerate(leaderboard[:10], start=1):
        embed.add_field(
            name=f"{rank}. <@{user_id}>",
            value=f"Net Worth: ${net_worth:,.2f}",
            inline=False,
        )

    await interaction.response.send_message(embed=embed)


@bot.tree.command(
    name="top", description="Show top 50 stocks by market cap (10 pages, 5 per page)"
)
async def top_command(interaction: discord.Interaction):
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


async def get_stock_history(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)

        period_map = {
            "1min": "1d",  # 1 minute data for 1 day
            "5min": "1d",  # 5 minute data for 1 day
            "1h": "1d",  # 1 hour data for 1 day
            "12h": "5d",  # hourly data for 5 days
            "1d": "1d",  # daily data for 1 day
            "5d": "5d",  # daily data for 5 days
            "1mo": "1mo",  # daily data for 1 month
            "1y": "1y",  # daily data for 1 year
            "ytd": "ytd",  # daily data year to date
            "5y": "5y",  # daily data for 5 years
            "max": "max",  # all available data
        }

        interval_map = {
            "1min": "1m",
            "5min": "5m",
            "1h": "1h",
            "12h": "1h",
            "1d": "1m",  # Changed to 1m to get more detailed data
            "5d": "1h",  # Changed to 1h for better resolution
            "1mo": "1d",
            "1y": "1d",
            "ytd": "1d",
            "5y": "1d",
            "max": "1d",
        }

        yf_period = period_map.get(period, "1d")
        yf_interval = interval_map.get(period, "1d")

        # Include pre/post market data
        history = stock.history(
            period=yf_period,
            interval=yf_interval,
            prepost=True,  # Include pre/post market data
        )

        if history.empty:
            return None

        return history
    except Exception as e:
        print(f"Error getting history for {ticker}: {e}")
        return None


@bot.tree.command(name="chart", description="Show price chart for a stock")
@app_commands.describe(
    ticker="Ticker symbol",
    period="Time period (1min,5min,1h,12h,1d,5d,1mo,1y,ytd,5y,max)",
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
    ]
)
async def chart_command(interaction: discord.Interaction, ticker: str, period: str):
    await interaction.response.defer()

    history = await get_stock_history(ticker.upper(), period)
    if history is None:
        await interaction.followup.send("Unable to fetch stock data.", ephemeral=True)
        return

    # Create the chart
    plt.figure(figsize=(10, 6))

    # Plot regular market hours in blue
    plt.plot(
        history.index, history["Close"], color="blue", label="Regular Hours", alpha=0.8
    )

    # If we have pre/post market data, it will be included in the same line
    # but we can highlight the current price point
    current_price = history["Close"].iloc[-1]
    plt.scatter(history.index[-1], current_price, color="green", s=100, zorder=5)

    plt.title(f"{ticker.upper()} Price Chart ({period})")
    plt.xlabel("Date/Time")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close()

    # Get current stock info for pre/post market data
    stock_info = await get_stock_info(ticker.upper())

    # Create embed with stock info
    price_change = history["Close"].iloc[-1] - history["Close"].iloc[0]
    price_change_pct = (price_change / history["Close"].iloc[0]) * 100

    embed = discord.Embed(
        title=f"{ticker.upper()} Price Chart", color=discord.Color.blue()
    )

    # Show current price with market status
    price_text = f"${current_price:,.2f} ({stock_info['market_status']})"
    embed.add_field(name="Current Price", value=price_text, inline=True)

    # Add pre/post market prices if available
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

    # Send the chart
    file = discord.File(buf, filename=f"{ticker}_chart.png")
    embed.set_image(url=f"attachment://{ticker}_chart.png")
    await interaction.followup.send(file=file, embed=embed)


def main():
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
