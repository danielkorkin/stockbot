import asyncio
import datetime
import io
import math
import os
import random
from typing import Literal, Optional

import discord
import matplotlib.dates as mdates

# For plotting
import matplotlib.pyplot as plt

# Motor (Async MongoDB driver)
import motor.motor_asyncio
import pandas as pd
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

###########################
# Configuration Constants #
###########################

OWNER_ID = int(os.getenv("OWNER_ID"))  # Default if not set
TEST_GUILD_ID = int(os.getenv("TEST_GUILD_ID"))  # Default if not set
MONGO_URI = os.getenv("MONGO_URI")
BOT_TOKEN = os.getenv("BOT_TOKEN")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Initial balance for every new user
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 1000.0))

# For demonstration, we run the price-updating task every 60 seconds.
PRICE_UPDATE_INTERVAL = int(os.getenv("PRICE_UPDATE_INTERVAL",60))

ENVIROMENT = os.getenv("ENVIROMENT", "development")

#################################
# Discord Bot (slash commands)  #
#################################

# Intents for the Bot
intents = discord.Intents.default()
intents.message_content = False  # We primarily use slash commands


def parse_time_period_to_timedelta(time_period: str) -> datetime.timedelta:
    """
    Convert a time period string like '1m', '1h', '1d', '5d', '10d', '1M' (1 month)
    into a proper timedelta. 1M is interpreted as 30 days here for simplicity.
    """
    time_period = time_period.lower().strip()
    if time_period == "1m":
        # 1 minute
        return datetime.timedelta(minutes=1)
    elif time_period == "1h":
        return datetime.timedelta(hours=1)
    elif time_period == "1d":
        return datetime.timedelta(days=1)
    elif time_period == "5d":
        return datetime.timedelta(days=5)
    elif time_period == "10d":
        return datetime.timedelta(days=10)
    elif (
        time_period == "1m" or time_period == "1mo"
    ):  # If you prefer to accept '1M' or '1mo'
        # This might conflict with '1m' for minute, so be careful.
        # We'll treat '1M' as 30 days for simplicity. If you want 1 minute,
        # you'd specify '1min' or something else. Let's use '1mo' for a month:
        return datetime.timedelta(days=30)
    else:
        # default fallback if user picks something unknown
        return datetime.timedelta(days=1)


class StockSimBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents, application_id=None)

        # Motor client
        self.motor_client = None
        self.db = None
        self.stock_collection = None
        self.user_collection = None
        self.events_collection = None
        self.earnings_collection = None
        self.price_history_collection = None

    async def setup_hook(self):
        # Register slash commands based on environment
        if ENVIROMENT.lower() == "development":
            # In development, only register to test guild
            self.tree.copy_global_to(guild=discord.Object(id=TEST_GUILD_ID))
            await self.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
            print("Slash commands synced to test guild.")
        else:
            # In production, register globally
            await self.tree.sync()
            print("Slash commands synced globally.")

        # Setup Mongo
        self.motor_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.motor_client[MONGO_DB_NAME]
        self.stock_collection = self.db["stocks"]
        self.user_collection = self.db["users"]
        self.events_collection = self.db["events"]
        self.earnings_collection = self.db["earnings_reports"]
        self.price_history_collection = self.db["stock_price_history"]

        # Start background price update task
        self.update_stock_prices.start()

    async def on_ready(self):
        print(f"Logged in as: {self.user} (ID: {self.user.id})")

    @tasks.loop(seconds=PRICE_UPDATE_INTERVAL)
    async def update_stock_prices(self):
        """Periodically update stock prices based on a simplified 'advanced' algorithm."""
        await self.update_prices_algorithm()

    async def update_prices_algorithm(self, triggered_by_event: bool = False):
        """
        Algorithm for updating stock prices, factoring in:
          - Volatility
          - Random movement
          - Events with 'impact' from -1.0 to +1.0
        """
        now = datetime.datetime.utcnow()
        one_day_ago = now - datetime.timedelta(days=1)

        # Fetch recent events from the last 24h
        recent_events = []
        async for evt in self.events_collection.find(
            {"timestamp": {"$gte": one_day_ago}}
        ):
            recent_events.append(evt)

        async for stock in self.stock_collection.find({}):
            ticker = stock["_id"]
            current_price = stock["price"]
            volatility = stock["volatility"]
            industry = stock["industry"]

            random_factor = random.uniform(-0.01, 0.01) * volatility

            # Sum all relevant event impacts
            event_factor = 0.0
            for evt in recent_events:
                impact_value = evt.get("impact", 0.0)
                targeted_tickers = evt.get("affected_tickers", [])
                targeted_industries = evt.get("affected_industries", [])

                applies_to_ticker = (
                    (ticker in targeted_tickers) if targeted_tickers else False
                )
                applies_to_industry = industry in targeted_industries
                if applies_to_ticker or applies_to_industry:
                    # Up to +/- 10% for impact of +/- 1.0
                    event_factor += impact_value * 0.10

            new_price = current_price * (1 + random_factor + event_factor)
            new_price = max(0.01, round(new_price, 2))

            # Update DB
            await self.stock_collection.update_one(
                {"_id": ticker}, {"$set": {"price": new_price}}
            )

            # Store price in history
            price_history_doc = {"ticker": ticker, "price": new_price, "timestamp": now}
            await self.price_history_collection.insert_one(price_history_doc)

    def is_owner_check(self, interaction: discord.Interaction):
        return interaction.user.id == OWNER_ID


bot = StockSimBot()

######################
#   Helper Methods   #
######################


async def get_user_data(user_collection, user_id: int):
    """
    Fetch user data from Mongo. If none exists, create a fresh record.
    """
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


def calculate_user_net_worth(user_data, stocks_dict):
    """
    Calculate user's net worth: balance + sum(shares * current_price).
    """
    total = user_data["balance"]
    for ticker, shares in user_data["portfolio"].items():
        if ticker in stocks_dict:
            total += stocks_dict[ticker]["price"] * shares
    return round(total, 2)


#########################
#   Slash Command Tree  #
#########################


@bot.tree.command(name="balance", description="Show your cash balance.")
async def balance_command(interaction: discord.Interaction):
    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    balance_amount = round(user_data["balance"], 2)
    await interaction.response.send_message(
        f"Your current balance: **${balance_amount}**"
    )


@bot.tree.command(
    name="portfolio", description="Show the portfolio of yourself or another user."
)
@app_commands.describe(
    member="Select a user to view their portfolio. Leave blank to view yours."
)
async def portfolio_command(
    interaction: discord.Interaction, member: Optional[discord.Member] = None
):
    target = member or interaction.user
    user_data = await get_user_data(bot.user_collection, target.id)

    embed = discord.Embed(
        title=f"{target.display_name}'s Portfolio", color=discord.Color.blue()
    )

    if not user_data["portfolio"]:
        embed.description = "No stocks held."
    else:
        # Gather current stock data
        all_stocks = {}
        async for stock in bot.stock_collection.find({}):
            all_stocks[stock["_id"]] = stock

        net_worth = calculate_user_net_worth(user_data, all_stocks)
        lines = []
        for ticker, shares in user_data["portfolio"].items():
            stock = all_stocks.get(ticker)
            if stock:
                price = stock["price"]
                value = round(price * shares, 2)
                lines.append(
                    f"{ticker}: {shares} share(s) @ ${price:.2f} (Value: ${value})"
                )

        embed.description = "\n".join(lines)
        embed.add_field(name="Net Worth", value=f"${net_worth}", inline=False)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(
    name="leaderboard", description="Show the top 10 players by net worth."
)
async def leaderboard_command(interaction: discord.Interaction):
    # Fetch all users and stocks
    all_users = bot.user_collection.find({})
    stocks_dict = {}
    async for stock in bot.stock_collection.find({}):
        stocks_dict[stock["_id"]] = stock

    leaderboard_data = []
    async for user_data in all_users:
        user_id = user_data["_id"]
        net_worth = calculate_user_net_worth(user_data, stocks_dict)
        leaderboard_data.append((user_id, net_worth))

    # Sort by net worth desc
    leaderboard_data.sort(key=lambda x: x[1], reverse=True)
    top_10 = leaderboard_data[:10]

    embed = discord.Embed(title="Leaderboard (Top 10)", color=discord.Color.gold())
    lines = []
    for idx, (uid, networth) in enumerate(top_10, start=1):
        # Tag the user by <@uid>
        lines.append(f"**{idx}.** <@{uid}> - ${networth}")

    embed.description = "\n".join(lines) if lines else "No data available."
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="buy", description="Buy shares of a stock.")
@app_commands.describe(
    ticker="Ticker symbol of the stock", amount="Number of shares to buy"
)
async def buy_command(interaction: discord.Interaction, ticker: str, amount: int):
    if amount <= 0:
        await interaction.response.send_message(
            "You must buy a positive number of shares.", ephemeral=True
        )
        return

    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message(
            "That stock does not exist.", ephemeral=True
        )
        return

    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    cost = stock_doc["price"] * amount

    if user_data["balance"] < cost:
        await interaction.response.send_message(
            f"Not enough balance to buy {amount} shares of {ticker}. (Cost: ${cost:.2f})",
            ephemeral=True,
        )
        return

    # Update portfolio
    portfolio = user_data["portfolio"]
    portfolio[ticker.upper()] = portfolio.get(ticker.upper(), 0) + amount
    new_balance = user_data["balance"] - cost

    transaction_record = {
        "type": "buy",
        "ticker": ticker.upper(),
        "shares": amount,
        "price": stock_doc["price"],
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": transaction_record},
        },
    )

    # Optional immediate price update
    await bot.update_prices_algorithm(triggered_by_event=True)

    await interaction.response.send_message(
        f"Bought **{amount}** share(s) of **{ticker.upper()}** @ **${stock_doc['price']}** each.\n"
        f"Total: **${cost:.2f}** | New balance: **${new_balance:.2f}**"
    )


@bot.tree.command(name="sell", description="Sell shares of a stock.")
@app_commands.describe(
    ticker="Ticker symbol of the stock", amount="Number of shares to sell"
)
async def sell_command(interaction: discord.Interaction, ticker: str, amount: int):
    if amount <= 0:
        await interaction.response.send_message(
            "You must sell a positive number of shares.", ephemeral=True
        )
        return

    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message(
            "That stock does not exist.", ephemeral=True
        )
        return

    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    portfolio = user_data["portfolio"]
    owned_shares = portfolio.get(ticker.upper(), 0)

    if amount > owned_shares:
        await interaction.response.send_message(
            f"You only own {owned_shares} share(s) of {ticker.upper()}.", ephemeral=True
        )
        return

    revenue = stock_doc["price"] * amount
    new_balance = user_data["balance"] + revenue
    updated_shares = owned_shares - amount

    if updated_shares <= 0:
        portfolio.pop(ticker.upper())
    else:
        portfolio[ticker.upper()] = updated_shares

    transaction_record = {
        "type": "sell",
        "ticker": ticker.upper(),
        "shares": amount,
        "price": stock_doc["price"],
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": transaction_record},
        },
    )

    # Optional immediate price update
    await bot.update_prices_algorithm(triggered_by_event=True)

    await interaction.response.send_message(
        f"Sold **{amount}** share(s) of **{ticker.upper()}** @ **${stock_doc['price']}** each.\n"
        f"Revenue: **${revenue:.2f}** | New balance: **${new_balance:.2f}**"
    )


@bot.tree.command(name="create_stock", description="Create a new stock (Owner Only).")
@app_commands.describe(
    name="Stock name (e.g. Apple Inc.)",
    ticker="Unique ticker symbol (e.g. AAPL)",
    price="Starting price per share",
    industry="Industry (e.g. Technology, Energy, Healthcare...)",
    volatility="Volatility (0.1 - 2.0, e.g.)",
    market_cap="Market Cap (optional)",
    dividend_yield="Dividend yield % (optional)",
    eps="Earnings per share (optional)",
    pe_ratio="P/E ratio (optional)",
    total_shares="Total shares (optional)",
)
async def create_stock_command(
    interaction: discord.Interaction,
    name: str,
    ticker: str,
    price: float,
    industry: str,
    volatility: float,
    market_cap: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    eps: Optional[float] = None,
    pe_ratio: Optional[float] = None,
    total_shares: Optional[int] = None,
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    existing = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if existing:
        await interaction.response.send_message(
            "Stock already exists with that ticker.", ephemeral=True
        )
        return

    if total_shares is None:
        total_shares = 1_000_000
    if market_cap is None:
        market_cap = round(price * total_shares, 2)
    if eps is None:
        eps = 0.0
    if pe_ratio is None and eps != 0:
        pe_ratio = round(price / eps, 2)
    elif pe_ratio is None:
        pe_ratio = 0.0
    if dividend_yield is None:
        dividend_yield = 0.0

    doc = {
        "_id": ticker.upper(),
        "name": name,
        "price": round(price, 2),
        "industry": industry,
        "volatility": volatility,
        "market_cap": market_cap,
        "dividend_yield": dividend_yield,
        "eps": eps,
        "pe_ratio": pe_ratio,
        "total_shares": total_shares,
    }

    await bot.stock_collection.insert_one(doc)
    await interaction.response.send_message(
        f"Created stock **{ticker.upper()}** at price ${price:.2f}."
    )


@bot.tree.command(
    name="update_stock", description="Update an existing stock's info (Owner Only)."
)
@app_commands.describe(
    ticker="Ticker to update",
    price="New price",
    industry="New industry",
    volatility="New volatility",
    market_cap="New market cap",
    dividend_yield="New dividend yield",
    eps="New EPS",
    pe_ratio="New P/E ratio",
    total_shares="New total shares",
)
async def update_stock_command(
    interaction: discord.Interaction,
    ticker: str,
    price: Optional[float] = None,
    industry: Optional[str] = None,
    volatility: Optional[float] = None,
    market_cap: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    eps: Optional[float] = None,
    pe_ratio: Optional[float] = None,
    total_shares: Optional[int] = None,
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("No such stock.", ephemeral=True)
        return

    update_data = {}
    if price is not None:
        update_data["price"] = round(price, 2)
    if industry is not None:
        update_data["industry"] = industry
    if volatility is not None:
        update_data["volatility"] = volatility
    if market_cap is not None:
        update_data["market_cap"] = market_cap
    if dividend_yield is not None:
        update_data["dividend_yield"] = dividend_yield
    if eps is not None:
        update_data["eps"] = eps
    if pe_ratio is not None:
        update_data["pe_ratio"] = pe_ratio
    if total_shares is not None:
        update_data["total_shares"] = total_shares

    if not update_data:
        await interaction.response.send_message("No changes provided.", ephemeral=True)
        return

    await bot.stock_collection.update_one(
        {"_id": ticker.upper()}, {"$set": update_data}
    )
    await interaction.response.send_message(f"Stock **{ticker.upper()}** updated.")


@bot.tree.command(name="stock", description="View a stock's current info.")
@app_commands.describe(ticker="Ticker symbol of the stock")
async def stock_command(interaction: discord.Interaction, ticker: str):
    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    embed = discord.Embed(
        title=f"{stock_doc['name']} ({stock_doc['_id']})", color=discord.Color.green()
    )
    embed.add_field(name="Price", value=f"${stock_doc['price']:.2f}", inline=True)
    embed.add_field(name="Industry", value=stock_doc["industry"], inline=True)
    embed.add_field(name="Volatility", value=str(stock_doc["volatility"]), inline=True)
    embed.add_field(name="Market Cap", value=str(stock_doc["market_cap"]), inline=True)
    embed.add_field(
        name="Dividend Yield", value=str(stock_doc["dividend_yield"]), inline=True
    )
    embed.add_field(name="EPS", value=str(stock_doc["eps"]), inline=True)
    embed.add_field(name="P/E Ratio", value=str(stock_doc["pe_ratio"]), inline=True)
    embed.add_field(
        name="Total Shares", value=str(stock_doc["total_shares"]), inline=True
    )

    await interaction.response.send_message(embed=embed)


#
# NEW COMMAND: /stock_chart
#
time_period_choices = Literal["1m", "1h", "1d", "5d", "10d", "1M"]


@bot.tree.command(
    name="stock_chart",
    description="View a stock's price chart for a given time period.",
)
@app_commands.describe(
    ticker="Ticker symbol of the stock",
    time_period="Choose: 1m (minute), 1h (hour), 1d (day), 5d, 10d, 1M (1 month).",
)
async def stock_chart_command(
    interaction: discord.Interaction, ticker: str, time_period: time_period_choices
):
    """
    Generate a chart for `ticker` covering the last `time_period`.
    """
    # Validate ticker
    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    # Convert time_period to a timedelta
    delta = parse_time_period_to_timedelta(time_period)
    now = datetime.datetime.utcnow()
    start_time = now - delta

    # Fetch historical data
    cursor = bot.price_history_collection.find(
        {"ticker": ticker.upper(), "timestamp": {"$gte": start_time}}
    ).sort("timestamp", 1)

    prices = []
    times = []
    async for doc in cursor:
        prices.append(doc["price"])
        times.append(doc["timestamp"])

    embed = discord.Embed(
        title=f"{stock_doc['name']} ({stock_doc['_id']}) - Last {time_period}",
        color=discord.Color.green(),
    )
    embed.add_field(
        name="Current Price", value=f"${stock_doc['price']:.2f}", inline=True
    )
    embed.add_field(name="Industry", value=stock_doc["industry"], inline=True)

    if not prices:
        embed.description = f"No price history found in the last {time_period}."
        await interaction.response.send_message(embed=embed)
        return

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(
        times, prices, marker="o", linestyle="-", color="blue", label=ticker.upper()
    )
    ax.set_title(f"{ticker.upper()} Price - Last {time_period}")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()

    # Convert figure to discord file
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    file = discord.File(buf, filename=f"{ticker.upper()}-{time_period}-chart.png")
    embed.set_image(url=f"attachment://{ticker.upper()}-{time_period}-chart.png")

    await interaction.response.send_message(embed=embed, file=file)


@bot.tree.command(name="market", description="Show all stocks categorized by industry.")
async def market_command(interaction: discord.Interaction):
    """
    Displays a list of all stocks, grouped by industry.
    """
    # We'll group stocks by industry
    cursor = bot.stock_collection.find({})
    industries_map = {}  # {industry: [(ticker, price), ...], ...}

    async for doc in cursor:
        industry = doc["industry"]
        ticker = doc["_id"]
        price = doc["price"]

        if industry not in industries_map:
            industries_map[industry] = []
        industries_map[industry].append((ticker, price))

    # Build an embed
    embed = discord.Embed(
        title="Market Overview",
        description="All stocks grouped by industry",
        color=discord.Color.blue(),
    )

    if not industries_map:
        embed.description = "No stocks found."
        await interaction.response.send_message(embed=embed)
        return

    for ind, stocks in industries_map.items():
        lines = []
        for ticker, price in stocks:
            lines.append(f"**{ticker}** - ${price:.2f}")
        field_value = "\n".join(lines)
        embed.add_field(name=ind, value=field_value, inline=False)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="news", description="Show recent published events.")
async def news_command(interaction: discord.Interaction):
    # Show the last 5 events
    cursor = bot.events_collection.find({}).sort("timestamp", -1).limit(5)
    events = []
    async for evt in cursor:
        events.append(evt)

    if not events:
        await interaction.response.send_message("No news at the moment.")
        return

    embed = discord.Embed(title="Recent News & Events", color=discord.Color.blue())
    desc_lines = []
    for evt in events:
        time_str = evt["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        impact = evt.get("impact", 0.0)
        affected_tickers = evt.get("affected_tickers", [])
        affected_industries = evt.get("affected_industries", [])

        desc_lines.append(
            f"**{evt['title']}**\n"
            f"Date: {time_str}\n"
            f"Affected Tickers: {', '.join(affected_tickers) if affected_tickers else 'None'}\n"
            f"Affected Industries: {', '.join(affected_industries) if affected_industries else 'None'}\n"
            f"Impact: {impact}  (range -1.0 to +1.0)\n"
            f"{evt['description']}\n"
            f"---"
        )

    embed.description = "\n".join(desc_lines)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(
    name="publish_event", description="Publish a custom news/event (Owner Only)."
)
@app_commands.describe(
    title="Title of the event",
    description="Description (use markdown if needed)",
    impact="Float from -1.0 (very negative) to +1.0 (very positive)",
    affected_tickers="Comma-separated list of ticker(s) (leave blank if none)",
    affected_industries="Comma-separated list of industries (e.g. Technology,Energy)",
)
async def publish_event_command(
    interaction: discord.Interaction,
    title: str,
    description: str,
    impact: float,
    affected_tickers: Optional[str] = "",
    affected_industries: Optional[str] = "",
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    tickers_list = (
        [t.strip().upper() for t in affected_tickers.split(",") if t.strip()]
        if affected_tickers
        else []
    )
    industries_list = (
        [i.strip() for i in affected_industries.split(",") if i.strip()]
        if affected_industries
        else []
    )

    doc = {
        "title": title,
        "description": description,
        "impact": impact,  # -1.0 to +1.0
        "affected_tickers": tickers_list,
        "affected_industries": industries_list,
        "timestamp": datetime.datetime.utcnow(),
    }

    await bot.events_collection.insert_one(doc)
    # Trigger immediate price update
    await bot.update_prices_algorithm(triggered_by_event=True)

    await interaction.response.send_message("Event published and prices updated.")


@bot.tree.command(
    name="earnings_report", description="Publish an earnings report (Owner Only)."
)
@app_commands.describe(
    ticker="Which stock's earnings to publish",
    markdown_text="Markdown text for the earnings report",
)
async def earnings_report_command(
    interaction: discord.Interaction, ticker: str, markdown_text: str
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    entry = {
        "ticker": ticker.upper(),
        "timestamp": datetime.datetime.utcnow(),
        "report": markdown_text,
    }
    await bot.earnings_collection.insert_one(entry)

    await interaction.response.send_message(
        f"Earnings report for {ticker.upper()} published."
    )


@bot.tree.command(
    name="history", description="Show transaction history for yourself or another user."
)
@app_commands.describe(
    member="The user to show history for (leave blank for yourself)."
)
async def history_command(
    interaction: discord.Interaction, member: Optional[discord.Member] = None
):
    target = member or interaction.user
    user_data = await get_user_data(bot.user_collection, target.id)

    transactions = user_data["transactions"]
    if not transactions:
        await interaction.response.send_message(
            f"No transaction history for {target.display_name}."
        )
        return

    # Sort newest first
    transactions.sort(key=lambda x: x["timestamp"], reverse=True)

    embed = discord.Embed(
        title=f"{target.display_name}'s Transaction History",
        color=discord.Color.purple(),
    )
    lines = []
    for tx in transactions[:10]:  # last 10
        ts_str = tx["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(
            f"**{tx['type'].upper()}** {tx['shares']} of {tx['ticker']} @ ${tx['price']:.2f}\n"
            f"Date: {ts_str}"
        )
    embed.description = "\n".join(lines) if lines else "No transactions."

    await interaction.response.send_message(embed=embed)


########################
# Run the Bot
########################


def main():
    # You can also load token from environment variables or a config file
    bot.run(BOT_TOKEN)


if __name__ "__main__":
    main()
