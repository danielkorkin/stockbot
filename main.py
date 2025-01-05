import asyncio
import datetime
import io
import math
import os
import random
from typing import Dict, List, Literal, Optional

import discord
import matplotlib.dates as mdates
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

OWNER_ID = int(os.getenv("OWNER_ID", "123456789012345678"))  # fallback
TEST_GUILD_ID = int(os.getenv("TEST_GUILD_ID", "123456789012345678"))  # fallback
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "StockBotDB")

# Initial balance for every new user
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 1000.0))

# For demonstration, we run the price-updating task every 60 seconds.
PRICE_UPDATE_INTERVAL = int(os.getenv("PRICE_UPDATE_INTERVAL", 60))

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

#################################
# Discord Bot (slash commands)  #
#################################

intents = discord.Intents.default()
intents.message_content = False  # We primarily use slash commands


def parse_time_period_to_timedelta(time_period: str) -> datetime.timedelta:
    """
    Convert a time period string like '1m', '1h', '1d', '5d', '10d', '1M'
    into a proper timedelta. We'll interpret 1M (1 month) as 30 days for simplicity.
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
    elif time_period in (
        "1m",
        "1mo",
    ):  # possible conflict with 1 minute, but let's do 1 month
        return datetime.timedelta(days=30)
    else:
        return datetime.timedelta(days=1)


class PaginatorView(discord.ui.View):
    """
    Generic Paginator for slash commands using a View with Next/Prev/Stop buttons.
    """

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
        if max_pages is not None:
            self.total_pages = min(total_full_pages, max_pages)
        else:
            self.total_pages = total_full_pages

    async def send_first_page(self, interaction: discord.Interaction):
        embed = self.build_embed()
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


class StockSimBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents, application_id=None)
        self.motor_client = None
        self.db = None

        # Collections
        self.stock_collection = None
        self.user_collection = None
        self.events_collection = None
        self.earnings_collection = None
        self.price_history_collection = None
        self.alerts_collection = None  # new alerts collection

    async def setup_hook(self):
        try:
            if ENVIRONMENT.lower() == "development":
                self.tree.copy_global_to(guild=discord.Object(id=TEST_GUILD_ID))
                await self.tree.sync(guild=discord.Object(id=TEST_GUILD_ID))
                print("Slash commands synced to test guild.")
            else:
                await self.tree.sync()
                print("Slash commands synced globally.")
        except Exception as e:
            print(f"Error syncing commands: {e}")

        # Setup Mongo
        self.motor_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.motor_client[MONGO_DB_NAME]

        self.stock_collection = self.db["stocks"]
        self.user_collection = self.db["users"]
        self.events_collection = self.db["events"]
        self.earnings_collection = self.db["earnings_reports"]
        self.price_history_collection = self.db["stock_price_history"]
        self.alerts_collection = self.db["alerts"]  # new

        # Start background price update task
        self.update_stock_prices.start()

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")

    @tasks.loop(seconds=PRICE_UPDATE_INTERVAL)
    async def update_stock_prices(self):
        await self.update_prices_algorithm()

    async def update_prices_algorithm(self, triggered_by_event: bool = False):
        """
        Extended price update logic:
          1) Filter out or ignore expired events
          2) For each stock, skip if plateau == True
          3) If target_price is set, gradually move stock toward it
          4) Otherwise do normal random + event-based movement
          5) Check alerts
        """
        now = datetime.datetime.utcnow()

        # 1) Identify valid events (not expired)
        valid_events = []
        async for evt in self.events_collection.find({}):
            timestamp = evt["timestamp"]
            impact = abs(evt.get("impact", 0.0))
            if impact > 1.0:
                impact = 1.0
            hours_duration = 24.0 * impact  # max 24h if impact=1.0
            expiration = timestamp + datetime.timedelta(hours=hours_duration)
            if now < expiration:
                valid_events.append(evt)

        # 2) For each stock
        async for stock in self.stock_collection.find({}):
            ticker = stock["_id"]
            current_price = stock["price"]
            volatility = stock["volatility"]
            industry = stock["industry"]
            plateau = stock.get("plateau", False)
            target_info = stock.get("target_price_info", None)

            if plateau:
                # skip normal price changes
                continue

            if target_info:
                # 3) Gradually move price
                t_price = target_info["target_price"]
                finish_time = target_info["finish_time"]
                start_time = target_info["start_time"]
                start_price = target_info["start_price"]
                if now >= finish_time:
                    # finalize
                    new_price = t_price
                    await self.stock_collection.update_one(
                        {"_id": ticker},
                        {
                            "$unset": {"target_price_info": ""},
                            "$set": {"price": new_price},
                        },
                    )
                else:
                    total_duration = (finish_time - start_time).total_seconds()
                    elapsed = (now - start_time).total_seconds()
                    ratio = min(elapsed / total_duration, 1.0)
                    new_price = start_price + (t_price - start_price) * ratio
                    new_price = round(new_price, 2)

                    await self.stock_collection.update_one(
                        {"_id": ticker}, {"$set": {"price": new_price}}
                    )

                # store price in history
                price_history_doc = {
                    "ticker": ticker,
                    "price": new_price,
                    "timestamp": now,
                }
                await self.price_history_collection.insert_one(price_history_doc)
                continue

            # 4) Normal random + event-based movement
            random_factor = random.uniform(-0.01, 0.01) * volatility
            event_factor = 0.0
            for evt in valid_events:
                impact_val = evt.get("impact", 0.0)
                t_tickers = evt.get("affected_tickers", [])
                t_inds = evt.get("affected_industries", [])
                applies_ticker = (ticker in t_tickers) if t_tickers else False
                applies_industry = industry in t_inds
                if applies_ticker or applies_industry:
                    event_factor += impact_val * 0.10

            new_price = current_price * (1 + random_factor + event_factor)
            new_price = max(0.01, round(new_price, 2))
            await self.stock_collection.update_one(
                {"_id": ticker}, {"$set": {"price": new_price}}
            )

            # store price in history
            price_history_doc = {"ticker": ticker, "price": new_price, "timestamp": now}
            await self.price_history_collection.insert_one(price_history_doc)

        # 5) Check alerts
        await self.check_alerts()

    async def check_alerts(self):
        alerts = self.alerts_collection.find({})
        alerts_map = {}  # ticker -> list of alert docs
        async for alert in alerts:
            tk = alert["ticker"]
            if tk not in alerts_map:
                alerts_map[tk] = []
            alerts_map[tk].append(alert)

        for ticker, alert_list in alerts_map.items():
            stock_doc = await self.stock_collection.find_one({"_id": ticker})
            if not stock_doc:
                continue
            current_price = stock_doc["price"]

            for alert in alert_list:
                user_id = alert["user_id"]
                channel_id = alert["channel_id"]
                t_price = alert["target_price"]
                alert_id = alert["_id"]

                triggered = False
                # If current_price >= t_price, or current_price <= t_price (depending on t_price sign)
                # We'll just check if we've "crossed" it in either direction:
                # If t_price > current_price, then triggered if current_price >= t_price
                # If t_price < current_price, then triggered if current_price <= t_price
                # This is a simple approach; can be refined.

                if current_price >= t_price and t_price >= 0:
                    triggered = True
                elif current_price <= t_price and t_price > current_price:
                    triggered = True

                # Alternate check:
                #   if (current_price >= t_price and current_price - t_price >= 0)
                #   or (current_price <= t_price and t_price - current_price >= 0)
                # but let's keep it simple.

                if triggered:
                    await self.alerts_collection.delete_one({"_id": alert_id})
                    channel = self.get_channel(channel_id)
                    if channel:
                        user_mention = f"<@{user_id}>"
                        await channel.send(
                            f"{user_mention}, alert triggered for **{ticker}**!\n"
                            f"Current Price: `${current_price}` reached target `${t_price}`."
                        )


######################
#   Helper Methods   #
######################


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


def calculate_user_net_worth(user_data, stocks_dict):
    total = user_data["balance"]
    for ticker, shares in user_data["portfolio"].items():
        if ticker in stocks_dict:
            total += stocks_dict[ticker]["price"] * shares
    return round(total, 2)


bot = StockSimBot()

#########################
#       Commands        #
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


################################
# PAGINATED /LEADERBOARD (max 10 pages, 5 users each)
################################
@bot.tree.command(
    name="leaderboard", description="Show the top players by net worth (paginated)."
)
async def leaderboard_command(interaction: discord.Interaction):
    all_users = bot.user_collection.find({})
    stocks_dict = {}
    async for stock in bot.stock_collection.find({}):
        stocks_dict[stock["_id"]] = stock

    lb_data = []
    async for user_data in all_users:
        uid = user_data["_id"]
        net_worth = calculate_user_net_worth(user_data, stocks_dict)
        lb_data.append((uid, net_worth))

    lb_data.sort(key=lambda x: x[1], reverse=True)

    def embed_factory(
        page_idx: int, total_pages: int, items_slice: list
    ) -> discord.Embed:
        embed = discord.Embed(
            title=f"Leaderboard - Page {page_idx+1}/{total_pages}",
            color=discord.Color.gold(),
        )
        lines = []
        start_rank = page_idx * 5 + 1
        for i, (user_id, netw) in enumerate(items_slice, start=start_rank):
            lines.append(f"**{i}.** <@{user_id}> - ${netw}")
        embed.description = "\n".join(lines) if lines else "No data"
        return embed

    if not lb_data:
        await interaction.response.send_message("No data available.")
        return

    view = PaginatorView(
        items=lb_data,
        items_per_page=5,
        embed_factory=embed_factory,
        author_id=interaction.user.id,
        max_pages=10,
    )
    await view.send_first_page(interaction)


################################
# PAGINATED /MARKET (1 industry per page)
################################
@bot.tree.command(
    name="market", description="Show the market by industry, one page per industry."
)
async def market_command(interaction: discord.Interaction):
    cursor = bot.stock_collection.find({})
    industries_map: Dict[str, List[dict]] = {}

    async for doc in cursor:
        industry = doc["industry"]
        ticker = doc["_id"]
        price = doc["price"]
        if industry not in industries_map:
            industries_map[industry] = []
        industries_map[industry].append({"ticker": ticker, "price": price})

    pages = []
    for ind, stocks_list in industries_map.items():
        stocks_list.sort(key=lambda x: x["ticker"])
        pages.append((ind, stocks_list))
    pages.sort(key=lambda x: x[0])

    if not pages:
        await interaction.response.send_message("No stocks found.")
        return

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        (industry, slist) = subset[0]  # 1 item per page
        embed = discord.Embed(
            title=f"Market Overview - Page {page_idx+1}/{total_pages}",
            description=f"Industry: **{industry}**",
            color=discord.Color.blue(),
        )
        lines = []
        for s in slist:
            lines.append(f"**{s['ticker']}** - ${s['price']:.2f}")
        if lines:
            embed.add_field(
                name="Stocks in this Industry", value="\n".join(lines), inline=False
            )
        else:
            embed.add_field(
                name="Stocks in this Industry", value="No stocks here.", inline=False
            )
        return embed

    view = PaginatorView(
        items=pages,
        items_per_page=1,
        embed_factory=embed_factory,
        author_id=interaction.user.id,
    )
    await view.send_first_page(interaction)


################################
# PAGINATED /NEWS (1 news item/page, newest first -> page 1)
################################
@bot.tree.command(
    name="news",
    description="Show all published events (1 item per page, newest first).",
)
async def news_command(interaction: discord.Interaction):
    cursor = bot.events_collection.find({}).sort("timestamp", -1)
    events = []
    async for doc in cursor:
        events.append(doc)

    if not events:
        await interaction.response.send_message("No news at the moment.")
        return

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        evt = subset[0]
        time_str = evt["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        impact = evt.get("impact", 0.0)
        t_tickers = evt.get("affected_tickers", [])
        t_inds = evt.get("affected_industries", [])

        desc = (
            f"**Date:** {time_str}\n"
            f"**Impact:** {impact}  (range -1.0 to +1.0)\n"
            f"**Affected Tickers:** {', '.join(t_tickers) if t_tickers else 'None'}\n"
            f"**Affected Industries:** {', '.join(t_inds) if t_inds else 'None'}\n\n"
            f"{evt['description']}"
        )
        embed = discord.Embed(
            title=f"{evt['title']} (Page {page_idx+1}/{total_pages})",
            description=desc,
            color=discord.Color.blue(),
        )
        return embed

    view = PaginatorView(
        items=events,
        items_per_page=1,
        embed_factory=embed_factory,
        author_id=interaction.user.id,
    )
    await view.send_first_page(interaction)


################################
# PAGINATED /HISTORY (10 transactions/page, newest first)
################################
@bot.tree.command(
    name="history",
    description="Show transaction history for yourself or another user (paginated).",
)
@app_commands.describe(member="The user to show history for. Leave blank for yourself.")
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

    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
    hist_data = []
    for tx in transactions:
        ts_str = tx["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"**{tx['type'].upper()}** {tx['shares']} of {tx['ticker']} @ ${tx['price']:.2f}\nDate: {ts_str}"
        hist_data.append(line)

    def embed_factory(page_idx: int, total_pages: int, subset: list) -> discord.Embed:
        embed = discord.Embed(
            title=f"{target.display_name}'s Transaction History (Page {page_idx+1}/{total_pages})",
            color=discord.Color.purple(),
        )
        embed.description = "\n\n".join(subset)
        return embed

    view = PaginatorView(
        items=hist_data,
        items_per_page=10,
        embed_factory=embed_factory,
        author_id=interaction.user.id,
    )
    await view.send_first_page(interaction)


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

    # show plateau or target info if set
    plateau = stock_doc.get("plateau", False)
    if plateau:
        embed.add_field(
            name="Plateau", value="**ON** (price changes frozen)", inline=False
        )

    tpi = stock_doc.get("target_price_info", None)
    if tpi:
        embed.add_field(
            name="Target Price",
            value=(
                f"Moving from ${tpi['start_price']:.2f} to "
                f"${tpi['target_price']:.2f} by "
                f"{tpi['finish_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
            ),
            inline=False,
        )

    await interaction.response.send_message(embed=embed)


time_period_choices = Literal["1m", "1h", "1d", "5d", "10d", "1M"]


@bot.tree.command(
    name="stock_chart",
    description="View a stock's price chart for a given time period.",
)
@app_commands.describe(
    ticker="Ticker symbol", time_period="Choose: 1m, 1h, 1d, 5d, 10d, 1M."
)
async def stock_chart_command(
    interaction: discord.Interaction, ticker: str, time_period: time_period_choices
):
    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    delta = parse_time_period_to_timedelta(time_period)
    now = datetime.datetime.utcnow()
    start_time = now - delta
    cursor = bot.price_history_collection.find(
        {"ticker": ticker.upper(), "timestamp": {"$gte": start_time}}
    ).sort("timestamp", 1)

    prices, times = [], []
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

    fig, ax = plt.subplots()
    ax.plot(
        times, prices, marker="o", linestyle="-", color="blue", label=ticker.upper()
    )
    ax.set_title(f"{ticker.upper()} Price - Last {time_period}")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    file = discord.File(buf, filename=f"{ticker.upper()}-{time_period}-chart.png")
    embed.set_image(url=f"attachment://{ticker.upper()}-{time_period}-chart.png")

    await interaction.response.send_message(embed=embed, file=file)


##################################################
#   New Commands: alerts, plateau, set_target_price
#   + existing create_stock, update_stock, publish_event, earnings_report
##################################################


@bot.tree.command(name="alert", description="Set a price alert for a specific stock.")
@app_commands.describe(
    ticker="Ticker symbol", target_price="Price at which you want to be alerted"
)
async def alert_command(
    interaction: discord.Interaction, ticker: str, target_price: float
):
    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    alert_doc = {
        "user_id": interaction.user.id,
        "channel_id": interaction.channel_id,
        "ticker": ticker.upper(),
        "target_price": target_price,
    }
    await bot.alerts_collection.insert_one(alert_doc)

    await interaction.response.send_message(
        f"Alert set: {ticker.upper()} @ ${target_price:.2f}\n"
        f"I'll ping you here when this threshold is reached or exceeded."
    )


@bot.tree.command(
    name="plateau",
    description="Toggle plateau (freeze) on/off for a stock (Owner Only).",
)
@app_commands.describe(ticker="Which stock to freeze/unfreeze", toggle="on or off")
async def plateau_command(
    interaction: discord.Interaction, ticker: str, toggle: Literal["on", "off"]
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

    new_val = toggle == "on"
    await bot.stock_collection.update_one(
        {"_id": ticker.upper()}, {"$set": {"plateau": new_val}}
    )

    msg = (
        f"Stock {ticker.upper()} is now plateaued (frozen)."
        if new_val
        else f"Stock {ticker.upper()} is no longer plateaued."
    )
    await interaction.response.send_message(msg)


@bot.tree.command(
    name="set_target_price",
    description="Gradually move a stock's price to a target (Owner Only).",
)
@app_commands.describe(
    ticker="Which stock",
    target_price="Desired final price",
    duration_minutes="How many minutes until the price is fully at the target",
)
async def set_target_price_command(
    interaction: discord.Interaction,
    ticker: str,
    target_price: float,
    duration_minutes: int,
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    if duration_minutes <= 0:
        await interaction.response.send_message(
            "Duration must be positive.", ephemeral=True
        )
        return

    stock_doc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not stock_doc:
        await interaction.response.send_message("Stock not found.", ephemeral=True)
        return

    current_price = stock_doc["price"]
    start_time = datetime.datetime.utcnow()
    finish_time = start_time + datetime.timedelta(minutes=duration_minutes)

    target_price_info = {
        "target_price": round(target_price, 2),
        "start_time": start_time,
        "finish_time": finish_time,
        "start_price": current_price,
    }

    await bot.stock_collection.update_one(
        {"_id": ticker.upper()}, {"$set": {"target_price_info": target_price_info}}
    )

    await interaction.response.send_message(
        f"Over the next {duration_minutes} minute(s), {ticker.upper()} will gradually move from "
        f"${current_price:.2f} to ${target_price:.2f}."
    )


@bot.tree.command(name="create_stock", description="Create a new stock (Owner Only).")
@app_commands.describe(
    name="Stock name",
    ticker="Ticker symbol",
    price="Starting price",
    industry="Industry",
    volatility="Volatility (0.1 - 2.0)",
    market_cap="Optional market cap",
    dividend_yield="Optional dividend yield",
    eps="Optional EPS",
    pe_ratio="Optional P/E ratio",
    total_shares="Optional total shares",
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
            "A stock with that ticker already exists.", ephemeral=True
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
        f"Created stock **{ticker.upper()}** @ ${price:.2f}."
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

    sdoc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not sdoc:
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
        await interaction.response.send_message("No fields to update.", ephemeral=True)
        return

    await bot.stock_collection.update_one(
        {"_id": ticker.upper()}, {"$set": update_data}
    )
    await interaction.response.send_message(
        f"Stock **{ticker.upper()}** updated successfully."
    )


@bot.tree.command(
    name="publish_event", description="Publish a custom news/event (Owner Only)."
)
@app_commands.describe(
    title="Title of the event",
    description="Event description",
    impact="Float from -1.0 to +1.0",
    affected_tickers="Comma-separated ticker(s)",
    affected_industries="Comma-separated industry names",
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

    evt_doc = {
        "title": title,
        "description": description,
        "impact": impact,
        "affected_tickers": tickers_list,
        "affected_industries": industries_list,
        "timestamp": datetime.datetime.utcnow(),
    }
    await bot.events_collection.insert_one(evt_doc)
    await bot.update_prices_algorithm(triggered_by_event=True)

    await interaction.response.send_message("Event published and prices updated.")


@bot.tree.command(
    name="earnings_report", description="Publish an earnings report (Owner Only)."
)
@app_commands.describe(
    ticker="Which stock's earnings to publish",
    markdown_text="The text of the earnings report in markdown",
)
async def earnings_report_command(
    interaction: discord.Interaction, ticker: str, markdown_text: str
):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    sdoc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not sdoc:
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


##################################################
#   Buy/Sell Commands
##################################################


@bot.tree.command(name="buy", description="Buy shares of a stock.")
@app_commands.describe(ticker="Ticker symbol", amount="Number of shares")
async def buy_command(interaction: discord.Interaction, ticker: str, amount: int):
    if amount <= 0:
        await interaction.response.send_message(
            "You must buy a positive number of shares.", ephemeral=True
        )
        return

    sdoc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not sdoc:
        await interaction.response.send_message(
            "That stock does not exist.", ephemeral=True
        )
        return

    user_data = await get_user_data(bot.user_collection, interaction.user.id)
    cost = sdoc["price"] * amount
    if user_data["balance"] < cost:
        await interaction.response.send_message(
            f"Not enough balance to buy {amount} share(s) of {ticker}. (Cost: ${cost:.2f})",
            ephemeral=True,
        )
        return

    portfolio = user_data["portfolio"]
    portfolio[ticker.upper()] = portfolio.get(ticker.upper(), 0) + amount
    new_balance = user_data["balance"] - cost

    tx_record = {
        "type": "buy",
        "ticker": ticker.upper(),
        "shares": amount,
        "price": sdoc["price"],
        "timestamp": datetime.datetime.utcnow(),
    }
    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": tx_record},
        },
    )

    await interaction.response.send_message(
        f"Bought **{amount}** share(s) of **{ticker.upper()}** @ ${sdoc['price']:.2f} each.\n"
        f"Total: ${cost:.2f} | New balance: ${new_balance:.2f}"
    )


@bot.tree.command(name="sell", description="Sell shares of a stock.")
@app_commands.describe(ticker="Ticker symbol", amount="Number of shares to sell")
async def sell_command(interaction: discord.Interaction, ticker: str, amount: int):
    if amount <= 0:
        await interaction.response.send_message(
            "You must sell a positive number of shares.", ephemeral=True
        )
        return

    sdoc = await bot.stock_collection.find_one({"_id": ticker.upper()})
    if not sdoc:
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

    revenue = sdoc["price"] * amount
    new_balance = user_data["balance"] + revenue
    updated_shares = owned_shares - amount

    if updated_shares <= 0:
        portfolio.pop(ticker.upper(), None)
    else:
        portfolio[ticker.upper()] = updated_shares

    tx_record = {
        "type": "sell",
        "ticker": ticker.upper(),
        "shares": amount,
        "price": sdoc["price"],
        "timestamp": datetime.datetime.utcnow(),
    }
    await bot.user_collection.update_one(
        {"_id": interaction.user.id},
        {
            "$set": {"balance": new_balance, "portfolio": portfolio},
            "$push": {"transactions": tx_record},
        },
    )

    await interaction.response.send_message(
        f"Sold **{amount}** share(s) of **{ticker.upper()}** @ ${sdoc['price']:.2f} each.\n"
        f"Revenue: ${revenue:.2f} | New balance: ${new_balance:.2f}"
    )


def main():
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
