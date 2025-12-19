#!/usr/bin/env python3
"""
Query BURGERS memecoin info on Base network
Fetches price/market data and trading info
"""

from datetime import datetime

import requests

# Base network chain ID
BASE_CHAIN_ID = 8453


def search_token_on_dexscreener(ticker):
    """Search for token by ticker on DexScreener"""
    url = f"https://api.dexscreener.com/latest/dex/search/?q={ticker}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Filter for Base chain
        base_pairs = [pair for pair in data.get("pairs", []) if pair.get("chainId") == "base"]

        return base_pairs
    except Exception as e:
        print(f"Error searching DexScreener: {e}")
        return []


def get_token_info(contract_address):
    """Get detailed token info from DexScreener"""
    url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching token info: {e}")
        return None


def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num is None:
        return "N/A"

    num = float(num)
    if num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:.2f}"


def format_percentage(num):
    """Format percentage with color indicators"""
    if num is None:
        return "N/A"

    num = float(num)
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.2f}%"


def display_price_and_market_data(pair):
    """Display price and market data"""
    print("\n" + "=" * 60)
    print("💰 PRICE & MARKET DATA")
    print("=" * 60)

    print(f"Token: {pair.get('baseToken', {}).get('name', 'N/A')} ({pair.get('baseToken', {}).get('symbol', 'N/A')})")
    print(f"Contract: {pair.get('baseToken', {}).get('address', 'N/A')}")
    print(f"\nPrice (USD): ${float(pair.get('priceUsd', 0)):.8f}")
    print(f"Price (Native): {float(pair.get('priceNative', 0)):.10f} ETH")

    print(f"\nMarket Cap: {format_number(pair.get('fdv'))}")

    # Volume history across all timeframes
    volume = pair.get("volume", {})
    print("\nVolume History:")
    print(f"  5m:  {format_number(volume.get('m5'))}")
    print(f"  1h:  {format_number(volume.get('h1'))}")
    print(f"  6h:  {format_number(volume.get('h6'))}")
    print(f"  24h: {format_number(volume.get('h24'))}")

    # Price change history
    price_change = pair.get("priceChange", {})
    print("\nPrice Change History:")
    print(f"  5m:  {format_percentage(price_change.get('m5'))}")
    print(f"  1h:  {format_percentage(price_change.get('h1'))}")
    print(f"  6h:  {format_percentage(price_change.get('h6'))}")
    print(f"  24h: {format_percentage(price_change.get('h24'))}")


def display_trading_info(pair):
    """Display trading and liquidity info"""
    print("\n" + "=" * 60)
    print("📊 TRADING INFO")
    print("=" * 60)

    dex_id = pair.get("dexId", "N/A")
    print(f"DEX: {dex_id}")
    print(f"Pair Address: {pair.get('pairAddress', 'N/A')}")
    print(f"Trading Pair: {pair.get('baseToken', {}).get('symbol')}/{pair.get('quoteToken', {}).get('symbol')}")

    # Liquidity details
    liquidity = pair.get("liquidity", {})
    print("\nLiquidity:")
    print(f"  USD:   {format_number(liquidity.get('usd'))}")
    print(f"  Base:  {format_number(liquidity.get('base'))} {pair.get('baseToken', {}).get('symbol')}")
    print(f"  Quote: {format_number(liquidity.get('quote'))} {pair.get('quoteToken', {}).get('symbol')}")

    # Transaction history across all timeframes
    txns = pair.get("txns", {})
    print("\nTransaction History:")

    for timeframe in ["m5", "h1", "h6", "h24"]:
        txn_data = txns.get(timeframe, {})
        buys = txn_data.get("buys", 0)
        sells = txn_data.get("sells", 0)
        total = buys + sells
        label = {"m5": "5m", "h1": "1h", "h6": "6h", "h24": "24h"}[timeframe]
        print(f"  {label:3s}: {buys:4d} buys | {sells:4d} sells | {total:4d} total")

    # Links
    print("\n🔗 Links:")
    if pair.get("url"):
        print(f"  DexScreener: {pair.get('url')}")

    info = pair.get("info", {})
    if info.get("websites"):
        print(f"  Website: {info['websites'][0]['url']}")
    if info.get("socials"):
        for social in info["socials"]:
            print(f"  {social['type'].title()}: {social['url']}")


def main():
    print("🍔 Querying BURGERS token on Base network...")

    # Search for BURGERS token
    pairs = search_token_on_dexscreener("BURGERS")

    if not pairs:
        print("❌ No BURGERS token found on Base network")
        return

    print(f"✅ Found {len(pairs)} trading pair(s) for BURGERS on Base")

    # Use the pair with highest liquidity
    main_pair = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0)))

    # Display data
    display_price_and_market_data(main_pair)
    display_trading_info(main_pair)

    # Show other pairs if available
    if len(pairs) > 1:
        print("\n" + "=" * 60)
        print(f"📋 OTHER TRADING PAIRS ({len(pairs) - 1})")
        print("=" * 60)
        for i, pair in enumerate(pairs[1:], 1):
            dex = pair.get("dexId", "N/A")
            quote = pair.get("quoteToken", {}).get("symbol", "N/A")
            liq = format_number(pair.get("liquidity", {}).get("usd"))
            print(f"{i}. {dex} - BURGERS/{quote} - Liquidity: {liq}")

    print("\n" + "=" * 60)
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
