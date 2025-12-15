#!/usr/bin/env python3
"""
Query trending tokens on Base network from GeckoTerminal
Shows price change percentages across all timeframes
"""

import requests


def get_trending_base_tokens():
    """Get trending pools on Base from GeckoTerminal"""
    url = 'https://api.geckoterminal.com/api/v2/networks/base/trending_pools'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        print(f"Error fetching trending pools: {e}")
        return []


def format_percentage(num):
    """Format percentage with sign"""
    if num is None:
        return "    N/A"
    num = float(num)
    sign = "+" if num > 0 else ""
    return f"{sign}{num:6.1f}%"


def main():
    print("🔥 TRENDING TOKENS ON BASE - PRICE CHANGES\n")
    print("="*120)

    pools = get_trending_base_tokens()

    if not pools:
        print("No trending pools found")
        return

    # Header
    print(f"{'#':<3} {'Token':<30} {'5m':>9} {'15m':>9} {'30m':>9} {'1h':>9} {'6h':>9} {'24h':>9}")
    print("-"*120)

    for i, pool in enumerate(pools, 1):
        attrs = pool.get('attributes', {})
        name = attrs.get('name', 'N/A')

        # Get price change percentages
        price_change = attrs.get('price_change_percentage', {}) or {}
        m5 = price_change.get('m5')
        m15 = price_change.get('m15')
        m30 = price_change.get('m30')
        h1 = price_change.get('h1')
        h6 = price_change.get('h6')
        h24 = price_change.get('h24')

        print(f"{i:<3} {name:<30} {format_percentage(m5)} {format_percentage(m15)} {format_percentage(m30)} {format_percentage(h1)} {format_percentage(h6)} {format_percentage(h24)}")

    print("="*120)


if __name__ == "__main__":
    main()
