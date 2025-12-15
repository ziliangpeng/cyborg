#!/usr/bin/env python3
"""
Monitor trending Base tokens and alert on >5% price pumps in 5m/15m/30m windows
Fetches data every minute and prints only tokens that are pumping
"""

import argparse
import requests
import time
from datetime import datetime


def get_trending_base_tokens():
    """Get trending pools on Base from GeckoTerminal"""
    url = 'https://api.geckoterminal.com/api/v2/networks/base/trending_pools'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        print(f"❌ Error fetching trending pools: {e}")
        return []


def check_for_pumps(pools):
    """Check pools for >5% gains in 5m/15m/30m and return pumping tokens"""
    pumping = []

    for pool in pools:
        attrs = pool.get('attributes', {})
        name = attrs.get('name', 'N/A')
        price_change = attrs.get('price_change_percentage', {}) or {}

        m5 = float(price_change.get('m5') or 0)
        m15 = float(price_change.get('m15') or 0)
        m30 = float(price_change.get('m30') or 0)

        # Check if any timeframe is pumping >5%
        if m5 > 5 or m15 > 5 or m30 > 5:
            # Get volume and transaction data
            volume = attrs.get('volume_usd', {}) or {}
            txns = attrs.get('transactions', {}) or {}
            h1 = float(price_change.get('h1') or 0)

            vol_h1 = volume.get('h1', 0) or 0
            h1_txns = txns.get('h1', {}) or {}
            buys = h1_txns.get('buys', 0) or 0
            sells = h1_txns.get('sells', 0) or 0

            pumping.append({
                'name': name,
                'm5': m5,
                'm15': m15,
                'm30': m30,
                'h1': h1,
                'vol_h1': float(vol_h1),
                'buys': int(buys),
                'sells': int(sells),
            })

    return pumping


def format_percentage(num):
    """Format percentage with color indicators"""
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Monitor trending Base tokens for price pumps >5%"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=-1,
        help="Maximum number of iterations (-1 for infinite, default: -1)",
    )
    args = parser.parse_args()

    print("🚀 BASE MEMECOIN PUMP MONITOR")
    if args.max_iter == -1:
        print("Scanning every 60s for >5% pumps (infinite mode)\n")
    else:
        print(f"Scanning every 60s for >5% pumps ({args.max_iter} iterations)\n")

    scan_count = 0

    while True:
        scan_count += 1
        now = datetime.now().strftime("%H:%M:%S")

        # Print scan header
        print(f"🔍 Scan #{scan_count} | {now}", end=" | ")

        pools = get_trending_base_tokens()

        if not pools:
            print("⚠️ No data")
        else:
            pumping = check_for_pumps(pools)

            if pumping:
                print(f"🔥 {len(pumping)} pumping:")
                for token in pumping:
                    name = token['name']
                    m5 = token['m5']
                    m15 = token['m15']
                    m30 = token['m30']
                    h1 = token['h1']
                    vol_h1 = token['vol_h1']
                    buys = token['buys']
                    sells = token['sells']

                    # Determine which timeframe is pumping hardest
                    max_pump = max(m5, m15, m30)
                    emoji = "🚀" if max_pump > 20 else "📈" if max_pump > 10 else "⬆️"

                    # Format volume
                    if vol_h1 >= 1_000_000:
                        vol_str = f"${vol_h1/1_000_000:.1f}M"
                    elif vol_h1 >= 1_000:
                        vol_str = f"${vol_h1/1_000:.1f}K"
                    else:
                        vol_str = f"${vol_h1:.0f}"

                    print(f"  {emoji} {name:35s} 5m:{m5:>+6.1f}% 15m:{m15:>+6.1f}% 30m:{m30:>+6.1f}% 1h:{h1:>+6.1f}% | Vol:{vol_str:>8s} Txn:{buys}B/{sells}S")
            else:
                print("😴 No pumps")

        # Check if we've reached max iterations
        if args.max_iter != -1 and scan_count >= args.max_iter:
            print(f"✅ Done ({scan_count} iterations)")
            break

        # Wait 60 seconds before next scan
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
