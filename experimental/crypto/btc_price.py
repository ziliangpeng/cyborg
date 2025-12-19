#!/usr/bin/env python3
"""
Query Bitcoin (WBTC) price from Uniswap V3 on Ethereum
Uses public RPC endpoint - no centralized exchange needed
"""
# /// script
# dependencies = [
#   "web3",
# ]
# ///

import argparse
import json
import os
import time
from urllib.parse import urlparse

from web3 import Web3

# Ethereum RPC endpoint - use Alchemy if available, otherwise public endpoint
RPC_URL = os.environ.get("ALCHEMY_RPC_URL", "https://eth.llamarpc.com")

# Token addresses
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"


def retry_with_backoff(max_retries=8, initial_delay=2):
    """
    Decorator to retry functions with exponential backoff
    Handles RPC rate limits (429) and temporary connection errors
    Default: 8 retries with 2s initial delay (2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit or retriable error
                    is_retriable = (
                        "429" in str(e)
                        or "Too Many Requests" in str(e)
                        or "rate limit" in str(e).lower()
                        or "connection" in str(e).lower()
                        or "no response" in str(e).lower()
                    )

                    if attempt < max_retries - 1 and is_retriable:
                        delay = initial_delay * (2**attempt)
                        print(f"RPC error: {e}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise

            raise Exception(f"Failed after {max_retries} attempts")

        return wrapper

    return decorator


def connect_with_retry(rpc_url, max_retries=5, initial_delay=1):
    """
    Create Web3 connection with exponential backoff retry
    Returns Web3 instance or raises exception after max retries
    """
    for attempt in range(max_retries):
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            if w3.is_connected():
                return w3
            else:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2**attempt)
                    print(f"Connection failed, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2**attempt)
                print(f"Connection error: {e}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise

    raise Exception(f"Failed to connect to {rpc_url} after {max_retries} attempts")


class UniswapPriceQuery:
    """Query Bitcoin prices from Uniswap V3 pools with a single reusable connection"""

    def __init__(self, w3):
        """Initialize with existing Web3 connection"""
        self.w3 = w3

    def discover_pools(self):
        """
        Discover all WBTC/USDC and WBTC/USDT Uniswap V3 pools by querying the Factory contract
        Returns list of (fee_tier, pool_address, pair_name) tuples for pools that exist
        """
        # Uniswap V3 Factory
        FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

        factory_abi = json.dumps(
            [
                {
                    "inputs": [
                        {"internalType": "address", "name": "tokenA", "type": "address"},
                        {"internalType": "address", "name": "tokenB", "type": "address"},
                        {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    ],
                    "name": "getPool",
                    "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
        )

        factory = self.w3.eth.contract(address=Web3.to_checksum_address(FACTORY), abi=factory_abi)

        # Fee tiers: 0.01%, 0.05%, 0.3%, 1%
        fee_tiers = [100, 500, 3000, 10000]
        pools = []

        # Check WBTC/USDC pools
        for fee in fee_tiers:
            pool_address = factory.functions.getPool(
                Web3.to_checksum_address(WBTC), Web3.to_checksum_address(USDC), fee
            ).call()

            if pool_address != "0x0000000000000000000000000000000000000000":
                pools.append((fee, pool_address, "USDC"))

        # Check WBTC/USDT pools
        for fee in fee_tiers:
            pool_address = factory.functions.getPool(
                Web3.to_checksum_address(WBTC), Web3.to_checksum_address(USDT), fee
            ).call()

            if pool_address != "0x0000000000000000000000000000000000000000":
                pools.append((fee, pool_address, "USDT"))

        return pools

    @retry_with_backoff()
    def get_tvl_from_pool(self, pool_address):
        """
        Query TVL (Total Value Locked) from a Uniswap V3 pool in USD
        Returns TVL by reading token balances and converting to USD
        """
        # ERC20 ABI for balanceOf
        ERC20_ABI = json.dumps(
            [
                {
                    "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
        )

        # Pool ABI for token addresses
        POOL_ABI = json.dumps(
            [
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]
        )

        pool_contract = self.w3.eth.contract(address=Web3.to_checksum_address(pool_address), abi=POOL_ABI)

        # Get token addresses
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()

        USDC_lower = USDC.lower()
        USDT_lower = USDT.lower()

        # Get stablecoin balance (USDC or USDT, both have 6 decimals)
        stablecoin_address = token0 if token0.lower() == USDC_lower or token0.lower() == USDT_lower else token1

        stablecoin_contract = self.w3.eth.contract(address=Web3.to_checksum_address(stablecoin_address), abi=ERC20_ABI)

        # Get stablecoin balance in the pool
        stablecoin_balance = stablecoin_contract.functions.balanceOf(Web3.to_checksum_address(pool_address)).call()

        # USDC and USDT have 6 decimals, TVL = 2 × stablecoin balance (since it's roughly 50/50)
        tvl_usd = 2 * stablecoin_balance / (10**6)

        return tvl_usd

    @retry_with_backoff()
    def get_price_from_pool(self, pool_address):
        """
        Query BTC price from a Uniswap V3 WBTC/stablecoin pool
        Returns price in USD
        """
        # Uniswap V3 Pool ABI - just the functions we need
        POOL_ABI = json.dumps(
            [
                {
                    "inputs": [],
                    "name": "slot0",
                    "outputs": [
                        {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                        {"internalType": "int24", "name": "tick", "type": "int24"},
                        {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
                        {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
                        {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
                        {"internalType": "bool", "name": "unlocked", "type": "bool"},
                    ],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]
        )

        # Create contract instance
        pool_contract = self.w3.eth.contract(address=Web3.to_checksum_address(pool_address), abi=POOL_ABI)

        # Get current pool state
        slot0 = pool_contract.functions.slot0().call()
        sqrt_price_x96 = slot0[0]

        # Get token order
        token0 = pool_contract.functions.token0().call()

        WBTC_lower = WBTC.lower()

        # Calculate price from sqrtPriceX96
        # price = (sqrtPriceX96 / 2^96) ^ 2
        price = (sqrt_price_x96 / (2**96)) ** 2

        # Adjust for decimals (WBTC: 8 decimals, USDC/USDT: 6 decimals)
        if token0.lower() == WBTC_lower:  # noqa: SIM108
            # token0 is WBTC, token1 is stablecoin
            # price gives us stablecoin per WBTC, need to adjust decimals
            btc_price = price * (10**8) / (10**6)
        else:
            # token1 is WBTC, token0 is stablecoin
            # price gives us WBTC per stablecoin, need to invert
            btc_price = (1 / price) * (10**8) / (10**6)

        return btc_price


class ChainlinkPriceQuery:
    """Query Bitcoin price from Chainlink oracle with a single reusable connection"""

    def __init__(self, w3):
        """Initialize with existing Web3 connection"""
        self.w3 = w3

    @retry_with_backoff()
    def get_btc_price(self):
        """
        Query BTC price from Chainlink Price Feed oracle
        Returns price in USD
        """
        # Chainlink BTC/USD Price Feed on Ethereum mainnet
        CHAINLINK_BTC_USD = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"

        # Chainlink Price Feed ABI - just latestRoundData function
        PRICE_FEED_ABI = json.dumps(
            [
                {
                    "inputs": [],
                    "name": "latestRoundData",
                    "outputs": [
                        {"internalType": "uint80", "name": "roundId", "type": "uint80"},
                        {"internalType": "int256", "name": "answer", "type": "int256"},
                        {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
                        {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
                        {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
                    ],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
        )

        price_feed = self.w3.eth.contract(address=Web3.to_checksum_address(CHAINLINK_BTC_USD), abi=PRICE_FEED_ABI)

        round_data = price_feed.functions.latestRoundData().call()
        price = round_data[1]

        # Chainlink BTC/USD feed returns price with 8 decimals
        btc_price = price / (10**8)

        return btc_price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Bitcoin price from on-chain sources")
    parser.add_argument("--n", type=int, default=1, help="Number of times to query (default: 1, -1 for infinite)")
    parser.add_argument("--gap", type=int, default=60, help="Seconds to wait between queries (default: 60)")
    args = parser.parse_args()

    # Create single Web3 connection with retry logic
    rpc_host = urlparse(RPC_URL).netloc
    print(f"Connecting to Ethereum RPC: {rpc_host}")
    w3 = connect_with_retry(RPC_URL)
    print("Connected!\n")

    # Initialize query classes with shared connection
    uniswap = UniswapPriceQuery(w3)
    chainlink = ChainlinkPriceQuery(w3)

    # Discover all WBTC/USDC and WBTC/USDT pools once
    print("Discovering Uniswap V3 WBTC/USDC and WBTC/USDT pools...")
    uniswap_pools = uniswap.discover_pools()
    print(f"Found {len(uniswap_pools)} Uniswap V3 pools\n")

    i = 0
    while True:
        if i > 0:
            time.sleep(args.gap)

        # Query prices from all discovered Uniswap V3 pools
        # Note: Uniswap AMMs have a single spot price (no bid-ask spread like CEXes).
        # The "spread" manifests as slippage + fees when you actually trade.
        # USDC and USDT are pegged 1:1 to USD, so we display the price once per pool.
        for fee_tier, pool_address, pair_name in uniswap_pools:
            fee_pct = fee_tier / 10000
            try:
                btc_price = uniswap.get_price_from_pool(pool_address)
                tvl = uniswap.get_tvl_from_pool(pool_address)
                label = f"Uni {pair_name} {fee_pct}%:"
                print(f"{label:<17} ${btc_price:,.2f} | TVL: ${tvl:,.2f}")
            except Exception as e:
                label = f"Uni {pair_name} {fee_pct}%:"
                print(f"{label:<17} Error: {e}")

        # Query Chainlink
        # Chainlink reports mid-market reference price aggregated from multiple exchanges
        chainlink_price = chainlink.get_btc_price()
        print(f"{'Chainlink:':<13} ${chainlink_price:,.2f}")

        i += 1

        if args.n != -1 and i >= args.n:
            break

        print()
