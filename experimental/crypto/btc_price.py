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

from web3 import Web3
import json
import argparse
import time

# Public Ethereum RPC endpoint
RPC_URL = "https://eth.llamarpc.com"

# Token addresses
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def connect_with_retry(rpc_url, max_retries=5, initial_delay=1):
    """
    Create Web3 connection with exponential backoff retry
    Returns Web3 instance or raises exception after max retries

    Note: Even if connection succeeds here, individual RPC calls can still fail
    with rate limits (429) or other errors. Consider adding retry logic to
    query methods as well.
    """
    for attempt in range(max_retries):
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            if w3.is_connected():
                return w3
            else:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Connection failed, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
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
        Discover all WBTC/USDC Uniswap V3 pools by querying the Factory contract
        Returns list of (fee_tier, pool_address) tuples for pools that exist
        """
        # Uniswap V3 Factory
        FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

        factory_abi = json.dumps([{
            "inputs": [
                {"internalType": "address", "name": "tokenA", "type": "address"},
                {"internalType": "address", "name": "tokenB", "type": "address"},
                {"internalType": "uint24", "name": "fee", "type": "uint24"}
            ],
            "name": "getPool",
            "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }])

        factory = self.w3.eth.contract(address=Web3.to_checksum_address(FACTORY), abi=factory_abi)

        # Fee tiers: 0.01%, 0.05%, 0.3%, 1%
        fee_tiers = [100, 500, 3000, 10000]
        pools = []

        for fee in fee_tiers:
            pool_address = factory.functions.getPool(
                Web3.to_checksum_address(WBTC),
                Web3.to_checksum_address(USDC),
                fee
            ).call()

            if pool_address != "0x0000000000000000000000000000000000000000":
                pools.append((fee, pool_address))

        return pools

    def get_price_from_pool(self, pool_address):
        """
        Query BTC price from a Uniswap V3 WBTC/USDC pool
        Returns price in USD
        """
        # Uniswap V3 Pool ABI - just the functions we need
        POOL_ABI = json.dumps([
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
                    {"internalType": "bool", "name": "unlocked", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token0",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token1",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ])

        # Create contract instance
        pool_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=POOL_ABI
        )

        # Get current pool state
        slot0 = pool_contract.functions.slot0().call()
        sqrt_price_x96 = slot0[0]

        # Get token order
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()

        WBTC_lower = WBTC.lower()

        # Calculate price from sqrtPriceX96
        # price = (sqrtPriceX96 / 2^96) ^ 2
        price = (sqrt_price_x96 / (2 ** 96)) ** 2

        # Adjust for decimals (WBTC: 8 decimals, USDC: 6 decimals)
        if token0.lower() == WBTC_lower:
            # token0 is WBTC, token1 is USDC
            # price gives us USDC per WBTC, need to adjust decimals
            btc_price = price * (10 ** 8) / (10 ** 6)
        else:
            # token1 is WBTC, token0 is USDC
            # price gives us WBTC per USDC, need to invert
            btc_price = (1 / price) * (10 ** 8) / (10 ** 6)

        return btc_price


class ChainlinkPriceQuery:
    """Query Bitcoin price from Chainlink oracle with a single reusable connection"""

    def __init__(self, w3):
        """Initialize with existing Web3 connection"""
        self.w3 = w3

    def get_btc_price(self):
        """
        Query BTC price from Chainlink Price Feed oracle
        Returns price in USD
        """
        # Chainlink BTC/USD Price Feed on Ethereum mainnet
        CHAINLINK_BTC_USD = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"

        # Chainlink Price Feed ABI - just latestRoundData function
        PRICE_FEED_ABI = json.dumps([
            {
                "inputs": [],
                "name": "latestRoundData",
                "outputs": [
                    {"internalType": "uint80", "name": "roundId", "type": "uint80"},
                    {"internalType": "int256", "name": "answer", "type": "int256"},
                    {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
                    {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
                    {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ])

        price_feed = self.w3.eth.contract(
            address=Web3.to_checksum_address(CHAINLINK_BTC_USD),
            abi=PRICE_FEED_ABI
        )

        round_data = price_feed.functions.latestRoundData().call()
        price = round_data[1]

        # Chainlink BTC/USD feed returns price with 8 decimals
        btc_price = price / (10 ** 8)

        return btc_price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Bitcoin price from on-chain sources")
    parser.add_argument("--n", type=int, default=1, help="Number of times to query (default: 1, -1 for infinite)")
    parser.add_argument("--gap", type=int, default=10, help="Seconds to wait between queries (default: 10)")
    args = parser.parse_args()

    # Create single Web3 connection with retry logic
    print("Connecting to Ethereum RPC...")
    w3 = connect_with_retry(RPC_URL)
    print("Connected!\n")

    # Initialize query classes with shared connection
    uniswap = UniswapPriceQuery(w3)
    chainlink = ChainlinkPriceQuery(w3)

    # Discover all WBTC/USDC pools once
    print("Discovering WBTC/USDC pools...")
    uniswap_pools = uniswap.discover_pools()
    print(f"Found {len(uniswap_pools)} pools\n")

    i = 0
    while True:
        if i > 0:
            time.sleep(args.gap)

        # Query prices from all discovered Uniswap pools
        for fee_tier, pool_address in uniswap_pools:
            fee_pct = fee_tier / 10000
            try:
                price = uniswap.get_price_from_pool(pool_address)
                print(f"{'Uni ' + str(fee_pct) + '%:':<13} ${price:,.2f}")
            except Exception as e:
                print(f"{'Uni ' + str(fee_pct) + '%:':<13} Error: {e}")

        # Query Chainlink
        chainlink_price = chainlink.get_btc_price()
        print(f"{'Chainlink:':<13} ${chainlink_price:,.2f}")

        i += 1

        if args.n != -1 and i >= args.n:
            break

        print()
