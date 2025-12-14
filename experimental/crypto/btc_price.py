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


def _get_btc_price_from_uniswap_pool(pool_address):
    """
    Helper function to query BTC price from a Uniswap V3 WBTC/USDC pool
    Returns price in USD
    """
    # Public Ethereum RPC endpoint
    RPC_URL = "https://eth.llamarpc.com"

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

    # Connect to Ethereum
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    if not w3.is_connected():
        raise Exception("Failed to connect to Ethereum RPC")

    # Create contract instance
    pool_contract = w3.eth.contract(
        address=Web3.to_checksum_address(pool_address),
        abi=POOL_ABI
    )

    # Get current pool state
    slot0 = pool_contract.functions.slot0().call()
    sqrt_price_x96 = slot0[0]

    # Get token order
    token0 = pool_contract.functions.token0().call()
    token1 = pool_contract.functions.token1().call()

    # WBTC address: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
    # USDC address: 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
    WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599".lower()

    # Calculate price from sqrtPriceX96
    # price = (sqrtPriceX96 / 2^96) ^ 2
    price = (sqrt_price_x96 / (2 ** 96)) ** 2

    # Adjust for decimals (WBTC: 8 decimals, USDC: 6 decimals)
    if token0.lower() == WBTC:
        # token0 is WBTC, token1 is USDC
        # price gives us USDC per WBTC, need to adjust decimals
        btc_price = price * (10 ** 8) / (10 ** 6)
    else:
        # token1 is WBTC, token0 is USDC
        # price gives us WBTC per USDC, need to invert
        btc_price = (1 / price) * (10 ** 8) / (10 ** 6)

    return btc_price


def get_btc_price_from_uniswap_005():
    """Query BTC price from Uniswap V3 WBTC/USDC 0.05% fee tier pool"""
    return _get_btc_price_from_uniswap_pool("0x9a772018FbD77fcD2d25657e5C547BAfF3Fd7D16")


def get_btc_price_from_uniswap_030():
    """Query BTC price from Uniswap V3 WBTC/USDC 0.3% fee tier pool"""
    return _get_btc_price_from_uniswap_pool("0x99ac8cA7087fA4A2A1FB6357269965A2014ABc35")


def get_btc_price_from_chainlink():
    """
    Query BTC price from Chainlink Price Feed oracle
    Returns price in USD
    """
    # Public Ethereum RPC endpoint
    RPC_URL = "https://eth.llamarpc.com"

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

    # Connect to Ethereum
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    if not w3.is_connected():
        raise Exception("Failed to connect to Ethereum RPC")

    # Create contract instance
    price_feed = w3.eth.contract(
        address=Web3.to_checksum_address(CHAINLINK_BTC_USD),
        abi=PRICE_FEED_ABI
    )

    # Get latest price data
    round_data = price_feed.functions.latestRoundData().call()
    price = round_data[1]  # answer is at index 1

    # Chainlink BTC/USD feed returns price with 8 decimals
    btc_price = price / (10 ** 8)

    return btc_price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Bitcoin price from on-chain sources")
    parser.add_argument("--n", type=int, default=1, help="Number of times to query (default: 1, -1 for infinite)")
    parser.add_argument("--gap", type=int, default=5, help="Seconds to wait between queries (default: 5)")
    args = parser.parse_args()

    i = 0
    while True:
        if i > 0:
            time.sleep(args.gap)

        uniswap_005 = get_btc_price_from_uniswap_005()
        uniswap_030 = get_btc_price_from_uniswap_030()
        chainlink_price = get_btc_price_from_chainlink()

        print(f"{'Uni 0.05%:':<13} ${uniswap_005:,.2f}")
        print(f"{'Uni 0.3%:':<13} ${uniswap_030:,.2f}")
        print(f"{'Chainlink:':<13} ${chainlink_price:,.2f}")

        i += 1

        if args.n != -1 and i >= args.n:
            break

        print()
