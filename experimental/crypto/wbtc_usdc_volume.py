#!/usr/bin/env python3
"""
Compute hourly WBTC/USDC swap volume on Uniswap V3 over the last 24 hours.

Uses an Ethereum RPC endpoint (e.g. Alchemy via ALCHEMY_RPC_URL) and only
talks directly to the chain (no subgraphs or off-chain APIs).
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
from pathlib import Path
from urllib.parse import urlparse

from btc_price import USDC, WBTC, connect_with_retry, retry_with_backoff
from requests.exceptions import HTTPError
from web3 import Web3

# Ethereum RPC endpoint - prefer Alchemy if available
RPC_URL = os.environ.get("ALCHEMY_RPC_URL", "https://eth.llamarpc.com")


UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

# Fee tiers in Uniswap V3 (in hundredths of a bip: 100 = 0.01%, 500 = 0.05%, etc.)
UNISWAP_V3_FEE_TIERS = [100, 500, 3000, 10000]


UNISWAP_V3_FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
        ],
        "name": "getPool",
        "outputs": [
            {"internalType": "address", "name": "pool", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


UNISWAP_V3_POOL_ABI = [
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
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "sender",
                "type": "address",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "recipient",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "int256",
                "name": "amount0",
                "type": "int256",
            },
            {
                "indexed": False,
                "internalType": "int256",
                "name": "amount1",
                "type": "int256",
            },
            {
                "indexed": False,
                "internalType": "uint160",
                "name": "sqrtPriceX96",
                "type": "uint160",
            },
            {
                "indexed": False,
                "internalType": "uint128",
                "name": "liquidity",
                "type": "uint128",
            },
            {
                "indexed": False,
                "internalType": "int24",
                "name": "tick",
                "type": "int24",
            },
        ],
        "name": "Swap",
        "type": "event",
    },
]


SWAP_EVENT_TOPIC0 = Web3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()


def discover_wbtc_usdc_pools(w3: Web3) -> list[tuple[str, int]]:
    """
    Discover all Uniswap V3 WBTC/USDC pools across standard fee tiers.
    Returns list of pool addresses.
    """
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(UNISWAP_V3_FACTORY),
        abi=UNISWAP_V3_FACTORY_ABI,
    )

    pools: list[tuple[str, int]] = []
    wbtc = Web3.to_checksum_address(WBTC)
    usdc = Web3.to_checksum_address(USDC)

    for fee in UNISWAP_V3_FEE_TIERS:
        pool_address = factory.functions.getPool(wbtc, usdc, fee).call()
        # Zero address means pool does not exist
        if pool_address != "0x0000000000000000000000000000000000000000":
            pools.append((Web3.to_checksum_address(pool_address), fee))

    return pools


@retry_with_backoff()
def get_block_timestamp(w3: Web3, block_number: int) -> int:
    """Get block timestamp with retry/backoff."""
    return w3.eth.get_block(block_number).timestamp


@retry_with_backoff()
def get_block_with_retry(w3: Web3, block_number: int):
    """Get full block (without tx objects) with retry/backoff."""
    return w3.eth.get_block(block_number, full_transactions=False)


def fetch_block_json(
    w3: Web3,
    block_number: int,
) -> dict:
    block = get_block_with_retry(w3, block_number)
    # Web3.to_json handles HexBytes/AttributeDict correctly.
    return json.loads(Web3.to_json(block))


def write_block_json(out_path: Path, block_json: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(block_json, indent=2, sort_keys=True) + "\n")


def write_jsonl(out_path: Path, items: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, sort_keys=True) + "\n")


def cache_swap_logs(
    raw_logs: list,
    from_block: int,
    to_block: int,
    out_dir: Path,
) -> None:
    """
    Cache raw eth_getLogs results (Swap logs) to disk for later inspection/reuse.
    Stores one JSONL file per queried block chunk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"swap_logs_{from_block}_{to_block}.jsonl"
    if out_path.exists():
        return
    logs_json = [json.loads(Web3.to_json(log)) for log in raw_logs]
    write_jsonl(out_path, logs_json)


def cache_blocks(
    w3: Web3,
    from_block: int,
    to_block: int,
    out_dir: Path,
) -> None:
    """
    Ensure blocks in [from_block, to_block] are cached on disk as JSON files.
    Skips any block already present to avoid unnecessary RPC calls.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for block_number in range(from_block, to_block + 1):
        out_path = out_dir / f"{block_number}.json"
        if out_path.exists():
            continue
        write_block_json(out_path, fetch_block_json(w3, block_number))


def align_block_range(from_block: int, to_block: int, chunk_size: int) -> tuple[int, int]:
    """
    Align the start of the range to a clean chunk boundary so chunking produces
    nicely-aligned requests (e.g., for chunk_size=10, starts at multiples of 10).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    aligned_from = (from_block // chunk_size) * chunk_size
    return aligned_from, to_block


def find_block_for_timestamp(w3: Web3, target_ts: int, latest_block: int | None = None) -> int:
    """
    Binary search to find the earliest block whose timestamp >= target_ts.
    Returns a block number in [0, latest_block].
    """
    if latest_block is None:
        latest_block = w3.eth.block_number

    # Guard for very early timestamps
    genesis_ts = get_block_timestamp(w3, 0)
    if target_ts <= genesis_ts:
        return 0

    low = 0
    high = latest_block

    while low < high:
        mid = (low + high) // 2
        ts = get_block_timestamp(w3, mid)
        if ts < target_ts:
            low = mid + 1
        else:
            high = mid

    return low


def get_logs_with_retry(
    w3: Web3,
    address: str | list[str],
    from_block: int,
    to_block: int,
    max_retries: int = 5,
    initial_delay: int = 2,
):
    """
    Call eth_getLogs with basic retry logic for transient RPC issues.
    """
    filter_params = {
        # Use explicit hex strings for block numbers for maximum RPC compatibility
        "fromBlock": Web3.to_hex(from_block),
        "toBlock": Web3.to_hex(to_block),
        "address": (
            [Web3.to_checksum_address(a) for a in address]
            if isinstance(address, list)
            else Web3.to_checksum_address(address)
        ),
        "topics": [SWAP_EVENT_TOPIC0],
    }

    attempt = 0
    while True:
        try:
            return w3.eth.get_logs(filter_params)
        except Exception as e:
            message = str(e)
            is_retriable = (
                "429" in message
                or "Too Many Requests" in message
                or "rate limit" in message.lower()
                or "connection" in message.lower()
                or "no response" in message.lower()
            )

            if is_retriable and attempt < max_retries - 1:
                delay = initial_delay * (2**attempt)
                print(f"RPC error (get_logs): {e}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                attempt += 1
                continue

            # Non-retriable (or max retries hit) – let caller decide
            raise


def get_logs_safely(
    w3: Web3,
    address: str | list[str],
    from_block: int,
    to_block: int,
    max_splits: int = 5,
    max_429_retries: int = 8,
):
    """
    Fetch logs for a block range, recursively splitting on provider 400 errors
    (e.g. payload too large) until the range is small enough.
    """
    attempt_429 = 0
    while True:
        try:
            return get_logs_with_retry(
                w3=w3,
                address=address,
                from_block=from_block,
                to_block=to_block,
            )
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else ""
            print(f"eth_getLogs HTTPError {status} for blocks {from_block}-{to_block}: {body}")

            if status == 429 and attempt_429 < max_429_retries:
                delay = min(60, 2**attempt_429)
                print(f"Rate limited (429), sleeping {delay}s then retrying...")
                time.sleep(delay)
                attempt_429 += 1
                continue

            if status == 400 and from_block < to_block and max_splits > 0:
                mid = (from_block + to_block) // 2
                left = get_logs_safely(
                    w3,
                    address,
                    from_block,
                    mid,
                    max_splits - 1,
                    max_429_retries=max_429_retries,
                )
                right = get_logs_safely(
                    w3,
                    address,
                    mid + 1,
                    to_block,
                    max_splits - 1,
                    max_429_retries=max_429_retries,
                )
                return left + right

            raise


def fetch_swap_events_for_pool(
    w3: Web3,
    pool_address: str,
    from_block: int,
    to_block: int,
    chunk_size: int = 5000,
    verbose: bool = False,
):
    """
    Fetch all Swap events for a given pool and block range, chunking requests
    to avoid RPC limits.
    """
    pool_contract = w3.eth.contract(
        address=Web3.to_checksum_address(pool_address),
        abi=UNISWAP_V3_POOL_ABI,
    )
    swap_event = pool_contract.events.Swap()

    logs = []
    out_dir = Path(__file__).resolve().parent / "eth_uniswap_wbtc_usdc_blocks"
    swap_logs_dir = Path(__file__).resolve().parent / "eth_uniswap_wbtc_usdc_swap_events"
    from_block, to_block = align_block_range(from_block, to_block, chunk_size)
    start = from_block
    while start <= to_block:
        end = min(start + chunk_size - 1, to_block)
        if verbose:
            print(
                f"eth_getLogs request | pools=1 | "
                f"from={start} ({Web3.to_hex(start)}) | to={end} ({Web3.to_hex(end)}) | "
                f"chunk_size={chunk_size} | address={pool_address}"
            )
        raw_logs = get_logs_safely(
            w3=w3,
            address=pool_address,
            from_block=start,
            to_block=end,
        )
        if verbose:
            blocks_with_logs = sorted({int(raw["blockNumber"]) for raw in raw_logs})
            print(f"eth_getLogs result  | logs={len(raw_logs)} | blocks_with_logs={blocks_with_logs}")
        cache_swap_logs(raw_logs, from_block=start, to_block=end, out_dir=swap_logs_dir)
        cache_blocks(w3=w3, from_block=start, to_block=end, out_dir=out_dir)
        # Decode logs using the contract event ABI
        for raw in raw_logs:
            logs.append(swap_event.process_log(raw))
        start = end + 1

    # Also return token ordering so we know which token is WBTC vs USDC
    token0 = pool_contract.functions.token0().call()
    token1 = pool_contract.functions.token1().call()

    return logs, Web3.to_checksum_address(token0), Web3.to_checksum_address(token1)


def fetch_swap_events_for_pools(
    w3: Web3,
    pools: list[str],
    from_block: int,
    to_block: int,
    chunk_size: int = 5000,
    verbose: bool = False,
) -> tuple[dict[str, list], dict[str, tuple[str, str]]]:
    """
    Fetch Swap events for multiple pools over a block range, chunking requests.
    This batches eth_getLogs calls by querying multiple pool addresses at once.
    """
    if not pools:
        return {}, {}

    pool_contract = w3.eth.contract(
        address=Web3.to_checksum_address(pools[0]),
        abi=UNISWAP_V3_POOL_ABI,
    )
    swap_event = pool_contract.events.Swap()

    logs_by_pool: dict[str, list] = {Web3.to_checksum_address(p): [] for p in pools}
    out_dir = Path(__file__).resolve().parent / "eth_uniswap_wbtc_usdc_blocks"
    swap_logs_dir = Path(__file__).resolve().parent / "eth_uniswap_wbtc_usdc_swap_events"

    from_block, to_block = align_block_range(from_block, to_block, chunk_size)
    start = from_block
    while start <= to_block:
        end = min(start + chunk_size - 1, to_block)
        if verbose:
            print(
                f"eth_getLogs request | pools={len(pools)} | "
                f"from={start} ({Web3.to_hex(start)}) | to={end} ({Web3.to_hex(end)}) | "
                f"chunk_size={chunk_size}"
            )
        raw_logs = get_logs_safely(
            w3=w3,
            address=[Web3.to_checksum_address(p) for p in pools],
            from_block=start,
            to_block=end,
        )
        if verbose:
            blocks_with_logs = sorted({int(raw["blockNumber"]) for raw in raw_logs})
            print(f"eth_getLogs result  | logs={len(raw_logs)} | blocks_with_logs={blocks_with_logs}")
        cache_swap_logs(raw_logs, from_block=start, to_block=end, out_dir=swap_logs_dir)
        cache_blocks(w3=w3, from_block=start, to_block=end, out_dir=out_dir)
        for raw in raw_logs:
            decoded = swap_event.process_log(raw)
            pool_addr = Web3.to_checksum_address(decoded["address"])
            logs_by_pool.setdefault(pool_addr, []).append(decoded)
        start = end + 1

    token_order_by_pool: dict[str, tuple[str, str]] = {}
    for pool_address in pools:
        pool_contract = w3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=UNISWAP_V3_POOL_ABI,
        )
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()
        token_order_by_pool[Web3.to_checksum_address(pool_address)] = (
            Web3.to_checksum_address(token0),
            Web3.to_checksum_address(token1),
        )

    return logs_by_pool, token_order_by_pool


def compute_hourly_buckets(
    w3: Web3,
    pools: list[tuple[str, int]],
    hours: int = 24,
    blocks: int | None = None,
    chunk_size: int = 5000,
    verbose: bool = False,
) -> tuple[
    list[float],
    list[float],
    dict[str, list[float]],
    dict[str, list[float]],
    int,
    int,
]:
    """
    Compute hourly WBTC and USDC volume for the given pools over the last
    `hours` hours.

    Returns:
      btc_buckets: aggregated BTC volume per hour across all pools
      usdc_buckets: aggregated USDC volume per hour across all pools
      per_pool_btc: mapping pool_address -> list of BTC volume per hour
      per_pool_usdc: mapping pool_address -> list of USDC volume per hour
      start_ts: timestamp of the start of the first bucket
      end_ts: timestamp of the end of the last bucket
    """
    latest_block = w3.eth.block_number
    latest_ts = get_block_timestamp(w3, latest_block)

    window_seconds = hours * 3600
    end_ts = latest_ts
    start_ts = end_ts - window_seconds

    from_block = find_block_for_timestamp(w3, start_ts, latest_block=latest_block)

    btc_buckets = [0.0 for _ in range(hours)]
    usdc_buckets = [0.0 for _ in range(hours)]
    per_pool_btc: dict[str, list[float]] = {}
    per_pool_usdc: dict[str, list[float]] = {}

    # Cache block timestamps so we only fetch each block once
    block_ts_cache: dict[int, int] = {}

    if blocks is not None:
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        from_block = max(0, latest_block - (blocks - 1))
        from_block, _ = align_block_range(from_block, latest_block, chunk_size=10)
        start_ts = get_block_timestamp(w3, from_block)
        window_seconds = max(1, end_ts - start_ts)
        hours = max(1, (window_seconds + 3599) // 3600)
        start_ts = end_ts - (hours * 3600)
    else:
        window_seconds = hours * 3600
        end_ts = latest_ts
        start_ts = end_ts - window_seconds
        from_block = find_block_for_timestamp(w3, start_ts, latest_block=latest_block)

    wbtc_addr = Web3.to_checksum_address(WBTC)
    usdc_addr = Web3.to_checksum_address(USDC)

    pool_addresses = [addr for addr, _fee in pools]
    logs_by_pool, token_order_by_pool = fetch_swap_events_for_pools(
        w3=w3,
        pools=pool_addresses,
        from_block=from_block,
        to_block=latest_block,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    for pool_address, _fee in pools:
        logs = logs_by_pool.get(Web3.to_checksum_address(pool_address), [])
        token0, token1 = token_order_by_pool.get(Web3.to_checksum_address(pool_address), ("", ""))
        pool_btc = [0.0 for _ in range(hours)]
        pool_usdc = [0.0 for _ in range(hours)]

        for log in logs:
            block_number = log["blockNumber"]
            if block_number not in block_ts_cache:
                block_ts_cache[block_number] = get_block_timestamp(w3, block_number)

            ts = block_ts_cache[block_number]

            # Skip events slightly outside the intended window due to block search approximation
            if ts < start_ts or ts > end_ts:
                continue

            offset = ts - start_ts
            bucket = int(offset // 3600)
            if bucket < 0 or bucket >= hours:
                continue

            args = log["args"]
            amount0 = args["amount0"]
            amount1 = args["amount1"]

            # Identify which token is WBTC vs USDC
            if token0 == wbtc_addr and token1 == usdc_addr:
                btc_raw = amount0
                usdc_raw = amount1
            elif token1 == wbtc_addr and token0 == usdc_addr:
                btc_raw = amount1
                usdc_raw = amount0
            else:
                # Unexpected tokens – skip
                continue

            # Use absolute value: volume is direction-agnostic
            btc_amount = abs(btc_raw) / (10**8)
            usdc_amount = abs(usdc_raw) / (10**6)

            btc_buckets[bucket] += btc_amount
            usdc_buckets[bucket] += usdc_amount
            pool_btc[bucket] += btc_amount
            pool_usdc[bucket] += usdc_amount

        per_pool_btc[pool_address] = pool_btc
        per_pool_usdc[pool_address] = pool_usdc

    return btc_buckets, usdc_buckets, per_pool_btc, per_pool_usdc, start_ts, end_ts


def format_utc(ts: int) -> str:
    """Format a Unix timestamp as a human-readable UTC string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))


def main():
    parser = argparse.ArgumentParser(
        description="Compute hourly WBTC/USDC volume on Uniswap V3 over the last 24 hours",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to look back (default: 24)",
    )
    window.add_argument(
        "--blocks",
        type=int,
        help="Only scan the last N blocks (free-tier friendly)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help=(
            "Block range chunk size for eth_getLogs queries (default: 10). "
            "Alchemy free tier enforces a 10-block max range per eth_getLogs request."
        ),
    )
    parser.add_argument(
        "--per-pool",
        action="store_true",
        help="Print per-pool (fee tier) hourly breakdown (default: false)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging for RPC requests/results (default: false)",
    )
    args = parser.parse_args()

    if args.blocks is None:
        hours = args.hours
        if hours <= 0:
            raise ValueError("hours must be positive")
    else:
        hours = 1

    rpc_host = urlparse(RPC_URL).netloc
    print(f"Connecting to Ethereum RPC: {rpc_host}")
    w3 = connect_with_retry(RPC_URL)
    print("Connected!\n")

    print("Discovering Uniswap V3 WBTC/USDC pools...")
    pools = discover_wbtc_usdc_pools(w3)
    if not pools:
        print("No WBTC/USDC Uniswap V3 pools found.")
        return

    print(f"Found {len(pools)} WBTC/USDC pools:")
    for addr, fee in pools:
        fee_pct = fee / 1_000_000  # fee is in hundredths of a bip
        print(f"  - {addr} (fee {fee_pct:.2%})")
    print()

    (
        btc_buckets,
        usdc_buckets,
        per_pool_btc,
        per_pool_usdc,
        start_ts,
        end_ts,
    ) = compute_hourly_buckets(
        w3=w3,
        pools=pools,
        hours=hours,
        blocks=args.blocks,
        chunk_size=min(args.chunk_size, 10) if args.blocks is not None else args.chunk_size,
        verbose=args.verbose,
    )

    print(f"Window start (UTC): {format_utc(start_ts)}")
    print(f"Window end   (UTC): {format_utc(end_ts)}\n")

    print("Hourly WBTC/USDC volume (Uniswap V3, aggregated across all fee tiers):")
    for i in range(hours):
        bucket_start = start_ts + i * 3600
        bucket_end = bucket_start + 3600
        btc_vol = btc_buckets[i]
        usdc_vol = usdc_buckets[i]

        print(f"{format_utc(bucket_start)} - {format_utc(bucket_end)} UTC | {btc_vol:,.6f} BTC | {usdc_vol:,.2f} USDC")

    total_btc = sum(btc_buckets)
    total_usdc = sum(usdc_buckets)
    print(f"TOTAL | {total_btc:,.6f} BTC | {total_usdc:,.2f} USDC")

    if args.per_pool:
        print("\nPer-pool hourly WBTC/USDC volume (Uniswap V3):")
        for addr, fee in pools:
            fee_pct = fee / 1_000_000
            print(f"\nPool {addr} (fee {fee_pct:.2%}):")
            pool_btc = per_pool_btc.get(addr, [])
            pool_usdc = per_pool_usdc.get(addr, [])

            for i in range(hours):
                bucket_start = start_ts + i * 3600
                bucket_end = bucket_start + 3600
                btc_vol = pool_btc[i] if i < len(pool_btc) else 0.0
                usdc_vol = pool_usdc[i] if i < len(pool_usdc) else 0.0

                print(
                    f"{format_utc(bucket_start)} - {format_utc(bucket_end)} UTC | "
                    f"{btc_vol:,.6f} BTC | {usdc_vol:,.2f} USDC"
                )


if __name__ == "__main__":
    main()
