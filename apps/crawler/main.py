import asyncio
import argparse
from crawler import SyncCrawler, AsyncCrawler
from profiler import enable_profiling, print_stats


def run_sync_crawler(args):
    seed_url = "https://news.ycombinator.com"
    crawler = SyncCrawler(seed_url, concurrency=args.concurrency)

    total = crawler.crawl(args.num)

    print(f"\nCrawling complete. Total iterations: {total}")


async def run_async_crawler(args):
    seed_url = "https://news.ycombinator.com"
    crawler = AsyncCrawler(seed_url, concurrency=args.concurrency)

    total = await crawler.crawl(args.num)

    print(f"\nCrawling complete. Total iterations: {total}")


def main():
    parser = argparse.ArgumentParser(description="Simple web crawler")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Number of concurrent workers (default: 1)")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async crawler with aiohttp (default: threading with requests)")
    parser.add_argument("--profile", action="store_true", help="Enable function profiling")
    args = parser.parse_args()

    if args.profile:
        enable_profiling()
        print("Profiling enabled")

    if args.use_async:
        print(f"Using AsyncCrawler (asyncio + aiohttp) with {args.concurrency} workers")
        asyncio.run(run_async_crawler(args))
    else:
        print(f"Using SyncCrawler (threading + requests) with {args.concurrency} workers")
        run_sync_crawler(args)

    if args.profile:
        print_stats()


if __name__ == "__main__":
    main()
