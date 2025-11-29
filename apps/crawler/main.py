import requests
import threading
import time
import random
from bs4 import BeautifulSoup
from url import filter_http_links
from pool import UrlPool, NoUrlAvailableError
import argparse
from profiler import profile, enable_profiling, print_stats


@profile
def fetch_url(url: str) -> str | None:
    if url.endswith('.pdf'):
        return None
    # Parquet can be huge.
    # TODO: we need to find a way to limit download size
    if url.endswith('.parquet'):
        return None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  -> ❌ Error fetching: {e}")
        return None


@profile
def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


def worker(worker_id: int, pool: UrlPool, iteration_counter: list, max_iterations: int):
    if worker_id != 0:
        delay = random.uniform(2, 5)
        # TODO: This is to warm up cold start with empty pool, optimize later
        time.sleep(delay)
        print(f"[Worker {worker_id}] Started after {delay:.2f}s delay")

    retry_count = 0
    max_retries = 3

    while True:
        with iteration_counter[1]:
            if iteration_counter[0] >= max_iterations:
                print(f"[Worker {worker_id}] Max iterations reached, exiting")
                break
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]

        try:
            url_obj = pool.get()
            retry_count = 0

            print(f"[{current_iter}][Worker {worker_id}] Crawling: {url_obj.url}")

            content = fetch_url(url_obj.url)
            if content is None:
                pool.error(url_obj.id)
            else:
                links = extract_links(content)
                links = filter_http_links(links, url_obj.url)
                pool.add_urls(links)
                # print(f"[{current_iter}][Worker {worker_id}] -> Found {len(links)} links")
                pool.done(url_obj.id)

        except NoUrlAvailableError:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"[Worker {worker_id}] No URLs available after {max_retries} retries, exiting")
                break
            print(f"[Worker {worker_id}] No URLs available, retry {retry_count}/{max_retries}")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple web crawler")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--threads", "-t", type=int, default=1, help="Number of worker threads (default: 1)")
    parser.add_argument("--profile", action="store_true", help="Enable function profiling")
    args = parser.parse_args()

    if args.profile:
        enable_profiling()
        print("Profiling enabled")

    pool = UrlPool()
    seed_url = "https://news.ycombinator.com"
    pool.add_url(seed_url)

    iteration_counter = [0, threading.Lock()]

    threads = []
    for i in range(args.threads):
        t = threading.Thread(target=worker, args=(i, pool, iteration_counter, args.num))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\nCrawling complete. Total iterations: {iteration_counter[0]}")

    if args.profile:
        print_stats()