import threading
import time
import random
from bs4 import BeautifulSoup
from url import filter_http_links, Fetcher
from pool import UrlPool, NoUrlAvailableError
import argparse
from profiler import profile, enable_profiling, print_stats


@profile
def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


def worker(worker_id: int, pool: UrlPool, fetcher: Fetcher, iteration_counter: list, max_iterations: int):
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

            content = fetcher.fetch(url_obj.url)
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
    fetcher = Fetcher()
    seed_url = "https://news.ycombinator.com"
    pool.add_url(seed_url)

    iteration_counter = [0, threading.Lock()]

    threads = []
    for i in range(args.threads):
        t = threading.Thread(target=worker, args=(i, pool, fetcher, iteration_counter, args.num))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\nCrawling complete. Total iterations: {iteration_counter[0]}")

    if args.profile:
        print_stats()