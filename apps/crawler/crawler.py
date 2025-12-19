import asyncio
import threading
import time

from bs4 import BeautifulSoup
from pool import NoUrlAvailableError, UrlPool
from profiler import profile
from url import AioHttpFetcher, Fetcher, filter_http_links


@profile
def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    return links


class Crawler:
    MAX_RETRIES = 3

    def __init__(self, seed_url: str, concurrency: int = 1, verbose: bool = False):
        self.seed_url = seed_url
        self.concurrency = concurrency
        self.verbose = verbose
        self.pool = UrlPool()


class SyncCrawler(Crawler):
    def __init__(self, seed_url: str, concurrency: int = 1, verbose: bool = False):
        super().__init__(seed_url, concurrency, verbose)
        self.fetcher = Fetcher(verbose=verbose)
        self.iteration_counter = [0, threading.Lock()]

    def crawl(self, num_urls: int) -> int:
        self.pool.add_url(self.seed_url)

        threads = []
        for i in range(self.concurrency):
            t = threading.Thread(target=self._worker, args=(i, num_urls))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return self.iteration_counter[0]

    def _worker(self, worker_id: int, max_iterations: int):
        retry_count = 0
        max_retries = self.MAX_RETRIES

        while True:
            with self.iteration_counter[1]:
                if self.iteration_counter[0] >= max_iterations:
                    if self.verbose:
                        print(f"[Worker {worker_id}] Max iterations reached, exiting")
                    break
                self.iteration_counter[0] += 1
                current_iter = self.iteration_counter[0]

            try:
                url_obj = self.pool.get()
                retry_count = 0

                if self.verbose:
                    print(f"[{current_iter}][Worker {worker_id}] Crawling: {url_obj.url}")

                content = self.fetcher.fetch(url_obj.url)
                if content is None:
                    self.pool.error(url_obj.id)
                else:
                    links = extract_links(content)
                    links = filter_http_links(links, url_obj.url)
                    self.pool.add_urls(links)
                    self.pool.done(url_obj.id)

            except NoUrlAvailableError:
                retry_count += 1
                if retry_count >= max_retries:
                    if self.verbose:
                        print(f"[Worker {worker_id}] No URLs available after {max_retries} retries, exiting")
                    break
                if self.verbose:
                    print(f"[Worker {worker_id}] No URLs available, retry {retry_count}/{max_retries}")
                time.sleep(1)


class AsyncCrawler(Crawler):
    def __init__(self, seed_url: str, concurrency: int = 1, verbose: bool = False):
        super().__init__(seed_url, concurrency, verbose)
        self.fetcher = None
        self.iteration_counter = [0, asyncio.Lock()]

    async def crawl(self, num_urls: int) -> int:
        self.pool.add_url(self.seed_url)

        async with AioHttpFetcher(verbose=self.verbose) as fetcher:
            self.fetcher = fetcher
            tasks = []
            for i in range(self.concurrency):
                task = asyncio.create_task(self._worker(i, num_urls))
                tasks.append(task)

            await asyncio.gather(*tasks)

        return self.iteration_counter[0]

    async def _worker(self, worker_id: int, max_iterations: int):
        retry_count = 0
        max_retries = self.MAX_RETRIES
        loop = asyncio.get_event_loop()

        while True:
            async with self.iteration_counter[1]:
                if self.iteration_counter[0] >= max_iterations:
                    if self.verbose:
                        print(f"[Worker {worker_id}] Max iterations reached, exiting")
                    break
                self.iteration_counter[0] += 1
                current_iter = self.iteration_counter[0]

            try:
                url_obj = self.pool.get()
                retry_count = 0

                if self.verbose:
                    print(f"[{current_iter}][Worker {worker_id}] Crawling: {url_obj.url}")

                content = await self.fetcher.fetch(url_obj.url)
                if content is None:
                    self.pool.error(url_obj.id)
                else:
                    # Run CPU-bound BeautifulSoup parsing in thread pool to avoid blocking event loop
                    links = await loop.run_in_executor(None, extract_links, content)
                    links = filter_http_links(links, url_obj.url)
                    self.pool.add_urls(links)
                    self.pool.done(url_obj.id)

            except NoUrlAvailableError:
                retry_count += 1
                if retry_count >= max_retries:
                    if self.verbose:
                        print(f"[Worker {worker_id}] No URLs available after {max_retries} retries, exiting")
                    break
                if self.verbose:
                    print(f"[Worker {worker_id}] No URLs available, retry {retry_count}/{max_retries}")
                await asyncio.sleep(1)
