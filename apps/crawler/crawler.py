import threading
import asyncio
import time
from bs4 import BeautifulSoup
from url import filter_http_links, Fetcher, AioHttpFetcher
from pool import UrlPool, NoUrlAvailableError
from profiler import profile


@profile
def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


class Crawler:
    def __init__(self, seed_url: str, workers: int = 1):
        self.seed_url = seed_url
        self.workers = workers
        self.pool = UrlPool()
        self.fetcher = Fetcher()
        self.iteration_counter = [0, threading.Lock()]

    def crawl(self, num_urls: int) -> int:
        self.pool.add_url(self.seed_url)

        threads = []
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, args=(i, num_urls))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return self.iteration_counter[0]

    def _worker(self, worker_id: int, max_iterations: int):
        retry_count = 0
        max_retries = 3

        while True:
            with self.iteration_counter[1]:
                if self.iteration_counter[0] >= max_iterations:
                    print(f"[Worker {worker_id}] Max iterations reached, exiting")
                    break
                self.iteration_counter[0] += 1
                current_iter = self.iteration_counter[0]

            try:
                url_obj = self.pool.get()
                retry_count = 0

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
                    print(f"[Worker {worker_id}] No URLs available after {max_retries} retries, exiting")
                    break
                print(f"[Worker {worker_id}] No URLs available, retry {retry_count}/{max_retries}")
                time.sleep(1)


class AsyncCrawler:
    def __init__(self, seed_url: str, concurrency: int = 1):
        self.seed_url = seed_url
        self.concurrency = concurrency
        self.pool = UrlPool()
        self.fetcher = None
        self.iteration_counter = [0, asyncio.Lock()]

    async def crawl(self, num_urls: int) -> int:
        self.pool.add_url(self.seed_url)

        async with AioHttpFetcher() as fetcher:
            self.fetcher = fetcher
            tasks = []
            for i in range(self.concurrency):
                task = asyncio.create_task(self._worker(i, num_urls))
                tasks.append(task)

            await asyncio.gather(*tasks)

        return self.iteration_counter[0]

    async def _worker(self, worker_id: int, max_iterations: int):
        retry_count = 0
        max_retries = 3

        while True:
            async with self.iteration_counter[1]:
                if self.iteration_counter[0] >= max_iterations:
                    print(f"[Worker {worker_id}] Max iterations reached, exiting")
                    break
                self.iteration_counter[0] += 1
                current_iter = self.iteration_counter[0]

            try:
                url_obj = self.pool.get()
                retry_count = 0

                print(f"[{current_iter}][Worker {worker_id}] Crawling: {url_obj.url}")

                content = await self.fetcher.fetch(url_obj.url)
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
                    print(f"[Worker {worker_id}] No URLs available after {max_retries} retries, exiting")
                    break
                print(f"[Worker {worker_id}] No URLs available, retry {retry_count}/{max_retries}")
                await asyncio.sleep(1)
