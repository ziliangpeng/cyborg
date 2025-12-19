import random
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse


class NoUrlAvailableError(Exception):
    pass


@dataclass
class Url:
    url: str
    id: int


class UrlPool:
    def __init__(self, requests_per_minute: int = 6):
        self._next_id = 0
        self._pending = {}
        self._url_to_id = {}
        self._done_urls = set()
        self._done_times = {}
        self._host_last_crawl = {}
        self._rate_limit = 60 / requests_per_minute
        self._lock = threading.Lock()
        self._inflight = {}
        self._inflight_timeout = 60

    def add_url(self, url: str) -> int | None:
        with self._lock:
            if url in self._url_to_id:
                return None

            url_id = self._next_id
            self._next_id += 1
            self._pending[url_id] = url
            self._url_to_id[url] = url_id
            return url_id

    def add_urls(self, urls: list[str]) -> list[int]:
        ids = []
        for url in urls:
            url_id = self.add_url(url)
            if url_id is not None:
                ids.append(url_id)
        return ids

    def error(self, url_id: int):
        with self._lock:
            if url_id in self._inflight:
                del self._inflight[url_id]
            # TODO: Better error handling - track errors, retry with backoff, categorize error types
            if url_id in self._pending:
                del self._pending[url_id]

    def done(self, url_id: int):
        with self._lock:
            if url_id in self._inflight:
                del self._inflight[url_id]

            if url_id not in self._pending:
                return

            url = self._pending[url_id]
            del self._pending[url_id]
            self._done_urls.add(url_id)
            self._done_times[url_id] = time.time()

            host = urlparse(url).netloc
            self._host_last_crawl[host] = time.time()

    def get(self) -> Url:
        with self._lock:
            current_time = time.time()

            # Expire old inflight URLs
            expired_ids = []
            for url_id, inflight_time in self._inflight.items():
                if current_time - inflight_time > self._inflight_timeout:
                    expired_ids.append(url_id)

            for url_id in expired_ids:
                del self._inflight[url_id]

            # TODO: a few things we can do
            #   - reuse the available [] so we do not need to recompute every time
            #   - somehow randomize the _pending.items() so we don't need to return some error url that's in the front of queue
            #   - error() handling need to be better
            available = []
            for url_id, url in self._pending.items():
                if url_id in self._inflight:
                    continue

                host = urlparse(url).netloc

                if host in self._host_last_crawl:
                    time_since_last = current_time - self._host_last_crawl[host]
                    if time_since_last < self._rate_limit:
                        continue

                available.append((url_id, url))
                if len(available) >= 64:
                    break

            if not available:
                raise NoUrlAvailableError("All URLs are rate-limited or inflight")

            url_id, url = random.choice(available)
            self._inflight[url_id] = current_time
            host = urlparse(url).netloc
            self._host_last_crawl[host] = current_time
            return Url(url=url, id=url_id)
