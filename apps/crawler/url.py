from enum import Enum
from urllib.parse import urljoin, urlparse
import requests
import aiohttp
import httpx
from profiler import profile


# User-Agent strings for Chrome from 2020-2025
# Only macOS version (e.g., 10_15_7) and Chrome version (e.g., 87.0.0.0) change by year
# Mozilla/5.0, AppleWebKit/537.36, Safari/537.36 are frozen compatibility tokens
CHROME_2020 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.0.0 Safari/537.36'
CHROME_2021 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.0.0 Safari/537.36'
CHROME_2022 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
CHROME_2023 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
CHROME_2024 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
CHROME_2025 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'

IGNORED_SUFFIXES = ['.pdf', '.parquet']


class Fetcher:
    def __init__(self):
        self._headers = {
            'User-Agent': CHROME_2025
        }
        self._timeout = 5

    @profile
    def fetch(self, url: str) -> str | None:
        for suffix in IGNORED_SUFFIXES:
            if url.endswith(suffix):
                return None
        try:
            response = requests.get(url, headers=self._headers, timeout=self._timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"  -> ❌ Error fetching: {e}")
            return None


class AioHttpFetcher:
    def __init__(self):
        self._headers = {
            'User-Agent': CHROME_2025
        }
        self._timeout = aiohttp.ClientTimeout(total=5)
        self._session = None

    async def __aenter__(self):
        # Configure connector for high concurrency (hundreds of workers, thousands of sites)
        connector = aiohttp.TCPConnector(
            limit=5000,              # Total connections across all hosts
            limit_per_host=2,        # Max connections per individual host (polite crawling)
            ttl_dns_cache=300        # Cache DNS for 5 minutes
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers=self._headers,
            timeout=self._timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()

    @profile
    async def fetch(self, url: str) -> str | None:
        for suffix in IGNORED_SUFFIXES:
            if url.endswith(suffix):
                return None
        try:
            async with self._session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            print(f"  -> ❌ Error fetching: {e}")
            return None


class HttpxFetcher:
    def __init__(self):
        self._headers = {
            'User-Agent': CHROME_2025
        }
        self._timeout = 5.0
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    @profile
    async def fetch(self, url: str) -> str | None:
        for suffix in IGNORED_SUFFIXES:
            if url.endswith(suffix):
                return None
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"  -> ❌ Error fetching: {e}")
            return None


def filter_http_links(links: list[str], base_url: str) -> list[str]:
    class LinkType(Enum):
        ABSOLUTE = 1
        EMAIL = 2
        ANCHOR = 3
        JAVASCRIPT = 4
        PHONE = 5
        DATA_URI = 6
        PROTOCOL_RELATIVE = 7
        FILE = 8
        OTHER_PROTOCOL = 9
        EMPTY = 10
        RELATIVE = 11

    valid = []
    for link in links:
        link = link.strip()

        if link.startswith('http://') or link.startswith('https://'):
            category = LinkType.ABSOLUTE
        elif link.startswith('mailto:'):
            category = LinkType.EMAIL
        elif link.startswith('#'):
            category = LinkType.ANCHOR
        elif link.startswith('javascript:'):
            category = LinkType.JAVASCRIPT
        elif link.startswith('tel:'):
            category = LinkType.PHONE
        elif link.startswith('data:'):
            category = LinkType.DATA_URI
        elif link.startswith('//'):
            category = LinkType.PROTOCOL_RELATIVE
        elif link.startswith('file://'):
            category = LinkType.FILE
        elif link.startswith(('ftp://', 'magnet:', 'blob:', 'about:', 'chrome://')):
            category = LinkType.OTHER_PROTOCOL
        elif link == '':
            category = LinkType.EMPTY
        else:
            category = LinkType.RELATIVE

        if category == LinkType.ABSOLUTE:
            valid.append(link)
        elif category == LinkType.RELATIVE:
            absolute_link = urljoin(base_url, link)
            valid.append(absolute_link)
        elif category == LinkType.PROTOCOL_RELATIVE:
            parsed = urlparse(base_url)
            absolute_link = f"{parsed.scheme}:{link}"
            valid.append(absolute_link)
    return valid
