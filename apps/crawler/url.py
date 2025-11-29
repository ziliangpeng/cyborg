from enum import Enum
from urllib.parse import urljoin, urlparse
import requests
from profiler import profile


CHROME_2020 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
CHROME_2025 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'

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
