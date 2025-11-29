from enum import Enum
from urllib.parse import urljoin, urlparse


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
