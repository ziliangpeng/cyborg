import requests
from bs4 import BeautifulSoup
from url import filter_http_links
from pool import UrlPool, NoUrlAvailableError


def fetch_url(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


if __name__ == "__main__":
    pool = UrlPool()
    seed_url = "https://news.ycombinator.com"
    pool.add_url(seed_url)

    for i in range(100):
        try:
            url_obj = pool.get()
            print(f"[{i}] Crawling: {url_obj.url}")

            content = fetch_url(url_obj.url)
            links = extract_links(content)
            links = filter_http_links(links, url_obj.url)

            pool.add_urls(links)
            pool.done(url_obj.id)

            print(f"  -> Found {len(links)} links")
        except NoUrlAvailableError:
            print(f"[{i}] No URLs available (rate limited)")
            break