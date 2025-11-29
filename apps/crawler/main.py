import requests
from bs4 import BeautifulSoup
from url import filter_http_links
from pool import UrlPool, NoUrlAvailableError
import argparse


def fetch_url(url: str) -> str | None:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  -> ❌ Error fetching: {e}")
        return None


def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple web crawler")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of iterations (default: 100)")
    args = parser.parse_args()

    pool = UrlPool()
    seed_url = "https://news.ycombinator.com"
    pool.add_url(seed_url)

    for i in range(args.num):
        try:
            url_obj = pool.get()
            print(f"[{i}] Crawling: {url_obj.url}")

            content = fetch_url(url_obj.url)
            if content is None:
                pool.error(url_obj.id)
            else:
                links = extract_links(content)
                links = filter_http_links(links, url_obj.url)
                pool.add_urls(links)
                print(f"  -> Found {len(links)} links")

                pool.done(url_obj.id)

        except NoUrlAvailableError:
            print(f"[{i}] No URLs available (rate limited)")
            break