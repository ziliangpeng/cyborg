import requests
from bs4 import BeautifulSoup
from url import filter_http_links


def fetch_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
    return links


if __name__ == "__main__":
    url = "https://news.ycombinator.com"
    content = fetch_url(url)
    print(f"Fetched {len(content)} characters from {url}")
    links = extract_links(content)
    links = filter_http_links(links, url)
    print(f"Found {len(links)} links")
    for link in links:
        print(link)