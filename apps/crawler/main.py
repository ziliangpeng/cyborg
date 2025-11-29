import requests


def fetch_url(url: str) -> str:
    """
    Fetch content from a given URL.

    Args:
        url: The URL to fetch

    Returns:
        The HTML content as a string
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for bad status codes
    return response.text


if __name__ == "__main__":
    # Example usage
    url = "https://news.ycombinator.com"
    content = fetch_url(url)
    print(f"Fetched {len(content)} characters from {url}")
