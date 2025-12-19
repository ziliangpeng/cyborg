import pytest
import responses
from aioresponses import aioresponses
from crawler import AsyncCrawler, SyncCrawler, extract_links
from pool import UrlPool
from url import AioHttpFetcher, Fetcher, filter_http_links


# Test 1: Sync Fetcher
@responses.activate
def test_fetcher_can_fetch():
    responses.add(responses.GET, "http://example.com", body="<html><body>Test</body></html>", status=200)

    fetcher = Fetcher()
    result = fetcher.fetch("http://example.com")

    assert result == "<html><body>Test</body></html>"


# Test 2: Async AioHttpFetcher
@pytest.mark.asyncio
async def test_aiohttpfetcher_can_fetch():
    with aioresponses() as m:
        m.get("http://example.com", body="<html>Test</html>")

        async with AioHttpFetcher() as fetcher:
            result = await fetcher.fetch("http://example.com")

        assert result == "<html>Test</html>"


# Test 3: Basic link filtering
def test_filter_http_links_basic():
    links = [
        "http://example.com/page1",
        "https://example.com/page2",
        "mailto:test@test.com",
        "javascript:void(0)",
        "/relative",
    ]
    base_url = "http://example.com"

    result = filter_http_links(links, base_url)

    # Should return HTTP/HTTPS URLs and converted relative URL
    assert "http://example.com/page1" in result
    assert "https://example.com/page2" in result
    assert "http://example.com/relative" in result
    assert "mailto:test@test.com" not in result
    assert "javascript:void(0)" not in result


# Test 4: Relative to absolute URL conversion
def test_filter_http_links_relative_to_absolute():
    links = ["/path", "/another/path"]
    base_url = "http://example.com"

    result = filter_http_links(links, base_url)

    assert "http://example.com/path" in result
    assert "http://example.com/another/path" in result


# Test 5: URL pool add and get
def test_url_pool_add_and_get():
    pool = UrlPool()

    # Use different hosts to avoid rate limiting
    pool.add_url("http://example1.com")
    pool.add_url("http://example2.com")

    url1 = pool.get()
    url2 = pool.get()

    assert url1.url in ["http://example1.com", "http://example2.com"]
    assert url2.url in ["http://example1.com", "http://example2.com"]
    assert url1.url != url2.url


# Test 6: URL pool deduplication
def test_url_pool_deduplication():
    pool = UrlPool()

    url_id1 = pool.add_url("http://example.com")
    url_id2 = pool.add_url("http://example.com")

    assert url_id1 is not None
    assert url_id2 is None


# Test 7: Extract links from HTML
def test_extract_links():
    html = """<html><body>
        <a href="http://example.com/page1">Link 1</a>
        <a href="http://example.com/page2">Link 2</a>
        <a href="/relative">Link 3</a>
    </body></html>"""

    links = extract_links(html)

    assert len(links) == 3
    assert "http://example.com/page1" in links
    assert "http://example.com/page2" in links
    assert "/relative" in links


# Test 8: SyncCrawler end-to-end
@responses.activate
def test_synccrawler_can_crawl():
    # Mock seed URL with one link
    responses.add(
        responses.GET, "http://example.com", body='<html><a href="http://example.com/page2">Link</a></html>', status=200
    )

    # Mock second URL
    responses.add(responses.GET, "http://example.com/page2", body="<html></html>", status=200)

    crawler = SyncCrawler("http://example.com", concurrency=1)
    total = crawler.crawl(num_urls=2)

    assert total == 2


# Test 9: AsyncCrawler end-to-end
@pytest.mark.asyncio
async def test_asynccrawler_can_crawl():
    with aioresponses() as m:
        # Mock seed URL with one link
        m.get("http://example.com", body='<html><a href="http://example.com/page2">Link</a></html>')

        # Mock second URL
        m.get("http://example.com/page2", body="<html></html>")

        crawler = AsyncCrawler("http://example.com", concurrency=1)
        total = await crawler.crawl(num_urls=2)

        assert total == 2


# Test 10: Crawler with multiple links
@responses.activate
def test_crawler_with_multiple_links():
    # Mock seed URL with 3 links
    responses.add(
        responses.GET,
        "http://example.com",
        body="""<html>
                      <a href="http://example.com/page1">Link 1</a>
                      <a href="http://example.com/page2">Link 2</a>
                      <a href="http://example.com/page3">Link 3</a>
                  </html>""",
        status=200,
    )

    # Mock the 3 linked pages
    for i in range(1, 4):
        responses.add(responses.GET, f"http://example.com/page{i}", body="<html></html>", status=200)

    crawler = SyncCrawler("http://example.com", concurrency=1)
    total = crawler.crawl(num_urls=4)

    assert total == 4
