# Web Crawler Types and Architecture Guide

## Overview

This document provides a comprehensive overview of different types of web crawlers, their architectures, use cases, and when to use each type.

## Types of Web Crawlers

### 1. General Purpose Crawler

**What it does:** Crawls any website, extracts whatever you configure it to extract.

**Architecture:**
- URL queue/frontier
- HTTP fetcher
- Configurable extractors (CSS selectors, XPath)
- Generic output (JSON, CSV)

**Good for:**
- One-off data collection tasks
- Multiple different sites with different structures
- Learning and experimentation
- When you control the extraction rules via config

**Limitations:**
- No domain-specific knowledge
- Requires manual configuration for each site
- No smart handling of site-specific quirks
- Basic rate limiting and politeness

---

### 2. Focused/Vertical Crawler

**What it does:** Specialized for a specific domain (e-commerce, news, academic papers, job listings).

**Architecture:**
- Domain-specific extractors (knows about products, articles, etc.)
- Structured output schemas (Product class, Article class)
- Smart URL prioritization (focus on relevant pages)
- Built-in knowledge of common site patterns

**Example:**
```python
# E-commerce focused crawler
class ProductCrawler:
    def extract_product(self, html):
        # Knows where to find: price, title, images, reviews
        # Handles: Amazon, eBay, Shopify patterns
        pass
```

**Good for:**
- When you're crawling many similar sites (all e-commerce, all news, etc.)
- When you need consistent data schema across sources
- Production systems with specific business needs

**Why not general purpose:**
- Too much manual config for each site
- Miss domain-specific optimizations
- No built-in knowledge of common patterns

---

### 3. Incremental/Refresh Crawler

**What it does:** Re-crawls sites to detect changes, only processes updated content.

**Architecture:**
- Stores checksums/hashes of previous crawls
- Compares current vs previous versions
- Only processes changed pages
- Scheduling based on update frequency

**Good for:**
- Monitoring price changes
- News aggregation (only new articles)
- Change detection systems
- Reducing bandwidth and processing

**Why not general purpose:**
- General crawlers re-fetch everything
- No state management across crawls
- Wasteful for frequently updated sites

---

### 4. Deep Web Crawler

**What it does:** Handles content behind forms, authentication, JavaScript-heavy sites.

**Architecture:**
- Form submission and interaction
- Session/cookie management
- JavaScript rendering (headless browsers)
- Authentication handling

**Good for:**
- Sites requiring login
- Search forms and databases
- Single Page Applications (SPAs)
- Dynamic content loaded via AJAX

**Why not general purpose:**
- General crawlers only handle static HTML
- Can't interact with forms or JavaScript
- Miss content behind authentication

---

### 5. Distributed/Parallel Crawler

**What it does:** Coordinates multiple crawler instances across machines.

**Architecture:**
- Centralized URL queue (Redis, RabbitMQ)
- Multiple worker processes/machines
- Deduplication across workers
- Load balancing

**Good for:**
- Large-scale crawling (millions of pages)
- Time-sensitive data collection
- High-throughput requirements

**Why not general purpose:**
- General crawlers are single-process
- Don't handle coordination between workers
- Can't scale beyond one machine

---

### 6. Politeness-Aware/Ethical Crawler

**What it does:** Strictly respects robots.txt, rate limits, crawl delays.

**Architecture:**
- robots.txt parser and enforcer
- Per-domain rate limiting
- Crawl delay from robots.txt
- User-agent identification
- Retry backoff strategies

**Good for:**
- Production crawling of third-party sites
- Avoiding IP blocks/bans
- Being a good web citizen
- Legal compliance

**Why not general purpose:**
- General crawlers have basic rate limiting
- May not fully respect robots.txt directives
- Less sophisticated politeness strategies

---

### 7. Archive/Snapshot Crawler

**What it does:** Preserves entire web pages with assets (images, CSS, JS) for historical record.

**Architecture:**
- Downloads all page assets
- Preserves directory structure
- Stores metadata (timestamp, headers)
- Creates WARC files (Web ARChive format)

**Good for:**
- Web archiving projects
- Compliance/legal record keeping
- Offline browsing
- Historical research

**Why not general purpose:**
- General crawlers extract data, not preserve pages
- Don't download assets
- No archival formats

---

### 8. Sitemap-Based Crawler

**What it does:** Uses sitemap.xml to discover URLs instead of following links.

**Architecture:**
- Parses sitemap.xml
- Respects priority and changefreq
- Much faster discovery
- No need to crawl navigation

**Good for:**
- Well-maintained sites with sitemaps
- Fast bulk data collection
- SEO-focused crawling

**Why not general purpose:**
- Not all sites have sitemaps
- Misses pages not in sitemap
- Less flexible

---

## Python Library Options for Web Crawling

### 1. Scrapy (Full Framework)

**What it is:** A complete web scraping framework with batteries included.

**Pros:**
- Built-in features: rate limiting, retry logic, concurrent requests, item pipelines
- Middleware system for customization
- Command-line tools for generating spiders
- Built-in support for exporting to JSON, CSV, XML
- Mature, well-documented, actively maintained

**Cons:**
- Steeper learning curve
- More complex project structure
- Overkill for simple crawling tasks
- Harder to integrate into existing codebases
- Async/Twisted-based (different paradigm)

**Best for:** Production crawling at scale, complex multi-site scraping, when you need middleware and pipelines.

---

### 2. requests + BeautifulSoup4 (Middle Ground - Most Popular)

**What it is:** HTTP library + HTML parser. The classic Python scraping combo.

**Pros:**
- Simple, intuitive API (everyone knows requests)
- BeautifulSoup handles malformed HTML well
- Easy to debug and understand
- Minimal boilerplate
- Flexible - you control everything
- Great documentation and community support

**Cons:**
- Synchronous (slower for many requests)
- No built-in rate limiting, retries (need to implement)
- BeautifulSoup is slower than lxml
- No built-in concurrency

**Best for:** Learning, prototyping, simple crawling, when you want full control with minimal setup.

---

### 3. httpx + lxml (Performance Focused)

**What it is:** Modern async HTTP client + fast XML/HTML parser.

**Pros:**
- httpx supports async/await for concurrent requests
- lxml is much faster than BeautifulSoup (C-based)
- Modern API, type hints
- Can use CSS selectors via lxml.cssselect

**Cons:**
- lxml is stricter about HTML (may fail on malformed HTML)
- Less forgiving parsing than BeautifulSoup
- Smaller community than requests
- More complex async code for beginners

**Best for:** Performance-critical crawling, when you need async, well-formed HTML sources.

---

### 4. Just requests (Minimal)

**What it is:** Only use requests library, parse JSON or regex HTML.

**Pros:**
- Single dependency
- Maximum simplicity
- Good for API crawling or structured data
- Fast and lightweight

**Cons:**
- Manual HTML parsing (regex is fragile)
- Not suitable for complex HTML extraction
- No structured data extraction

**Best for:** API crawling, JSON endpoints, very simple HTML extraction.

---

### 5. playwright / selenium (Browser Automation)

**What it is:** Control real browsers for JavaScript-heavy sites.

**Pros:**
- Handles JavaScript rendering
- Can interact with dynamic content
- Real browser environment

**Cons:**
- Much slower (spinning up browsers)
- Heavy resource usage
- Overkill for static HTML
- More complex setup

**Best for:** JavaScript-heavy SPAs, sites that require browser interaction.

---

## When to Use General Purpose vs Specialized Crawlers

### Use General Purpose Crawler When:

✅ You're crawling a few different sites with different structures
✅ You can configure extraction rules per site
✅ You don't need JavaScript rendering
✅ Scale is modest (thousands of pages, not millions)
✅ You want flexibility and simplicity
✅ This is for experimentation/learning

### Use Specialized Crawlers When:

❌ You're building a production system for a specific vertical (all e-commerce sites)
❌ You need to crawl JavaScript-heavy SPAs
❌ You're crawling millions of pages and need distribution
❌ You need to monitor changes over time (incremental)
❌ Content is behind authentication/forms
❌ You need to be very respectful of server resources at scale

---

## Recommendation for This Project

**Start with a general purpose crawler** because:

1. **You don't know your exact use case yet** - general purpose gives you flexibility
2. **MVP mindset** - get something working, learn what you actually need
3. **Easy to specialize later** - you can always add focused extractors, authentication, etc.
4. **Learning value** - understanding the basics helps you appreciate specialized crawlers

**Structure it so you can evolve:**
```
apps/crawler/
├── core/           # General purpose crawling logic
├── extractors/     # Easy to add focused extractors later
├── plugins/        # Extension points for auth, JS rendering, etc.
```

Then if you realize you need focused crawling for e-commerce sites, you add an `extractors/ecommerce.py` without rewriting everything.

---

## References and Further Reading

- [Scrapy Documentation](https://docs.scrapy.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [requests Documentation](https://requests.readthedocs.io/)
- [Web Crawling Best Practices](https://en.wikipedia.org/wiki/Web_crawler)
- [robots.txt Specification](https://www.robotstxt.org/)
