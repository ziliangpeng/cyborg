import pytest


@pytest.fixture
def sample_html_with_links():
    return '''<html><body>
        <a href="http://example.com/page1">Link 1</a>
        <a href="/relative">Link 2</a>
        <a href="mailto:test@test.com">Email</a>
    </body></html>'''


@pytest.fixture
def empty_html():
    return '<html><body></body></html>'
