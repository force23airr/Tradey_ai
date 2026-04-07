"""Base HTTP client for Polymarket APIs with retry logic and rate limiting."""

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.settings import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF, RATE_LIMIT_DELAY


class PolymarketClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._last_request_time = 0.0
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",  # disable brotli to avoid decode errors
        })
        return session

    def _throttle(self):
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    def get(self, endpoint: str, params: dict = None) -> dict | list:
        self._throttle()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
