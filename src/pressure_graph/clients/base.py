from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class HttpClientConfig:
    base_url: str
    timeout: float = 30.0
    max_retries: int = 4
    retry_sleep_seconds: float = 0.5


class RestClient:
    def __init__(self, config: HttpClientConfig) -> None:
        self.config = config
        self._client = httpx.Client(base_url=config.base_url, timeout=config.timeout)

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("retCode") not in (None, 0):
                    raise RuntimeError(f"API error {payload.get('retCode')}: {payload.get('retMsg')}")
                return payload
            except Exception as exc:  # pragma: no cover - exercised only on live network failures
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(self.config.retry_sleep_seconds * (2**attempt))
        raise RuntimeError(f"GET {path} failed after retries") from last_error
