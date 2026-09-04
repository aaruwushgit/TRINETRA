"""
Redis Service — Caching & Real-time Pub/Sub for WebSockets.

Handles:
  1. Caching analytics endpoints (summary, heatmap, density).
  2. Publishing live alerts to the 'alerts:live' channel.
  3. Publishing traffic updates to the 'stats:live' channel.
Fallback gracefully if Redis is offline or disabled.
"""
from __future__ import annotations

import json
from typing import Any

from backend.config import get_settings

settings = get_settings()

_redis_client = None


def get_redis():
    """Get connected Redis instance if enabled, or None if disabled/unreachable."""
    global _redis_client
    if not settings.USE_REDIS:
        return None

    if _redis_client is None:
        try:
            import redis
            client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            _redis_client = client
        except Exception as err:
            print(f"⚠️ Redis connection warning: {err}. Caching disabled.")
            _redis_client = None

    return _redis_client


# In-process fallback cache.
#
# The analytics endpoints aggregate over the whole event table, which on a
# city-scale database (12M+ rows) costs seconds per call — and the dashboard
# polls them every few seconds. Those endpoints already ask for a cached value,
# but with USE_REDIS=False that was a silent no-op, so every poll paid full
# price. A tiny TTL dict keeps the dashboard responsive with no external
# dependency. It is per-process (so it does not survive a reload and is not
# shared across workers), which is the correct trade for short TTLs on
# read-only aggregates.
_local_cache: dict[str, tuple[float, Any]] = {}
_LOCAL_CACHE_MAX = 256


class RedisService:
    @staticmethod
    def publish_alert(alert_data: dict[str, Any]) -> None:
        """Publish alert JSON payload to Redis channel 'alerts:live'."""
        r = get_redis()
        if r:
            try:
                r.publish("alerts:live", json.dumps(alert_data, default=str))
            except Exception as err:
                print(f"Failed to publish alert to Redis: {err}")

    @staticmethod
    def publish_stats(stats_data: dict[str, Any]) -> None:
        """Publish traffic stats JSON payload to Redis channel 'stats:live'."""
        r = get_redis()
        if r:
            try:
                r.publish("stats:live", json.dumps(stats_data, default=str))
            except Exception as err:
                print(f"Failed to publish stats to Redis: {err}")

    @staticmethod
    def cache_get(key: str) -> Any | None:
        r = get_redis()
        if not r:
            import time

            entry = _local_cache.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                _local_cache.pop(key, None)
                return None
            return value
        try:
            val = r.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    @staticmethod
    def cache_set(key: str, value: Any, ttl_seconds: int = 30) -> None:
        r = get_redis()
        if not r:
            import time

            if len(_local_cache) >= _LOCAL_CACHE_MAX:
                # Cheap bound: drop whatever has expired, else clear. These are
                # regenerable aggregates, so evicting too much is harmless.
                now = time.time()
                for k in [k for k, (exp, _) in _local_cache.items() if exp < now]:
                    _local_cache.pop(k, None)
                if len(_local_cache) >= _LOCAL_CACHE_MAX:
                    _local_cache.clear()
            _local_cache[key] = (time.time() + ttl_seconds, value)
            return
        try:
            r.setex(key, ttl_seconds, json.dumps(value, default=str))
        except Exception:
            pass


redis_service = RedisService()
