"""
WebSocket Router — Streams live alerts and traffic statistics.

Endpoints:
  WS /ws/alerts — Streams real-time alert events (blacklist matches, route anomalies)
  WS /ws/stats  — Streams per-camera live traffic statistics

Both streams are pubsub relays, and both are driven by the *async* Redis
client. That is not a style choice: the synchronous client's `get_message`
blocks the calling thread for its whole timeout, and these handlers run on the
event loop, so a single connected dashboard was enough to stall every other
request the worker had — the API looked unreachable while the process sat
healthy.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import get_settings
from backend.services.redis_service import get_async_redis

router = APIRouter(tags=["WebSockets"])
settings = get_settings()


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """Return once the client goes away.

    Nothing is ever *expected* from the browser on these streams, but the
    disconnect only surfaces as a received frame. Without something reading it,
    a handler learns the tab is gone only when it next sends — which for a
    quiet channel may be never, so every page reload would leak a handler that
    polls Redis forever.
    """
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


async def _stream_channel(websocket: WebSocket, channel: str, message_type: str) -> None:
    """Relay a Redis pubsub channel to a WebSocket until either end hangs up."""
    await websocket.accept()
    client = await get_async_redis()

    if client is None:
        await websocket.send_json(
            {"type": "info", "message": "WebSocket connected (Redis offline — polling mode)"}
        )
        await _wait_for_disconnect(websocket)
        return

    pubsub = client.pubsub()
    await pubsub.subscribe(channel)
    disconnected = asyncio.create_task(_wait_for_disconnect(websocket))

    try:
        await websocket.send_json(
            {"type": "info", "message": f"Subscribed to live {message_type} stream"}
        )
        while not disconnected.done():
            # Awaited, so the loop is free to serve everything else while this
            # connection waits for its next message.
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                await websocket.send_json(
                    {"type": message_type, "data": json.loads(msg["data"])}
                )
    except WebSocketDisconnect:
        pass
    finally:
        disconnected.cancel()
        await pubsub.aclose()


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts."""
    await _stream_channel(websocket, "alerts:live", "alert")


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket endpoint for live traffic metrics."""
    await _stream_channel(websocket, "stats:live", "stats")
