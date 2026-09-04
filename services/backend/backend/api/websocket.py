"""
WebSocket Router — Streams live alerts and traffic statistics.

Endpoints:
  WS /ws/alerts — Streams real-time alert events (blacklist matches, route anomalies)
  WS /ws/stats  — Streams per-camera live traffic statistics
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import get_settings
from backend.services.redis_service import get_redis

router = APIRouter(tags=["WebSockets"])
settings = get_settings()


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts."""
    await websocket.accept()
    r = get_redis()

    if not r:
        # Fallback if Redis is disabled or offline: send connected notification and keep socket alive
        await websocket.send_json({"type": "info", "message": "WebSocket connected (Redis offline — polling mode)"})
        try:
            while True:
                await asyncio.sleep(10)
        except WebSocketDisconnect:
            return

    pubsub = r.pubsub()
    pubsub.subscribe("alerts:live")

    try:
        await websocket.send_json({"type": "info", "message": "Subscribed to live alerts stream"})
        while True:
            # Poll Redis pubsub non-blockingly
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                data = json.loads(msg["data"])
                await websocket.send_json({"type": "alert", "data": data})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pubsub.unsubscribe("alerts:live")


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket endpoint for live traffic metrics."""
    await websocket.accept()
    r = get_redis()

    if not r:
        await websocket.send_json({"type": "info", "message": "WebSocket connected (Redis offline)"})
        try:
            while True:
                await asyncio.sleep(10)
        except WebSocketDisconnect:
            return

    pubsub = r.pubsub()
    pubsub.subscribe("stats:live")

    try:
        await websocket.send_json({"type": "info", "message": "Subscribed to live stats stream"})
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                data = json.loads(msg["data"])
                await websocket.send_json({"type": "stats", "data": data})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pubsub.unsubscribe("stats:live")
