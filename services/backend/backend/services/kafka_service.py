"""
Kafka Service — Producer for scalable camera ingestion.

When USE_KAFKA=True, camera workers push raw events into Kafka topics:
  - raw_vehicle_events
  - traffic_snapshots
When USE_KAFKA=False, falls back to direct REST/API ingestion.
"""
from __future__ import annotations

import json
from typing import Any

from backend.config import get_settings

settings = get_settings()

_kafka_producer = None


def get_kafka_producer():
    """Get active Kafka producer instance if enabled, or None."""
    global _kafka_producer
    if not settings.USE_KAFKA:
        return None

    if _kafka_producer is None:
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                max_block_ms=2000,
            )
            _kafka_producer = producer
        except Exception as err:
            print(f"⚠️ Kafka producer init warning: {err}. Using fallback direct ingestion.")
            _kafka_producer = None

    return _kafka_producer


class KafkaService:
    @staticmethod
    def publish_event(event_data: dict[str, Any]) -> bool:
        """Publish a vehicle event payload to Kafka topic."""
        producer = get_kafka_producer()
        if not producer:
            return False

        try:
            producer.send(settings.KAFKA_TOPIC_EVENTS, value=event_data)
            producer.flush()
            return True
        except Exception as err:
            print(f"Kafka publish_event failed: {err}")
            return False

    @staticmethod
    def publish_snapshot(snapshot_data: dict[str, Any]) -> bool:
        """Publish a traffic snapshot payload to Kafka topic."""
        producer = get_kafka_producer()
        if not producer:
            return False

        try:
            producer.send(settings.KAFKA_TOPIC_SNAPSHOTS, value=snapshot_data)
            producer.flush()
            return True
        except Exception as err:
            print(f"Kafka publish_snapshot failed: {err}")
            return False


kafka_service = KafkaService()
