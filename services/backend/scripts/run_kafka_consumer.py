"""
Standalone launcher for the Kafka Consumer service.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.services.kafka_consumer import run_kafka_consumer_loop

if __name__ == "__main__":
    run_kafka_consumer_loop()
