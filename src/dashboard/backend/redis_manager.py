"""
Redis Manager for FL Security Dashboard
Handles Redis pub/sub messaging between simulator and dashboard.
"""

import json
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis not available. Install with: pip install redis")


class RedisManager:
    """
    Manages Redis connections and pub/sub messaging for the dashboard.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Initialize Redis manager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")

        self.host = host
        self.port = port
        self.db = db
        self.password = password

        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to Redis server.

        Returns:
            True if connection successful
        """
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )

            # Test connection
            self.redis_client.ping()
            self.connected = True

            logging.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self.pubsub:
            self.pubsub.close()
            self.pubsub = None

        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None

        self.connected = False
        logging.info("Disconnected from Redis")

    def publish_training_round(self, round_data: Dict[str, Any]) -> None:
        """
        Publish training round update.

        Args:
            round_data: Training round data
        """
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return

        message = json.dumps({
            "type": "training_round",
            "data": round_data,
            "timestamp": datetime.now().isoformat()
        })

        self.redis_client.publish("fl:training", message)

    def publish_security_event(self, event: Dict[str, Any]) -> None:
        """
        Publish security event.

        Args:
            event: Security event data
        """
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return

        message = json.dumps({
            "type": "security_event",
            "data": event,
            "timestamp": datetime.now().isoformat()
        })

        self.redis_client.publish("fl:security", message)

    def publish_training_status(self, status: str) -> None:
        """
        Publish training status change.

        Args:
            status: Status (started, stopped, paused, resumed)
        """
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return

        message = json.dumps({
            "type": "training_status",
            "status": status,
            "timestamp": datetime.now().isoformat()
        })

        self.redis_client.publish("fl:status", message)

    def subscribe_to_training(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to training round updates.

        Args:
            callback: Function to call with each update
        """
        if not self.connected:
            logging.warning("Not connected to Redis")
            return

        if not self.pubsub:
            self.pubsub = self.redis_client.pubsub()

        self.pubsub.subscribe("fl:training")

        # Start listening in background
        def listen():
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        callback(data)
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON: {message['data']}")

        import threading
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def subscribe_to_security(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to security event updates.

        Args:
            callback: Function to call with each update
        """
        if not self.connected:
            logging.warning("Not connected to Redis")
            return

        if not self.pubsub:
            self.pubsub = self.redis_client.pubsub()

        self.pubsub.subscribe("fl:security")

        # Start listening in background
        def listen():
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        callback(data)
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON: {message['data']}")

        import threading
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def store_metrics(self, key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """
        Store metrics in Redis with TTL.

        Args:
            key: Storage key
            data: Data to store
            ttl: Time to live in seconds (default 1 hour)
        """
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return

        self.redis_client.setex(
            f"fl:metrics:{key}",
            ttl,
            json.dumps(data)
        )

    def retrieve_metrics(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metrics from Redis.

        Args:
            key: Storage key

        Returns:
            Retrieved data or None
        """
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return None

        data = self.redis_client.get(f"fl:metrics:{key}")

        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in key: {key}")
                return None

        return None

    def clear_all_metrics(self) -> None:
        """Clear all metrics from Redis."""
        if not self.connected or not self.redis_client:
            logging.warning("Not connected to Redis")
            return

        # Delete all keys with fl:metrics prefix
        for key in self.redis_client.scan_iter("fl:metrics:*"):
            self.redis_client.delete(key)

        logging.info("Cleared all metrics from Redis")


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


def get_redis_manager(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0
) -> Optional[RedisManager]:
    """
    Get or create Redis manager instance.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database number

    Returns:
        Redis manager instance or None if Redis not available
    """
    global _redis_manager

    if not REDIS_AVAILABLE:
        return None

    if _redis_manager is None:
        _redis_manager = RedisManager(host=host, port=port, db=db)
        _redis_manager.connect()

    return _redis_manager


def is_redis_available() -> bool:
    """Check if Redis is available."""
    return REDIS_AVAILABLE


if __name__ == "__main__":
    # Test Redis connection
    manager = get_redis_manager()
    if manager:
        print("Redis connection successful!")

        # Test publish
        manager.publish_training_status("test")

        # Test store/retrieve
        manager.store_metrics("test", {"round": 1, "accuracy": 0.9})
        data = manager.retrieve_metrics("test")
        print(f"Retrieved: {data}")

        manager.disconnect()
    else:
        print("Redis not available")
