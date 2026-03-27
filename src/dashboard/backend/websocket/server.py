"""
WebSocket Server for FL Security Dashboard
Provides real-time updates to dashboard clients.
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLDashboardWebSocketServer:
    """
    WebSocket server for real-time FL training updates.
    Broadcasts training progress to connected dashboard clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        self.running = False

    async def register_client(self, websocket: WebSocketServerProtocol) -> None:
        """Register a new client connection."""
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")

        # Send welcome message
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "Connected to FL Security Dashboard",
            "timestamp": datetime.now().isoformat()
        }))

    async def unregister_client(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister a client connection."""
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    async def broadcast_update(self, update: Dict[str, Any]) -> None:
        """
        Broadcast update to all connected clients.

        Args:
            update: Update data to broadcast
        """
        if not self.connected_clients:
            return

        message = json.dumps(update)

        # Send to all clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)

    async def broadcast_training_round(self, round_data: Dict[str, Any]) -> None:
        """
        Broadcast training round update.

        Args:
            round_data: Training round data
        """
        await self.broadcast_update({
            "type": "training_round",
            "data": round_data,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_security_event(self, event: Dict[str, Any]) -> None:
        """
        Broadcast security event.

        Args:
            event: Security event data
        """
        await self.broadcast_update({
            "type": "security_event",
            "data": event,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_training_status(self, status: str) -> None:
        """
        Broadcast training status change.

        Args:
            status: Status (started, stopped, paused, resumed)
        """
        await self.broadcast_update({
            "type": "training_status",
            "status": status,
            "timestamp": datetime.now().isoformat()
        })

    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """
        Handle incoming message from client.

        Args:
            websocket: Client websocket connection
            message: Message from client
        """
        try:
            data = json.loads(message)

            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif message_type == "subscribe":
                # Client wants to subscribe to specific updates
                logger.info(f"Client subscribed: {data.get('channel', 'all')}")

            elif message_type == "get_status":
                # Send current server status
                await websocket.send(json.dumps({
                    "type": "status",
                    "connected_clients": len(self.connected_clients),
                    "server_running": self.running
                }))

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client: {message}")

    async def client_handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Handle client websocket connection.

        Args:
            websocket: Client websocket connection
            path: WebSocket URL path
        """
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by client")

        finally:
            await self.unregister_client(websocket)

    async def start(self) -> None:
        """Start the WebSocket server."""
        self.running = True
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        async with websockets.serve(
            self.client_handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20
        ):
            logger.info("WebSocket server running")
            await asyncio.Future()  # Run forever

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self.running = False
        logger.info("WebSocket server stopped")


# Global server instance
_server_instance: Optional[FLDashboardWebSocketServer] = None


def get_server(host: str = "localhost", port: int = 8765) -> FLDashboardWebSocketServer:
    """
    Get or create WebSocket server instance.

    Args:
        host: Server host
        port: Server port

    Returns:
        WebSocket server instance
    """
    global _server_instance

    if _server_instance is None:
        _server_instance = FLDashboardWebSocketServer(host, port)

    return _server_instance


async def run_server(host: str = "localhost", port: int = 8765) -> None:
    """
    Run WebSocket server (async entry point).

    Args:
        host: Server host
        port: Server port
    """
    server = get_server(host, port)
    await server.start()


def run_server_sync(host: str = "localhost", port: int = 8765) -> None:
    """
    Run WebSocket server (sync entry point).

    Args:
        host: Server host
        port: Server port
    """
    asyncio.run(run_server(host, port))


if __name__ == "__main__":
    run_server_sync()
