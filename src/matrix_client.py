"""
src/matrix_client.py
====================
Thin async wrapper around ``matrix-nio`` that handles login, room management,
and sending / receiving messages.

Usage
-----
    from src.matrix_client import MatrixClient
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    client = MatrixClient(cfg.matrix)

    async with client:
        rooms = await client.get_joined_rooms()
        await client.send_message(room_id, "Hello!")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Coroutine, List, Optional

from nio import (  # type: ignore[import-untyped]
    AsyncClient,
    AsyncClientConfig,
    LoginError,
    MatrixRoom,
    RoomMessageText,
    SyncError,
)

from src.config_manager import MatrixConfig

logger = logging.getLogger(__name__)


class MatrixClient:
    """Async Matrix client with login, sync, and message helpers.

    Parameters
    ----------
    config:
        The ``matrix`` section of the bot configuration.
    message_callback:
        An async function ``(room, event) -> None`` that is called for every
        incoming text message.  Wire this up to the bot's message handler.
    """

    def __init__(
        self,
        config: MatrixConfig,
        message_callback: Optional[
            Callable[[MatrixRoom, RoomMessageText], Coroutine]
        ] = None,
    ) -> None:
        self._cfg = config
        self._message_callback = message_callback

        store_path = Path(config.store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        client_config = AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
            store_sync_tokens=True,
            encryption_enabled=True,  # Enable E2E encryption by default
        )

        self._client = AsyncClient(
            homeserver=config.homeserver,
            user=config.username,
            store_path=str(store_path),
            config=client_config,
        )


        # Register handlers for plaintext and encrypted events
        if message_callback:
            self._client.add_event_callback(self._on_message, RoomMessageText)
            # Handle encrypted events (MegolmEvent = m.room.encrypted)
            try:
                from nio.events.room_events import MegolmEvent
                self._client.add_event_callback(self._on_encrypted_message, MegolmEvent)
            except ImportError:
                # Fallback: register for all events of type "m.room.encrypted"
                self._client.add_event_callback(self._on_encrypted_message, "m.room.encrypted")

        # Register interactive verification handler (use event class, not string)
        try:
            from nio.events import KeyVerificationStart
            self._client.add_event_callback(self._on_verification_event, KeyVerificationStart)
        except ImportError:
            logger.warning("Could not import KeyVerificationStart for verification handler.")

    async def _on_verification_event(self, room, event):
        """Handle interactive verification requests (SAS/emoji)."""
        try:
            from nio.events import KeyVerificationStart
            from nio.responses import ToDeviceError
            if not isinstance(event, KeyVerificationStart):
                return
            logger.info("Received verification request from %s", event.sender)
            # Accept the verification (SAS/emoji)
            resp = await self._client.accept_key_verification(event.sender, event.transaction_id)
            if isinstance(resp, ToDeviceError):
                logger.error("Failed to accept verification: %s", resp.message)
                return
            logger.info("Accepted verification request. Waiting for SAS/emoji confirmation…")
            # Listen for the next steps (SAS/emoji)
            # In a real bot, you would show the emoji/SAS to the user for manual confirmation
            # For simplicity, we auto-confirm here
            await asyncio.sleep(2)
            await self._client.confirm_short_auth_string(event.sender, event.transaction_id)
            logger.info("Confirmed SAS/emoji verification with %s", event.sender)
        except Exception:
            logger.exception("Failed to handle verification event.")

    async def _on_encrypted_message(self, room, event):
        """Handle encrypted messages by passing to the main handler if decrypted."""
        try:
            # MegolmEvent does not have a decrypt method; nio handles decryption automatically if possible
            if hasattr(event, 'decrypted') and event.decrypted and hasattr(event, 'body') and event.body:
                from nio.events.room_events import RoomMessageText
                fake_event = RoomMessageText(
                    source=event.source,
                    decrypted=True,
                    sender=event.sender,
                    server_timestamp=event.server_timestamp,
                    event_id=event.event_id,
                    room_id=getattr(event, 'room_id', None),
                    body=event.body,
                    msgtype=getattr(event, 'msgtype', 'm.text'),
                )
                await self._on_message(room, fake_event)
            else:
                logger.warning("Encrypted event could not be decrypted or has no body. (decrypted=%s, body=%s)", getattr(event, 'decrypted', None), getattr(event, 'body', None))
        except Exception:
            logger.exception("Failed to handle encrypted message.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MatrixClient":
        await self.login()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Login / logout
    # ------------------------------------------------------------------

    async def login(self) -> None:
        """Log in to the Matrix homeserver."""
        logger.info("Logging in as %s …", self._cfg.username)
        response = await self._client.login(
            password=self._cfg.password,
            device_name=self._cfg.device_name,
        )
        if isinstance(response, LoginError):
            raise RuntimeError(
                f"Matrix login failed: {response.message} (status {response.status_code})"
            )
        logger.info("Logged in successfully.  Access token obtained.")

    async def close(self) -> None:
        """Log out and close the client session."""
        try:
            await self._client.logout()
        except Exception:
            pass
        await self._client.close()

    # ------------------------------------------------------------------
    # Room helpers
    # ------------------------------------------------------------------

    async def get_joined_rooms(self) -> List[str]:
        """Return a list of room IDs the bot is currently a member of."""
        response = await self._client.joined_rooms()
        return getattr(response, "rooms", [])

    def is_room_allowed(self, room_id: str, room: Optional[MatrixRoom] = None) -> bool:
        """Return True if the bot is allowed to interact in this room.

        If ``allowed_rooms`` is empty in the config, all rooms are allowed.
        """
        allowed = self._cfg.allowed_rooms
        if not allowed:
            return True
        # Match against room ID or canonical alias
        aliases = []
        if room and hasattr(room, "canonical_alias") and room.canonical_alias:
            aliases.append(room.canonical_alias)
        return room_id in allowed or any(a in allowed for a in aliases)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(self, room_id: str, text: str) -> None:
        """Send a plain-text message to the given room.

        Parameters
        ----------
        room_id:
            The room's internal ID (``!abc123:example.org``).
        text:
            The message body to send.
        """
        await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": text},
        )
        logger.debug("Sent message to %s: %s", room_id, text[:80])

    async def set_display_name(self, display_name: str) -> None:
        """Update the bot's Matrix display name."""
        await self._client.set_displayname(display_name)
        logger.info("Display name set to %r", display_name)

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def sync_forever(self, timeout_ms: int = 30_000) -> None:
        """Run the Matrix sync loop indefinitely.

        This is the main event loop that receives incoming messages and
        delivers them to the registered callback.

        Parameters
        ----------
        timeout_ms:
            Long-poll timeout in milliseconds.
        """
        logger.info("Starting Matrix sync loop …")
        # First sync — catch up without processing old messages
        await self._client.sync(timeout=timeout_ms, full_state=True)
        logger.info("Initial sync complete.  Listening for messages …")

        while True:
            try:
                response = await self._client.sync(
                    timeout=timeout_ms,
                    full_state=False,
                )
                if isinstance(response, SyncError):
                    logger.warning("Sync error: %s — retrying in 5s …", response.message)
                    await asyncio.sleep(5)
            except Exception:
                logger.exception("Unexpected error in sync loop — retrying in 10s …")
                await asyncio.sleep(10)

    # ------------------------------------------------------------------
    # Internal event handler
    # ------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Dispatch incoming messages to the registered callback."""
        # Ignore messages sent by the bot itself
        if event.sender == self._cfg.username:
            return

        if not self.is_room_allowed(room.room_id, room):
            return

        if self._message_callback:
            try:
                await self._message_callback(room, event)
            except Exception:
                logger.exception(
                    "Error in message callback for room %s", room.room_id
                )
