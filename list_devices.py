import asyncio
from nio import AsyncClient, AsyncClientConfig, LoginResponse
import getpass

HOMESERVER = "https://matrix.cf-hub.net"
USER_ID = "@hub-bot:matrix.cf-hub.net"
DEVICE_NAME = "hub-bot"
STORE_PATH = "data/matrix_store"

async def main():
    password = getpass.getpass(f"Enter password for {USER_ID}: ")
    client = AsyncClient(
        HOMESERVER,
        USER_ID,
        store_path=STORE_PATH,
        config=AsyncClientConfig(encryption_enabled=True)
    )
    resp = await client.login(password, device_name=DEVICE_NAME)
    if isinstance(resp, LoginResponse):
        print("Logged in as", USER_ID)
    else:
        print("Login failed:", resp)
        return

    print("Fetching device list...")
    devices_resp = await client.devices()
    if hasattr(devices_resp, 'devices'):
        print(f"\nDevices for {USER_ID}:")
        for dev in devices_resp.devices:
            print(f"- Device ID: {dev.device_id}")
            print(f"  Display Name: {dev.display_name}")
            print(f"  Last Seen IP: {dev.last_seen_ip}")
            print(f"  Last Seen: {dev.last_seen_ts}")
            print(f"  Is Current Device: {dev.device_id == client.device_id}")
            print()
    else:
        print("Failed to fetch devices:", devices_resp)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
