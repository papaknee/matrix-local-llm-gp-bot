import asyncio
from nio import AsyncClient, AsyncClientConfig, LoginResponse, KeyVerificationStart, KeyVerificationAccept, KeyVerificationKey, KeyVerificationMac
import getpass

HOMESERVER = "https://matrix.cf-hub.net"
USER_ID = "@hub-bot:matrix.cf-hub.net"
DEVICE_NAME = "hub-bot"
STORE_PATH = "data/matrix_store"

async def main():
    password = getpass.getpass("Enter password for {}: ".format(USER_ID))
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

    print("Now, from your Element client, go to Settings > Security & Privacy > Device Management and start verification for this device.")
    print("Waiting for verification request...")

    async def verification_cb(event):
        if isinstance(event, KeyVerificationStart):
            print("Verification request received from", event.sender)
            resp = await client.accept_key_verification(event.sender, event.transaction_id)
            print("Accepted verification request.")
            # Wait for emoji/SAS step
            # The rest of the process is handled by Element UI and nio
        else:
            print("Received event:", event)

    client.add_to_device_callback(verification_cb, (KeyVerificationStart,))

    # Keep syncing to receive events
    while True:
        await client.sync(timeout=30000)

if __name__ == "__main__":
    asyncio.run(main())