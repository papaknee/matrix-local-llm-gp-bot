import asyncio
from nio import AsyncClient, AsyncClientConfig, LoginResponse
import getpass

HOMESERVER = "https://matrix.cf-hub.net"
USER_ID = "@hub-bot:matrix.cf-hub.net"
DEVICE_NAME = "hub-bot"
STORE_PATH = "data/matrix_store"

# Set this to the device ID of your Element session (not the bot)
# You can find this in Element: Settings > Security & Privacy > Device Management
# Example: "ITCQNLFNIN"
TARGET_DEVICE_ID = input("Enter the device ID of your Element session (from Device Management): ")

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

    print(f"Sending verification request to device {TARGET_DEVICE_ID}...")
    # Send verification request to your other device
    resp = await client.request_key_verification(
        methods=["m.sas.v1"],
        user_id=USER_ID,
        device_id=TARGET_DEVICE_ID
    )
    print("Verification request sent. Check your Element client for a prompt.")
    print("Keep this script running to handle the verification process.")

    # Keep syncing to handle the verification process
    while True:
        await client.sync(timeout=30000)

if __name__ == "__main__":
    asyncio.run(main())
