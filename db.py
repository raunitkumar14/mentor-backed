"""Motor client lifecycle — opened once per process via FastAPI's lifespan."""
import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

MONGO_URI = os.environ["MONGO_URI"]
MONGO_DB = os.environ.get("MONGO_DB", "kylas")

_client: AsyncIOMotorClient | None = None


def connect() -> AsyncIOMotorDatabase:
    global _client
    _client = AsyncIOMotorClient(MONGO_URI)
    return _client[MONGO_DB]


def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
