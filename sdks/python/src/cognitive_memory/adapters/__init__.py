"""Adapter implementations for cognitive-memory storage backends."""

from .base import MemoryAdapter
from .errors import AdapterError, DuplicateMemoryError, MemoryNotFoundError
from .memory import InMemoryAdapter
from .jsonl import JsonlFileAdapter
from .remote import RemoteAdapter, RemoteAdapterError

__all__ = [
    "MemoryAdapter",
    "InMemoryAdapter",
    "JsonlFileAdapter",
    "RemoteAdapter",
    "RemoteAdapterError",
    "AdapterError",
    "DuplicateMemoryError",
    "MemoryNotFoundError",
]
