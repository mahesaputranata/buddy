# brain/retrieval/sources/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from brain.core.RAG.types import ContextItem, RetrievalOptions

class RetrievalSource(ABC):
    name: str