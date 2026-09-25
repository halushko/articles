from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RawBlock


class DocumentPartitioner(ABC):
    @abstractmethod
    def partition(self, artifact_id: str, text: str) -> list[RawBlock]:
        raise NotImplementedError
