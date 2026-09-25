from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    example_docs_dir: Path
    control_process_path: Path = Path("examples/access_recovery_process.md")
    max_archive_bytes: int = 20 * 1024 * 1024
    max_uncompressed_bytes: int = 50 * 1024 * 1024
    max_documents: int = 100

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://segmenter:segmenter@localhost:5432/segmenter",
            ),
            example_docs_dir=Path(
                os.getenv("EXAMPLE_DOCS_DIR", "examples/access_recovery_source_docs")
            ),
            control_process_path=Path(
                os.getenv(
                    "CONTROL_PROCESS_PATH",
                    "examples/access_recovery_process.md",
                )
            ),
            max_archive_bytes=int(
                os.getenv("MAX_ARCHIVE_BYTES", str(20 * 1024 * 1024))
            ),
            max_uncompressed_bytes=int(
                os.getenv("MAX_UNCOMPRESSED_BYTES", str(50 * 1024 * 1024))
            ),
            max_documents=int(os.getenv("MAX_DOCUMENTS", "100")),
        )
