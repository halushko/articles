from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    example_docs_dir: Path
    control_process_path: Path = Path("examples/access_recovery_process.md")
    process_model_mode: str = "llm"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = "gpt-5-mini"
    llm_timeout_seconds: float = 180.0
    llm_max_input_chars: int = 120_000
    llm_max_output_tokens: int = 12_000
    llm_response_format: str = "json_schema"
    llm_reasoning_effort: str = "low"
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
            process_model_mode=os.getenv("PROCESS_MODEL_MODE", "llm").strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                "https://api.openai.com/v1",
            ).strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "gpt-5-mini").strip(),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            llm_max_input_chars=int(os.getenv("LLM_MAX_INPUT_CHARS", "120000")),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "12000")),
            llm_response_format=os.getenv(
                "LLM_RESPONSE_FORMAT",
                "json_schema",
            ).strip(),
            llm_reasoning_effort=os.getenv(
                "LLM_REASONING_EFFORT",
                "low",
            ).strip(),
            max_archive_bytes=int(
                os.getenv("MAX_ARCHIVE_BYTES", str(20 * 1024 * 1024))
            ),
            max_uncompressed_bytes=int(
                os.getenv("MAX_UNCOMPRESSED_BYTES", str(50 * 1024 * 1024))
            ),
            max_documents=int(os.getenv("MAX_DOCUMENTS", "100")),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model and self.llm_base_url)
