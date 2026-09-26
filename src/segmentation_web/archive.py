from __future__ import annotations

import io
import json
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class InputValidationError(ValueError):
    pass


EXAMPLE_MANIFEST = "example.json"
EXAMPLE_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


@dataclass(frozen=True)
class SourceDocument:
    relative_path: str
    content: bytes

    def text(self) -> str:
        try:
            return self.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputValidationError(
                f"{self.relative_path}: Markdown files must use UTF-8 encoding"
            ) from exc


@dataclass(frozen=True)
class ExampleCorpus:
    id: str
    title: str
    description: str
    directory: Path
    document_paths: tuple[str, ...]

    @property
    def document_count(self) -> int:
        return len(self.document_paths)


def _validate_relative_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InputValidationError(f"Unsafe archive path: {name}")
    if path.parts[0] == "__MACOSX" or path.name in {".DS_Store", "Thumbs.db"}:
        return ""
    return path.as_posix()


def documents_from_zip(
    archive: bytes,
    *,
    max_archive_bytes: int,
    max_uncompressed_bytes: int,
    max_documents: int,
) -> list[SourceDocument]:
    if not archive:
        raise InputValidationError("The uploaded ZIP archive is empty")
    if len(archive) > max_archive_bytes:
        raise InputValidationError(
            f"The ZIP archive exceeds the {max_archive_bytes}-byte upload limit"
        )

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise InputValidationError(
            "The uploaded file is not a valid ZIP archive"
        ) from exc

    documents: list[SourceDocument] = []
    paths_seen: set[str] = set()
    total_uncompressed = 0

    with zip_file:
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise InputValidationError(
                    f"Encrypted ZIP entries are not supported: {info.filename}"
                )

            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise InputValidationError(
                    f"Symbolic links are not allowed: {info.filename}"
                )

            relative_path = _validate_relative_path(info.filename)
            if not relative_path:
                continue
            if PurePosixPath(relative_path).suffix.lower() != ".md":
                raise InputValidationError(
                    f"Only .md files are supported in this version: {relative_path}"
                )

            path_key = relative_path.casefold()
            if path_key in paths_seen:
                raise InputValidationError(f"Duplicate document path: {relative_path}")
            paths_seen.add(path_key)

            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise InputValidationError(
                    "The uncompressed archive exceeds the configured size limit"
                )
            if len(documents) >= max_documents:
                raise InputValidationError(
                    f"The archive contains more than {max_documents} Markdown documents"
                )

            content = zip_file.read(info)
            document = SourceDocument(relative_path=relative_path, content=content)
            document.text()
            documents.append(document)

    if not documents:
        raise InputValidationError(
            "The ZIP archive does not contain Markdown documents"
        )

    return sorted(documents, key=lambda document: document.relative_path)


def _markdown_paths(path: Path) -> tuple[str, ...]:
    return tuple(
        file.relative_to(path).as_posix()
        for file in sorted(path.rglob("*.md"))
        if file.is_file() and file.name.casefold() != "readme.md"
    )


def documents_from_directory(
    path: Path,
    *,
    max_documents: int,
    document_paths: tuple[str, ...] | None = None,
) -> list[SourceDocument]:
    if not path.is_dir():
        raise InputValidationError(
            f"Example documentation directory does not exist: {path}"
        )

    relative_paths = document_paths or _markdown_paths(path)
    if not relative_paths:
        raise InputValidationError(f"No Markdown example documents found in {path}")
    if len(relative_paths) > max_documents:
        raise InputValidationError(
            f"The example set contains more than {max_documents} Markdown documents"
        )

    documents: list[SourceDocument] = []
    seen: set[str] = set()
    for raw_path in relative_paths:
        relative_path = _validate_relative_path(raw_path)
        if not relative_path or PurePosixPath(relative_path).suffix.lower() != ".md":
            raise InputValidationError(
                f"Example document must be a safe .md path: {raw_path}"
            )
        key = relative_path.casefold()
        if key in seen:
            raise InputValidationError(
                f"Duplicate example document path: {relative_path}"
            )
        seen.add(key)

        file = path.joinpath(*PurePosixPath(relative_path).parts)
        if file.is_symlink():
            raise InputValidationError(
                f"Example document must be a regular file: {relative_path}"
            )
        try:
            resolved = file.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InputValidationError(
                f"Example document does not exist: {relative_path}"
            ) from exc
        try:
            resolved.relative_to(path.resolve())
        except ValueError as exc:
            raise InputValidationError(
                f"Example document escapes its directory: {relative_path}"
            ) from exc
        if not resolved.is_file():
            raise InputValidationError(
                f"Example document must be a regular file: {relative_path}"
            )
        documents.append(
            SourceDocument(relative_path=relative_path, content=resolved.read_bytes())
        )

    for document in documents:
        document.text()
    return documents


def _manifest_string(payload: dict[str, Any], key: str, manifest: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{manifest}: '{key}' must be a non-empty string")
    return value.strip()


def _read_example_manifest(
    manifest: Path,
    *,
    max_documents: int,
) -> ExampleCorpus:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Invalid example manifest: {manifest}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError(f"{manifest}: root value must be an object")

    example_id = _manifest_string(payload, "id", manifest)
    if not EXAMPLE_ID_RE.fullmatch(example_id):
        raise InputValidationError(
            f"{manifest}: 'id' must contain lowercase letters, numbers or hyphens"
        )
    title = _manifest_string(payload, "title", manifest)
    description = _manifest_string(payload, "description", manifest)
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise InputValidationError(f"{manifest}: 'documents' must be a non-empty array")
    if not all(isinstance(value, str) for value in raw_documents):
        raise InputValidationError(f"{manifest}: every document path must be a string")
    document_paths = tuple(str(value) for value in raw_documents)
    # Reading the corpus here validates paths, UTF-8 and configured limits at startup.
    documents_from_directory(
        manifest.parent,
        max_documents=max_documents,
        document_paths=document_paths,
    )
    return ExampleCorpus(
        id=example_id,
        title=title,
        description=description,
        directory=manifest.parent,
        document_paths=document_paths,
    )


def discover_example_corpora(
    *,
    catalog_dir: Path | None,
    fallback_dir: Path,
    max_documents: int,
) -> tuple[ExampleCorpus, ...]:
    corpora: list[ExampleCorpus] = []
    if catalog_dir is not None and catalog_dir.is_dir():
        corpora.extend(
            _read_example_manifest(manifest, max_documents=max_documents)
            for manifest in sorted(catalog_dir.glob(f"*/{EXAMPLE_MANIFEST}"))
        )

    if not corpora:
        document_paths = _markdown_paths(fallback_dir)
        documents_from_directory(
            fallback_dir,
            max_documents=max_documents,
            document_paths=document_paths,
        )
        corpora.append(
            ExampleCorpus(
                id="default",
                title="Built-in example",
                description="Process the configured Markdown example documents.",
                directory=fallback_dir,
                document_paths=document_paths,
            )
        )

    seen: set[str] = set()
    for corpus in corpora:
        if corpus.id in seen:
            raise InputValidationError(f"Duplicate example corpus id: {corpus.id}")
        seen.add(corpus.id)
    return tuple(corpora)
