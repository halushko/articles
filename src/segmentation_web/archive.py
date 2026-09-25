from __future__ import annotations

import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class InputValidationError(ValueError):
    pass


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


def documents_from_directory(path: Path, *, max_documents: int) -> list[SourceDocument]:
    if not path.is_dir():
        raise InputValidationError(
            f"Example documentation directory does not exist: {path}"
        )

    files = sorted(file for file in path.glob("D*.md") if file.is_file())
    if not files:
        raise InputValidationError(f"No D*.md example documents found in {path}")
    if len(files) > max_documents:
        raise InputValidationError(
            f"The example set contains more than {max_documents} Markdown documents"
        )

    documents = [
        SourceDocument(relative_path=file.name, content=file.read_bytes())
        for file in files
    ]
    for document in documents:
        document.text()
    return documents
