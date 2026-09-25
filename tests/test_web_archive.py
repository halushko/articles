import io
import zipfile

import pytest

from segmentation_web.archive import InputValidationError, documents_from_zip


def make_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def parse_zip(payload: bytes):
    return documents_from_zip(
        payload,
        max_archive_bytes=1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_documents=10,
    )


def test_zip_documents_are_sorted_and_utf8_validated():
    documents = parse_zip(
        make_zip(
            {
                "nested/B.md": b"# B\n",
                "A.md": b"# A\n",
            }
        )
    )

    assert [document.relative_path for document in documents] == ["A.md", "nested/B.md"]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../outside.md", "Unsafe archive path"),
        ("notes.txt", "Only .md files"),
    ],
)
def test_zip_rejects_unsafe_or_unsupported_entries(name, message):
    with pytest.raises(InputValidationError, match=message):
        parse_zip(make_zip({name: b"test"}))


def test_zip_rejects_invalid_utf8_markdown():
    with pytest.raises(InputValidationError, match="UTF-8"):
        parse_zip(make_zip({"document.md": b"\xff\xfe"}))
