import io
import json
import zipfile

import pytest

from segmentation_web.archive import (
    InputValidationError,
    discover_example_corpora,
    documents_from_directory,
    documents_from_zip,
)


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


def test_directory_loader_accepts_descriptive_markdown_names_and_skips_readme(
    tmp_path,
):
    (tmp_path / "order_guide.md").write_text("# Order guide\n", encoding="utf-8")
    (tmp_path / "activation_runbook.md").write_text(
        "# Activation runbook\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("Not a corpus document", encoding="utf-8")

    documents = documents_from_directory(tmp_path, max_documents=10)

    assert [document.relative_path for document in documents] == [
        "activation_runbook.md",
        "order_guide.md",
    ]


def test_example_catalog_is_data_driven(tmp_path):
    catalog = tmp_path / "catalog"
    for example_id, title in (("alpha-flow", "Alpha flow"), ("beta-flow", "Beta flow")):
        directory = catalog / example_id
        directory.mkdir(parents=True)
        (directory / "procedure.md").write_text(
            f"# {title}\n\nRegister the request.\n", encoding="utf-8"
        )
        (directory / "example.json").write_text(
            json.dumps(
                {
                    "id": example_id,
                    "title": title,
                    "description": f"Documentation for {title}.",
                    "documents": ["procedure.md"],
                }
            ),
            encoding="utf-8",
        )

    corpora = discover_example_corpora(
        catalog_dir=catalog,
        fallback_dir=catalog / "alpha-flow",
        max_documents=10,
    )

    assert [(corpus.id, corpus.document_count) for corpus in corpora] == [
        ("alpha-flow", 1),
        ("beta-flow", 1),
    ]
