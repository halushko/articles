from __future__ import annotations

import copy
import io
import json
import re
import zipfile
from pathlib import PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db_models import (
    DocumentationGraphResult,
    DocumentVersion,
    ProcessModelResult,
    RunDocument,
    SegmentationResult,
    SegmentationRun,
)

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _result_filename(position: int, relative_path: str, content_hash: str) -> str:
    stem = PurePosixPath(relative_path).stem
    safe_stem = SAFE_FILENAME_RE.sub("_", stem).strip("._") or "document"
    return f"documents/{position:03d}_{safe_stem}.{content_hash[:12]}.fragments.json"


def build_result_zip(session: Session, run_id: str) -> bytes:
    run = session.get(SegmentationRun, run_id)
    if run is None:
        raise KeyError(run_id)

    rows = session.execute(
        select(RunDocument, DocumentVersion, SegmentationResult)
        .join(DocumentVersion, DocumentVersion.id == RunDocument.document_version_id)
        .outerjoin(
            SegmentationResult,
            SegmentationResult.id == RunDocument.segmentation_result_id,
        )
        .where(RunDocument.run_id == run_id)
        .order_by(RunDocument.position)
    ).all()

    manifest_documents: list[dict[str, object]] = []
    exported_documents: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    graph = session.scalar(
        select(DocumentationGraphResult)
        .where(DocumentationGraphResult.run_id == run_id)
        .order_by(DocumentationGraphResult.created_at.desc())
    )
    process_model = session.scalar(
        select(ProcessModelResult)
        .where(ProcessModelResult.run_id == run_id)
        .order_by(ProcessModelResult.created_at.desc())
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for run_document, version, result in rows:
            manifest_entry: dict[str, object] = {
                "path": run_document.relative_path,
                "sha256": version.content_sha256,
                "size_bytes": version.size_bytes,
                "status": run_document.status,
                "cache_hit": run_document.cache_hit,
            }

            if result is not None:
                payload = copy.deepcopy(result.payload)
                payload["source"]["path"] = run_document.relative_path
                payload["source"]["result_cache_hit"] = run_document.cache_hit
                result_name = _result_filename(
                    run_document.position,
                    run_document.relative_path,
                    version.content_sha256,
                )
                archive.writestr(result_name, _json_bytes(payload))
                manifest_entry["result_file"] = result_name
                manifest_entry["fragment_count"] = result.fragment_count
                exported_documents.append(payload)
            else:
                error = run_document.error or "Unknown segmentation error"
                manifest_entry["error"] = error
                errors.append({"path": run_document.relative_path, "error": error})

            manifest_documents.append(manifest_entry)

        manifest = {
            "schema_version": "1.0",
            "run": {
                "id": run.id,
                "source_type": run.source_type,
                "corpus_sha256": run.corpus_sha256,
                "status": run.status,
                "document_count": run.document_count,
                "fragment_count": run.fragment_count,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat()
                if run.completed_at
                else None,
            },
            "documents": manifest_documents,
        }
        if graph is not None:
            graph_file = "documentation_graph.json"
            archive.writestr(graph_file, _json_bytes(graph.payload))
            manifest["documentation_graph"] = {
                "id": graph.id,
                "builder_version": graph.builder_version,
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
                "result_file": graph_file,
            }
        if process_model is not None:
            process_model_file = "process_model_hierarchy.json"
            archive.writestr(process_model_file, _json_bytes(process_model.payload))
            hierarchy = process_model.payload.get("hierarchy") or {}
            level_files: dict[str, str] = {}
            base_graph = hierarchy.get("base_graph")
            if base_graph:
                level_files["L0"] = "process_model_l0.json"
                archive.writestr(level_files["L0"], _json_bytes(base_graph))
            hierarchy_levels = list(hierarchy.get("levels") or [])
            for level in hierarchy_levels:
                target_level = int(level.get("target_level") or 0)
                if target_level < 1 or not level.get("graph"):
                    continue
                label = f"L{target_level}"
                level_files[label] = f"process_model_l{target_level}.json"
                archive.writestr(level_files[label], _json_bytes(level["graph"]))

            aggregation_file = "aggregation_results.json"
            archive.writestr(
                aggregation_file,
                _json_bytes(
                    {
                        "configuration": process_model.payload.get("configuration"),
                        "levels": hierarchy_levels,
                    }
                ),
            )
            provenance_file = "provenance.json"
            archive.writestr(
                provenance_file,
                _json_bytes(
                    {
                        "nodes": process_model.payload.get("provenance") or {},
                        "transitions": process_model.payload.get(
                            "transition_provenance"
                        )
                        or {},
                    }
                ),
            )
            warnings_file = "process_model_warnings.json"
            archive.writestr(
                warnings_file,
                _json_bytes(process_model.payload.get("warnings") or []),
            )
            manifest["process_model"] = {
                "id": process_model.id,
                "builder_version": process_model.builder_version,
                "derivation_mode": process_model.derivation_mode,
                "level_count": process_model.level_count,
                "atomic_node_count": process_model.atomic_node_count,
                "result_file": process_model_file,
                "level_files": level_files,
                "aggregation_file": aggregation_file,
                "provenance_file": provenance_file,
                "warnings_file": warnings_file,
            }
        archive.writestr("manifest.json", _json_bytes(manifest))
        archive.writestr(
            "all_fragments.json",
            _json_bytes({"schema_version": "1.0", "documents": exported_documents}),
        )
        archive.writestr("errors.json", _json_bytes(errors))

    return output.getvalue()
