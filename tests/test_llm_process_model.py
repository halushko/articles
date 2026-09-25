import asyncio
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import SourceDocument
from segmentation_web.database import Base
from segmentation_web.db_models import LLMExtractionResult, ProcessModelResult
from segmentation_web.llm_process import (
    LLMCompletion,
    LLMProcessModelBuilder,
    OpenAICompatibleClient,
    ProcessExtractionError,
)
from segmentation_web.main import create_app
from segmentation_web.pipeline import SegmentationPipeline
from segmentation_web.settings import Settings


class FakeStructuredLLMClient:
    provider = "test-provider"
    model = "test-process-model"

    def __init__(self, *, invalid_ref: bool = False):
        self.calls = 0
        self.invalid_ref = invalid_ref

    @property
    def config_identity(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "response_format": "json_schema",
        }

    def complete_json(
        self,
        *,
        schema_name,
        schema,
        system_prompt,
        user_payload,
    ):
        self.calls += 1
        refs = [
            fragment["ref"]
            for document in user_payload["documents"]
            for fragment in document["fragments"]
        ]
        assert schema_name == "grounded_process_graph"
        assert schema["type"] == "object"
        assert "untrusted source data" in system_prompt
        assert len(refs) >= 6
        if self.invalid_ref:
            refs[0] = "F99999"
        keys = [
            "receive_request",
            "register_incident",
            "classify_incident",
            "check_account",
            "restore_access",
            "close_incident",
        ]
        operations = [
            "Receive access request",
            "Register incident",
            "Classify incident",
            "Check account status",
            "Restore user access",
            "Close incident",
        ]
        nodes = []
        for index, (key, operation) in enumerate(zip(keys, operations)):
            nodes.append(
                {
                    "key": key,
                    "operation": operation,
                    "role": "Service Desk"
                    if index < 3 or index == 5
                    else "Identity Analyst",
                    "system": "ITSM" if index < 3 or index == 5 else "IAM",
                    "node_type": "action",
                    "evidence_fragment_refs": [refs[index]],
                    "confidence": 0.9,
                }
            )
        edges = []
        for index in range(len(keys) - 1):
            edges.append(
                {
                    "source_key": keys[index],
                    "target_key": keys[index + 1],
                    "edge_type": "sequence",
                    "condition": None,
                    "reason": "The documentation states the next operational step.",
                    "evidence_fragment_refs": [refs[index], refs[index + 1]],
                    "confidence": 0.85,
                }
            )
        return LLMCompletion(
            data={
                "process_title": "Access recovery",
                "nodes": nodes,
                "edges": edges,
                "warnings": [],
            },
            request_id="fake-request-1",
            model=self.model,
            finish_reason="stop",
            input_tokens=800,
            output_tokens=500,
        )


def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def source_documents():
    return [
        SourceDocument(
            "D1_service_desk.md",
            b"""# Service Desk Guide

## Intake

- Receive the access request from the user.
- Register the incident in ITSM.
- Classify the incident as an access issue.
""",
        ),
        SourceDocument(
            "D2_recovery.md",
            b"""# Identity Recovery Guide

## Recovery

- Check the account status in IAM.
- Restore user access after confirming the lock.
- Close the incident in ITSM after recovery.
""",
        ),
    ]


def test_llm_builder_creates_grounded_graph_and_reuses_corpus_cache():
    sessions = session_factory()
    client = FakeStructuredLLMClient()
    builder = LLMProcessModelBuilder(
        sessions,
        client,
        max_input_chars=100_000,
    )
    pipeline = SegmentationPipeline(sessions)

    first_run = pipeline.process(source_documents(), source_type="upload")
    first = builder.build(first_run.run_id)
    cached_model = builder.build(first_run.run_id)

    assert first.cache_hit is False
    assert cached_model.cache_hit is True
    assert client.calls == 1
    assert first.payload["derivation"]["mode"] == "llm_grounded"
    assert first.payload["derivation"]["universal_extraction"] is True
    assert first.payload["llm"]["extraction_cache_hit"] is False
    assert len(first.payload["hierarchy"]["base_graph"]["nodes"]) == 6
    assert len(first.payload["hierarchy"]["base_graph"]["edges"]) == 5
    assert all(first.payload["provenance"].values())
    assert all(
        item["evidence"] for item in first.payload["transition_provenance"].values()
    )

    second_run = pipeline.process(source_documents(), source_type="upload")
    second = builder.build(second_run.run_id)

    assert client.calls == 1
    assert second.payload["llm"]["extraction_cache_hit"] is True
    assert second.payload["llm"]["input_tokens"] == 800
    with sessions() as session:
        extraction_count = session.scalar(
            select(func.count()).select_from(LLMExtractionResult)
        )
        model_count = session.scalar(
            select(func.count()).select_from(ProcessModelResult)
        )
    assert extraction_count == 1
    assert model_count == 2


def test_llm_builder_rejects_unknown_fragment_references():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        source_documents(),
        source_type="upload",
    )
    builder = LLMProcessModelBuilder(
        sessions,
        FakeStructuredLLMClient(invalid_ref=True),
        max_input_chars=100_000,
    )

    with pytest.raises(ProcessExtractionError, match="unknown fragments"):
        builder.build(run.run_id)

    with sessions() as session:
        assert (
            session.scalar(select(func.count()).select_from(LLMExtractionResult)) == 0
        )
        assert session.scalar(select(func.count()).select_from(ProcessModelResult)) == 0


def test_uploaded_zip_can_build_llm_process_model_via_api():
    sessions = session_factory()
    client = FakeStructuredLLMClient()
    builder = LLMProcessModelBuilder(
        sessions,
        client,
        max_input_chars=100_000,
    )
    settings = Settings(
        database_url="sqlite://",
        example_docs_dir=Path("examples/access_recovery_source_docs"),
        process_model_mode="llm",
        max_archive_bytes=1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_documents=10,
    )
    app = create_app(
        settings=settings,
        session_factory=sessions,
        process_model_builder=builder,
    )
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for document in source_documents():
            archive.writestr(document.relative_path, document.content)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as web_client:
            run_response = await web_client.post(
                "/api/v1/runs",
                data={"source": "upload"},
                files={
                    "archive": (
                        "documentation.zip",
                        archive_buffer.getvalue(),
                        "application/zip",
                    )
                },
            )
            assert run_response.status_code == 201
            run_body = run_response.json()
            assert run_body["process_model_available"] is True

            model_response = await web_client.post(run_body["process_model_url"])
            assert model_response.status_code == 201
            model_body = model_response.json()
            assert model_body["atomic_node_count"] == 6
            assert model_body["process_model"]["process_title"] == "Access recovery"

            result = await web_client.get(run_body["result_url"])
            with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
                hierarchy = json.loads(archive.read("process_model_hierarchy.json"))
                manifest = json.loads(archive.read("manifest.json"))
            assert hierarchy["derivation"]["mode"] == "llm_grounded"
            assert manifest["process_model"]["derivation_mode"] == "llm_grounded"

    asyncio.run(scenario())


def test_openai_compatible_client_requests_strict_structured_output(monkeypatch):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "request-123"},
            json={
                "id": "completion-123",
                "model": "test-model-2026-09-01",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"nodes":[],"edges":[]}'},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        return real_client(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("segmentation_web.llm_process.httpx.Client", client_factory)
    client = OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key="local-test-key",
        model="test-model",
        timeout_seconds=10,
        max_output_tokens=500,
        response_format="json_schema",
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
        },
        "required": ["nodes", "edges"],
    }

    completion = client.complete_json(
        schema_name="process_graph",
        schema=schema,
        system_prompt="Return grounded JSON",
        user_payload={"documents": []},
    )

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer local-test-key"
    assert captured["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "process_graph",
            "strict": True,
            "schema": schema,
        },
    }
    assert captured["body"]["max_completion_tokens"] == 500
    assert completion.data == {"nodes": [], "edges": []}
    assert completion.request_id == "request-123"
    assert completion.input_tokens == 12
    assert completion.output_tokens == 7
