# Independent runtime images, all built from the same audited source revision.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /project
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY services/__init__.py services/__init__.py

FROM base AS llm
COPY services/llm/ services/llm/
COPY config/llm.json config/llm.json
EXPOSE 8000
CMD ["uvicorn", "services.llm.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS core
COPY services/core/ services/core/
EXPOSE 8000
CMD ["uvicorn", "services.core.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS research
COPY services/research/ services/research/
COPY config/research.json config/research.json
EXPOSE 8000
CMD ["uvicorn", "services.research.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS gateway
COPY services/gateway/ services/gateway/
COPY web/ web/
EXPOSE 8000
CMD ["uvicorn", "services.gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Bundled SearXNG configuration enables its JSON API without publishing it.
FROM searxng/searxng:latest AS searxng
COPY config/searxng.yml /etc/searxng/settings.yml
