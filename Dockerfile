FROM python:3.12-slim
ARG SERVICE
WORKDIR /app
COPY common.py /app/common.py
COPY ${SERVICE}.py /app/service.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "/app/service.py"]
