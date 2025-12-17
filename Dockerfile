FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY src ./src

ENTRYPOINT ["uv", "run", "uvicorn", "src.api:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]