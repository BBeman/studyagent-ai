FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    BYPASS_TOOL_CONSENT=true \
    AWS_REGION=eu-west-1 \
    AWS_DEFAULT_REGION=eu-west-1

# Install slim cloud requirements (no Flask/dev dependencies)
COPY agentcore_requirements.txt agentcore_requirements.txt
RUN uv pip install -r agentcore_requirements.txt

RUN uv pip install aws-opentelemetry-distro==0.12.2

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy entire project (respecting .dockerignore) with proper ownership
COPY --chown=bedrock_agentcore:bedrock_agentcore . .

CMD ["opentelemetry-instrument", "python", "-m", "agentcore_app"]
