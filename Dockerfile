FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

# Default command to show uv commands
# CMD ["uv", "run"]

RUN uv run src/main.py
RUN uv run python src/predict.py --data report/ireland_first_5_overs.csv --model models/model.pkl
