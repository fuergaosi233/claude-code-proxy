# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy files required for dependency resolution
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv sync
RUN uv sync --frozen --no-dev

# Copy the rest of the application's code from the host to the container at /app
COPY . .

# Run claude-code-proxy when the container launches
CMD ["uv", "run", "claude-code-proxy"]
