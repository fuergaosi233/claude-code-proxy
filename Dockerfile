# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy files required to build and install the package
COPY pyproject.toml README.md ./
COPY src ./src/

# Install dependencies using uv
RUN uv pip install --system --no-cache .

# Copy the rest of the application's code from the host to the container at /app
COPY . .

# Run start_proxy.py when the container launches
CMD ["python", "start_proxy.py"]
