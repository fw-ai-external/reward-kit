# Local Docker testing for FrozenLake MCP Simulation Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project from the root
COPY ../../../ .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the simulation server
CMD ["python", "-m", "examples.frozen_lake_mcp_complete.mcp_server.simulation_server", "--host", "0.0.0.0", "--port", "8000"]
