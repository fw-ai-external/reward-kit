# Deploying MCP Servers to Google Cloud Run

This guide shows how to deploy MCP servers (like the FrozenLake MCP server) to Google Cloud Run using the `reward-kit deploy-mcp` command.

## Prerequisites

1. **Google Cloud Platform (GCP) Account**: Active GCP account with billing enabled
2. **`gcloud` CLI**: [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. **Required APIs**: Enable these APIs in your GCP project:
   - Cloud Build API
   - Artifact Registry API
   - Cloud Run Admin API
   - Secret Manager API (optional, for secrets)
4. **Permissions**: Ensure you have sufficient permissions:
   - Cloud Build Editor
   - Artifact Registry Administrator
   - Cloud Run Admin
   - Secret Manager Admin (if using secrets)
5. **`reward-kit` installed**: Install from the repository root:
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Configure GCP Settings

Create a `rewardkit.yaml` file in your project root:

```bash
# Copy the example configuration
cp examples/frozen_lake_mcp/rewardkit-mcp.example.yaml rewardkit.yaml
```

Edit `rewardkit.yaml` with your GCP project details:
```yaml
gcp_cloud_run:
  project_id: "your-actual-gcp-project-id"
  region: "us-central1"  # or your preferred region
  artifact_registry_repository: "reward-kit-mcp-servers"
```

### 2. Deploy the FrozenLake MCP Server

From the repository root, run:

```bash
reward-kit deploy-mcp \
    --id frozen-lake-mcp \
    --mcp-server-module examples.frozen_lake_mcp.frozen_lake_mcp_server \
    --port 8000
```

Or with explicit GCP parameters (if not using rewardkit.yaml):

```bash
reward-kit deploy-mcp \
    --id frozen-lake-mcp \
    --mcp-server-module examples.frozen_lake_mcp.frozen_lake_mcp_server \
    --gcp-project your-gcp-project-id \
    --gcp-region us-central1 \
    --port 8000
```

### 3. Test the Deployment

Once deployed, you'll get a Cloud Run URL. Test it:

```bash
# Replace with your actual Cloud Run URL
SERVICE_URL="https://frozen-lake-mcp-12345-uc.a.run.app"

# Test the health endpoint
curl ${SERVICE_URL}/health

# Test MCP connection (if your MCP server supports HTTP)
curl -X POST ${SERVICE_URL} \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}'
```

## Command Options

### Required Arguments
- `--id`: Unique identifier for your MCP server deployment
- `--mcp-server-module`: Python import path to your MCP server module

### Optional Arguments
- `--gcp-project`: GCP Project ID (can be set in rewardkit.yaml)
- `--gcp-region`: GCP Region (can be set in rewardkit.yaml)
- `--gcp-ar-repo`: Artifact Registry repo name (defaults to "reward-kit-mcp-servers")
- `--port`: Port for MCP server (default: 8000)
- `--python-version`: Python version for container (default: 3.11)
- `--requirements`: Additional pip requirements (newline separated)
- `--env-vars`: Environment variables in KEY=VALUE format

### Example with Additional Requirements

```bash
reward-kit deploy-mcp \
    --id my-custom-mcp \
    --mcp-server-module my_mcp_servers.custom_server \
    --requirements "numpy>=1.24.0\nscipy>=1.10.0" \
    --env-vars "DEBUG=true" "LOG_LEVEL=info"
```

## What Happens During Deployment

1. **Environment Check**: Validates gcloud CLI and permissions
2. **Configuration**: Resolves GCP project, region, and repository settings
3. **Artifact Registry**: Creates repository if it doesn't exist
4. **Docker Build**:
   - Generates optimized Dockerfile for MCP servers
   - Installs base requirements (fastmcp, pydantic, uvicorn, etc.)
   - Adds your custom requirements
   - Multi-stage build for smaller image size
5. **Cloud Run Deployment**:
   - Deploys with public access (unauthenticated)
   - Configures health checks
   - Sets up environment variables

## Monitoring and Management

### View Logs
```bash
# Replace with your values
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=frozen-lake-mcp' \
    --project your-gcp-project-id \
    --limit 50
```

### Update Deployment
Simply run the same deploy command again - it will update the existing service.

### Delete Deployment
```bash
gcloud run services delete frozen-lake-mcp \
    --region us-central1 \
    --project your-gcp-project-id
```

## Custom MCP Servers

To deploy your own MCP server:

1. **Create MCP Server Module**: Follow the FastMCP patterns from the FrozenLake example
2. **Ensure Proper Entry Point**: Your module should be runnable as `python -m your.module.path`
3. **Handle Command Line Args**: Support `--transport` and `--port` arguments
4. **Deploy**: Use the same `reward-kit deploy-mcp` command with your module path

### Example Custom Server Structure

```python
# my_mcp_servers/custom_server.py
from mcp.server.fastmcp import FastMCP

app = FastMCP("CustomServer", stateless_http=True)

@app.tool(name="my_tool")
def my_tool(param: str) -> str:
    return f"Processed: {param}"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "streamable-http":
        import os
        os.environ["PORT"] = str(args.port)

    app.run(transport=args.transport)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have the required IAM roles
2. **API Not Enabled**: Enable all required GCP APIs
3. **Port Issues**: MCP servers should listen on the PORT environment variable
4. **Module Import**: Ensure your MCP server module is importable from the project root

### Debug Tips

1. **Check Local Setup**: Test your MCP server locally first
2. **Verify Module Path**: Ensure `--mcp-server-module` path is correct
3. **Review Logs**: Use gcloud logging to see container startup logs
4. **Test Health Check**: Verify the `/health` endpoint responds

## Cost Considerations

Cloud Run charges based on:
- **CPU and Memory**: Only while serving requests
- **Requests**: Per million requests
- **Artifact Registry**: Storage for container images

MCP servers typically have low resource usage, making Cloud Run very cost-effective for most use cases.
