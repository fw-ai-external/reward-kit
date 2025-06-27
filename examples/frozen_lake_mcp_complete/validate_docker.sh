#!/bin/bash

# Comprehensive Docker Validation Script for FrozenLake MCP Server
# This script validates the complete Docker workflow before remote deployment

set -e  # Exit on any error

echo "🐳 DOCKER VALIDATION SUITE"
echo "============================================================"
echo "🎯 Purpose: Validate complete Docker workflow"
echo "📋 Tests: Build, Run, Connect, North Star Interface"
echo "============================================================"
echo

# Configuration
IMAGE_NAME="frozen-lake-mcp:local"
CONTAINER_NAME="frozen-lake-test"
PORT=8000
MCP_URL="http://localhost:${PORT}/mcp/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

cleanup() {
    log_info "Cleaning up..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Step 1: Build Docker image
echo "🔨 Step 1: Building Docker Image"
echo "----------------------------------------"
cd ../../  # Go to project root from examples/frozen_lake_mcp_complete
if docker build -f examples/frozen_lake_mcp_complete/Dockerfile -t ${IMAGE_NAME} .; then
    log_success "Docker image built successfully"
else
    log_error "Docker build failed"
    exit 1
fi
echo

# Step 2: Run container
echo "🚀 Step 2: Starting Docker Container"
echo "----------------------------------------"
log_info "Starting container on port ${PORT}..."
if docker run -d --name ${CONTAINER_NAME} -p ${PORT}:${PORT} ${IMAGE_NAME}; then
    log_success "Container started successfully"
else
    log_error "Failed to start container"
    exit 1
fi

# Wait for container to be ready
log_info "Waiting for container to be ready..."
sleep 10

# Check if container is still running
if ! docker ps | grep -q ${CONTAINER_NAME}; then
    log_error "Container stopped unexpectedly"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

log_success "Container is running"
echo

# Step 3: Test basic HTTP connectivity
echo "🌐 Step 3: Testing HTTP Connectivity"
echo "----------------------------------------"
log_info "Testing MCP endpoint: ${MCP_URL}"
if curl -s -I ${MCP_URL} | grep -q "405 Method Not Allowed"; then
    log_success "MCP endpoint responding correctly (405 expected for HEAD)"
else
    log_error "MCP endpoint not responding correctly"
    docker logs ${CONTAINER_NAME}
    exit 1
fi
echo

# Step 4: Test MCP connection
echo "🔌 Step 4: Testing MCP Connection"
echo "----------------------------------------"
cd examples/frozen_lake_mcp_complete/local_testing
log_info "Testing MCP connection to Docker container..."
if ../../../.venv/bin/python test_simple_connection.py; then
    log_success "MCP connection test passed"
else
    log_error "MCP connection test failed"
    exit 1
fi
echo

# Step 5: Test North Star Interface
echo "⭐ Step 5: Testing North Star Interface"
echo "----------------------------------------"
log_info "Running full north star interface test..."
if ../../../.venv/bin/python test_north_star.py; then
    log_success "North Star interface test passed"
else
    log_error "North Star interface test failed"
    exit 1
fi
echo

# Step 6: Container logs check
echo "📋 Step 6: Container Health Check"
echo "----------------------------------------"
log_info "Checking container logs for errors..."
if docker logs ${CONTAINER_NAME} 2>&1 | grep -i error | grep -v "Session terminated"; then
    log_warning "Found some errors in container logs (see above)"
else
    log_success "No critical errors found in container logs"
fi
echo

# Final summary
echo "🎉 DOCKER VALIDATION COMPLETE"
echo "============================================================"
log_success "All Docker validation tests passed!"
echo "✅ Docker image builds correctly"
echo "✅ Container starts and runs properly"
echo "✅ MCP endpoint is accessible"
echo "✅ MCP connections work"
echo "✅ North Star interface functions correctly"
echo "✅ Container logs show no critical errors"
echo
echo "🚀 Ready for remote deployment!"
echo "============================================================"
