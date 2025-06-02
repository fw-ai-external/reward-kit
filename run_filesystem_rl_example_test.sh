#!/bin/bash

# Script to run the MCP Agent Filesystem RL example end-to-end.
# This script starts the RewardKitIntermediaryServer, runs the
# example evaluation using the reward-kit CLI, and then shuts down the server.

# Ensure the script is run from the repository root
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
if [ "$PWD" != "$SCRIPT_DIR" ]; then
  echo "Please run this script from the repository root: $SCRIPT_DIR"
  exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Make sure it's set up at ./.venv"
    exit 1
fi
echo "Virtual environment activated."

# Define server command
# The mcp_agent_config.yaml is expected to be in the root directory
SERVER_COMMAND=".venv/bin/python reward_kit/mcp_agent/main.py --config mcp_agent_config.yaml --host localhost --port 8001"

# Define client command for the filesystem RL example
# The --config-path is relative to the repository root (current directory)
CLIENT_COMMAND=".venv/bin/python -m reward_kit.cli run --config-path ./examples/mcp_agent_filesystem_rl --config-name config"

echo "Starting MCP Intermediary Server in the background..."
$SERVER_COMMAND &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for the server to start up (adjust sleep time if necessary)
# Check for server readiness by polling or checking logs if possible in future
echo "Waiting for server to initialize (10 seconds)..."
sleep 10

echo "Running Filesystem RL Example Evaluation via reward-kit CLI..."
$CLIENT_COMMAND

CLIENT_EXIT_CODE=$?

echo "Filesystem RL Example Evaluation finished with exit code: $CLIENT_EXIT_CODE"

echo "Shutting down MCP Intermediary Server (PID: $SERVER_PID)..."
# Send SIGINT (Ctrl+C) for graceful shutdown, then SIGTERM if needed, then SIGKILL
kill -SIGINT $SERVER_PID
# Wait a bit for graceful shutdown
sleep 5
if ps -p $SERVER_PID > /dev/null; then
   echo "Server still running, sending SIGTERM..."
   kill -SIGTERM $SERVER_PID
   sleep 2
   if ps -p $SERVER_PID > /dev/null; then
      echo "Server still running, sending SIGKILL..."
      kill -SIGKILL $SERVER_PID
   fi
fi
wait $SERVER_PID 2>/dev/null # Wait for the process to terminate and suppress "Terminated" message

echo "Server shut down."

# Exit with the client's exit code
exit $CLIENT_EXIT_CODE
