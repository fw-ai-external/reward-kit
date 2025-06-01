#!/bin/bash

# Script to run MCP Intermediary Server and Test Client

# Activate virtual environment
source .venv/bin/activate

# Define server command
SERVER_COMMAND=".venv/bin/python reward_kit/mcp_agent/main.py --config mcp_agent_config.yaml --host localhost --port 8001"

# Define client command
CLIENT_COMMAND=".venv/bin/python tests/mcp_agent/test_rl_filesystem_scenario.py" # Updated client script

echo "Starting MCP Intermediary Server in the background..."
$SERVER_COMMAND &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for the server to start up (adjust sleep time if necessary)
echo "Waiting for server to initialize (5 seconds)..."
sleep 5

echo "Running RL Filesystem Scenario Test Client..." # Updated echo message
$CLIENT_COMMAND

CLIENT_EXIT_CODE=$?

echo "RL Filesystem Scenario Test Client finished with exit code: $CLIENT_EXIT_CODE" # Updated echo message

echo "Shutting down MCP Intermediary Server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null # Wait for the process to terminate and suppress "Terminated" message

echo "Server shut down."

# Exit with the client's exit code
exit $CLIENT_EXIT_CODE
