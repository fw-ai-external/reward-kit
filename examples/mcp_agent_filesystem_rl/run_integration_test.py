#!/usr/bin/env python3
"""
Simple integration test runner for MCP Agent Filesystem RL Example
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def main():
    """Run the integration test with proper server management."""
    
    print("Starting MCP Agent Filesystem RL Integration Test")
    print("=" * 55)
    
    # Change to the reward-kit root directory
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Start the MCP server
    print("1. Starting MCP intermediary server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "reward_kit.mcp_agent.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    def cleanup_server():
        """Clean up the server process."""
        if server_process.poll() is None:
            print("Stopping MCP server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing MCP server...")
                server_process.kill()
                server_process.wait()
    
    # Set up signal handler for cleanup
    def signal_handler(signum, frame):
        cleanup_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Wait for server to start
        print("2. Waiting for server to initialize...")
        time.sleep(8)
        
        # Check if server is still running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print("Server failed to start!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return 1
        
        print("✓ Server should be running")
        
        # Run the integration test
        print("3. Running integration test...")
        
        # Change to the example directory
        example_dir = root_dir / "examples" / "mcp_agent_filesystem_rl"
        os.chdir(example_dir)
        
        test_process = subprocess.run(
            [sys.executable, "simple_integration_test.py"],
            capture_output=True,
            text=True
        )
        
        print("Test output:")
        print(test_process.stdout)
        if test_process.stderr:
            print("Test errors:")
            print(test_process.stderr)
        
        success = test_process.returncode == 0
        
        if success:
            print("\n🎉 Integration test PASSED!")
        else:
            print("\n❌ Integration test FAILED!")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_server()


if __name__ == "__main__":
    sys.exit(main())