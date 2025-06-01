#!/usr/bin/env python3
"""
Test script to verify reward-kit CLI integration with MCP agent filesystem RL example.

This script:
1. Starts the MCP intermediary server
2. Runs reward-kit with our configuration
3. Verifies the results
"""

import subprocess
import sys
import time
import signal
import os
import json
from pathlib import Path


def main():
    """Test the reward-kit CLI integration."""
    
    print("Testing Reward-Kit CLI Integration with MCP Agent Filesystem RL")
    print("=" * 65)
    
    # Change to the reward-kit root directory
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we have API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("❌ FIREWORKS_API_KEY environment variable not set!")
        print("Please set your Fireworks API key:")
        print("export FIREWORKS_API_KEY='your_api_key_here'")
        return 1
    
    print("✓ Fireworks API key found")
    
    # Start the MCP server
    print("\n1. Starting MCP intermediary server...")
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
            print("❌ Server failed to start!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return 1
        
        print("✓ MCP server should be running")
        
        # Run reward-kit evaluation
        print("\n3. Running reward-kit evaluation...")
        
        reward_kit_process = subprocess.run(
            [
                sys.executable, "-m", "reward_kit.cli", "run",
                "--config-path", "examples/mcp_agent_filesystem_rl",
                "--config-name", "config"
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("Reward-kit output:")
        print(reward_kit_process.stdout)
        if reward_kit_process.stderr:
            print("Reward-kit errors:")
            print(reward_kit_process.stderr)
        
        success = reward_kit_process.returncode == 0
        
        if success:
            print("\n4. Analyzing results...")
            
            # Look for results file
            results_files = list(Path(".").glob("**/mcp_filesystem_rl_results.jsonl"))
            if results_files:
                results_file = results_files[0]
                print(f"✓ Results file found: {results_file}")
                
                # Analyze results
                with open(results_file, "r") as f:
                    results = [json.loads(line) for line in f]
                
                print(f"✓ Processed {len(results)} samples")
                
                total_score = 0
                successful_moves = 0
                
                for result in results:
                    score = result.get("reward_result", {}).get("score", 0)
                    total_score += score
                    if score >= 0.9:  # Consider 0.9+ as successful
                        successful_moves += 1
                    
                    print(f"  Sample {result.get('sample_id', 'unknown')}: score={score:.2f}")
                
                avg_score = total_score / len(results) if results else 0
                success_rate = successful_moves / len(results) if results else 0
                
                print(f"\n📊 Results Summary:")
                print(f"  Average Score: {avg_score:.2f}")
                print(f"  Success Rate: {success_rate:.1%} ({successful_moves}/{len(results)})")
                
                if avg_score >= 0.5:
                    print("\n🎉 Integration test PASSED!")
                    print("The MCP agent filesystem RL example is working with reward-kit CLI!")
                else:
                    print("\n⚠️ Integration test partially successful, but low scores")
                    print("The LLM may not be generating the expected file operations")
            else:
                print("⚠️ No results file found, but command succeeded")
                success = False
        else:
            print("\n❌ Reward-kit command failed!")
        
        return 0 if success else 1
        
    except subprocess.TimeoutExpired:
        print("\n❌ Reward-kit command timed out!")
        return 1
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_server()


if __name__ == "__main__":
    sys.exit(main())