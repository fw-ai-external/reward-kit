import argparse
import os
import sys
import logging
from typing import Optional

from .server import serve, serve_tunnel
from .models import RewardOutput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fireworks Reward Kit CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a reward function")
    serve_parser.add_argument("func_path", help="Path to the reward function to serve (e.g., 'module.path:function_name')")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    # Serve tunnel command
    tunnel_parser = subparsers.add_parser("serve-tunnel", help="Serve a reward function with a tunnel")
    tunnel_parser.add_argument("func_path", help="Path to the reward function to serve")
    tunnel_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a reward function locally")
    test_parser.add_argument("func_path", help="Path to the reward function to test")
    test_parser.add_argument("--messages", required=True, help="JSON string of messages to test with")
    
    # Deploy command (placeholder)
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a reward function to Fireworks")
    deploy_parser.add_argument("func_path", help="Path to the reward function to deploy")
    deploy_parser.add_argument("--name", required=True, help="Name for the deployed function")
    
    # Deploy to Cloud Run (placeholder)
    cloudrun_parser = subparsers.add_parser("deploy-cloudrun", help="Deploy a reward function to Cloud Run")
    cloudrun_parser.add_argument("func_path", help="Path to the reward function to deploy")
    cloudrun_parser.add_argument("--project", required=True, help="Google Cloud project ID")
    cloudrun_parser.add_argument("--name", required=True, help="Name for the Cloud Run service")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "serve":
        # Serve the reward function
        serve(func_path=args.func_path, host=args.host, port=args.port)
    
    elif args.command == "serve-tunnel":
        # Serve the reward function with a tunnel
        serve_tunnel(func_path=args.func_path, port=args.port)
    
    elif args.command == "test":
        # Test the reward function
        import json
        import importlib
        
        try:
            # Parse the messages
            messages = json.loads(args.messages)
            
            # Load the function
            if ":" not in args.func_path:
                raise ValueError(f"Invalid func_path format: {args.func_path}, expected 'module.path:function_name'")
            
            module_path, func_name = args.func_path.split(":", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            
            # Call the function
            result = func(messages=messages)
            
            # Print the result
            if isinstance(result, RewardOutput):
                print(f"Score: {result.score}")
                print("Metrics:")
                for name, metric in result.metrics.items():
                    print(f"  {name}: {metric.score} ({metric.reason or 'No reason provided'})")
            elif isinstance(result, tuple) and len(result) == 2:
                score, components = result
                print(f"Score: {score}")
                print("Components:")
                for name, value in components.items():
                    print(f"  {name}: {value}")
            else:
                print(f"Invalid result type: {type(result)}")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"Error testing reward function: {str(e)}")
            sys.exit(1)
    
    elif args.command == "deploy":
        # This would be implemented with actual API calls to deploy the function
        logger.info(f"Deploying {args.func_path} as {args.name}...")
        logger.info("This is a placeholder. In a complete implementation, this would deploy the function to Fireworks.")
    
    elif args.command == "deploy-cloudrun":
        # This would be implemented with actual Google Cloud API calls
        logger.info(f"Deploying {args.func_path} to Cloud Run as {args.name} in project {args.project}...")
        logger.info("This is a placeholder. In a complete implementation, this would deploy to Google Cloud Run.")
    
    else:
        logger.error("No command specified. Use --help to see available commands.")
        sys.exit(1)

if __name__ == "__main__":
    main()