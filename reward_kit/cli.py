"""
Command-line interface for reward-kit.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from reward_kit.evaluation import preview_evaluation, create_evaluation

def setup_logging(verbose=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s"
    )

def check_environment():
    """Check if required environment variables are set"""
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Warning: FIREWORKS_API_KEY environment variable is not set.")
        print("This is required for API calls. Set this variable before running the command.")
        print("Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY reward-kit [command]")
        return False
    return True

def preview_command(args):
    """Preview an evaluator with sample data"""
    
    # Check environment variables
    if not check_environment():
        return 1
    
    # Validate paths
    if args.metrics_folders:
        for folder in args.metrics_folders:
            if "=" not in folder:
                print(f"Error: Metric folder format should be 'name=path', got '{folder}'")
                return 1

    if not args.samples:
        print("Error: Sample file (--samples) is required for preview")
        return 1
    
    if not Path(args.samples).exists():
        print(f"Error: Sample file '{args.samples}' not found")
        return 1
        
    # Run preview
    try:
        preview_result = preview_evaluation(
            metric_folders=args.metrics_folders,
            sample_file=args.samples,
            max_samples=args.max_samples
        )
        
        preview_result.display()
        return 0
    except Exception as e:
        print(f"Error previewing evaluator: {str(e)}")
        return 1

def deploy_command(args):
    """Create and deploy an evaluator"""
    
    # Check environment variables
    if not check_environment():
        return 1
    
    # Validate paths
    if args.metrics_folders:
        for folder in args.metrics_folders:
            if "=" not in folder:
                print(f"Error: Metric folder format should be 'name=path', got '{folder}'")
                return 1
                
    if not args.id:
        print("Error: Evaluator ID (--id) is required for deployment")
        return 1
        
    # Create the evaluator
    try:
        evaluator = create_evaluation(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force
        )
        
        print(f"Successfully created evaluator: {evaluator['name']}")
        return 0
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        return 1

def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="reward-kit: Tools for evaluation and reward modeling"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preview command
    preview_parser = subparsers.add_parser(
        "preview", 
        help="Preview an evaluator with sample data"
    )
    preview_parser.add_argument(
        "--metrics-folders", "-m",
        nargs="+",
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'"
    )
    preview_parser.add_argument(
        "--samples", "-s",
        required=True,
        help="Path to JSONL file containing sample data"
    )
    preview_parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (default: 5)"
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", 
        help="Create and deploy an evaluator"
    )
    deploy_parser.add_argument(
        "--id",
        required=True,
        help="ID for the evaluator"
    )
    deploy_parser.add_argument(
        "--metrics-folders", "-m",
        nargs="+",
        required=True,
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'"
    )
    deploy_parser.add_argument(
        "--display-name",
        help="Display name for the evaluator (defaults to ID if not provided)"
    )
    deploy_parser.add_argument(
        "--description",
        help="Description for the evaluator"
    )
    deploy_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force update if evaluator already exists"
    )
    
    return parser.parse_args(args)

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.command == "preview":
        return preview_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    else:
        # No command provided, show help
        parser = argparse.ArgumentParser()
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
