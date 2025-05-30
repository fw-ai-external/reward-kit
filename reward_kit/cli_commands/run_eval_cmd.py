"""
CLI command for running the full evaluation pipeline (generation + evaluation).
This script is intended to be a Hydra application.
"""

import asyncio
import logging
import sys  # Added import for sys.exit

import hydra

# Ensure hydra.core.hydra_config is available if used for output_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import (  # Ensure MISSING is imported if used in configs
    MISSING,
    DictConfig,
    OmegaConf,
)

from reward_kit.execution.pipeline import EvaluationPipeline

logger = logging.getLogger(__name__)


def run_evaluation_command_logic(cfg: DictConfig) -> None:
    """
    Main logic for the 'run-evaluation' command.
    """
    logger.info("Starting 'run-evaluation' command with resolved Hydra config.")

    # Make Hydra's runtime output directory available to the pipeline if needed
    # This assumes 'hydra_output_dir' is a valid field in the pipeline's config if it uses it.
    # A cleaner way is for the pipeline to be Hydra-aware or for this function to pass it explicitly.
    # For now, let's add it to the cfg object that pipeline receives.
    # Ensure the config is not frozen if we add keys, then restore its original struct state.
    was_struct = OmegaConf.is_struct(cfg)
    if was_struct:
        OmegaConf.set_struct(cfg, False)
    cfg.hydra_output_dir = HydraConfig.get().runtime.output_dir
    if was_struct:
        OmegaConf.set_struct(cfg, True)

    logger.debug(f"Full configuration for pipeline:\n{OmegaConf.to_yaml(cfg)}")

    try:
        pipeline = EvaluationPipeline(pipeline_cfg=cfg)
        asyncio.run(pipeline.run())
        logger.info("'run-evaluation' command finished successfully.")
    except ValueError as ve:
        logger.error(f"Configuration or Value error in pipeline: {ve}", exc_info=True)
        sys.exit(1)  # Exit with error code for critical failures
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the evaluation pipeline: {e}",
            exc_info=True,
        )
        sys.exit(1)  # Exit with error code


# This is the Hydra entry point for this command.
# It needs a config_path relative to where this script is, or an absolute one.
# If reward-kit is installed, conf might not be easily found via relative paths.
# Using `pkg://` provider is more robust for installed packages.
# For now, assume a `conf` dir at project root, and this script is in `reward_kit/cli_commands`.
import os  # Ensure os is imported for path manipulation

# So, `config_path` would be `../../conf`.
# The `config_name` will be the primary config for this `run` command.
# Let's point directly to the example's config for now to simplify debugging Hydra pathing.
# Path from reward_kit/cli_commands/ to examples/math_example/conf/
# Construct an absolute path or a file:// URL to make it more robust.
_RUN_EVAL_CMD_DIR = os.path.dirname(os.path.abspath(__file__))
# Default config_path for @hydra.main, relative to this file.
# Points to the project's top-level 'conf' directory.
_DEFAULT_HYDRA_CONFIG_PATH = os.path.abspath(
    os.path.join(_RUN_EVAL_CMD_DIR, "..", "..", "conf")
)


@hydra.main(
    config_path=_DEFAULT_HYDRA_CONFIG_PATH, config_name=None, version_base="1.3"
)
def hydra_cli_entry_point(cfg: DictConfig) -> None:
    # config_path and config_name from CLI will override the defaults in the decorator.
    # If --config-name is not provided via CLI, Hydra would look for a default config
    # (e.g., config.yaml) in the _DEFAULT_HYDRA_CONFIG_PATH.
    # However, our reward-kit run command will always pass --config-path and --config-name.
    # passed to `reward-kit run` (e.g., --config-path, --config-name)
    # or by Hydra's default search behavior if not provided via CLI.
    run_evaluation_command_logic(cfg)


# This allows running `python -m reward_kit.cli_commands.run_eval_cmd` (if __main__.py in folder)
# or if this file itself is made executable.
if __name__ == "__main__":
    # This will parse sys.argv for Hydra overrides.
    # Example: python reward_kit/cli_commands/run_eval_cmd.py dataset=gsm8k_local_prompts generation.enabled=false
    import sys  # Required for sys.exit

    hydra_cli_entry_point()
