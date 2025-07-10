#!/usr/bin/env python3
"""
Debug Seed Propagation in Multi-Environment Proxy

This script performs a step-by-step diagnosis of seed propagation to identify
where different seeds (42, 123, 456) might be getting lost or not properly
creating different environment configurations.

Focus areas from mcp_gym_implementation_summary.md:
1. Seed flow: test → proxy → backend servers → FrozenLake initialization
2. Environment diversity: verify different seeds create different maps
3. Session management: ensure different sessions use different seeds
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import reward_kit as rk

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Reduce noise from other modules
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class SeedPropagationDiagnostic:
    """Diagnoses seed propagation through the multi-environment system."""

    def __init__(self):
        self.base_dir = Path(__file__).parent / "examples" / "frozen_lake_mcp"
        self.proxy_process = None

    async def run_diagnostic(self):
        """Run complete seed propagation diagnostic."""
        print("🔍 === SEED PROPAGATION DIAGNOSTIC ===")
        print(
            "Goal: Verify seeds (42, 123, 456) create different FrozenLake environments"
        )
        print()

        try:
            # Step 1: Start multi-environment proxy
            await self._start_proxy_server()

            # Step 2: Test seed propagation
            await self._test_seed_propagation()

            # Step 3: Test environment diversity
            await self._test_environment_diversity()

            print("\n✅ Seed propagation diagnostic completed!")

        except Exception as e:
            logger.error(f"Diagnostic failed: {e}")
            raise
        finally:
            await self._cleanup()

    async def _start_proxy_server(self):
        """Start the multi-environment proxy server."""
        print("📡 Starting multi-environment proxy server...")

        server_script = str(self.base_dir / "server.py")
        requirements_path = str(self.base_dir / "requirements.txt")

        cmd = [
            "python",
            "-m",
            "reward_kit.mcp.multi_environment_proxy",
            "--server-script",
            server_script,
            "--requirements",
            requirements_path,
            "--port",
            "8090",
            "--max-envs",
            "5",
        ]

        self.proxy_process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for startup
        await asyncio.sleep(8)

        if self.proxy_process.poll() is not None:
            stdout, stderr = self.proxy_process.communicate()
            raise RuntimeError(
                f"Proxy failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        print("✅ Proxy server started successfully")

    async def _test_seed_propagation(self):
        """Test that seeds are properly propagated through the system."""
        print("\n🌱 Testing seed propagation...")

        # Create test dataset with different seeds
        test_seeds = [42, 123, 456]
        dataset = []

        for i, seed in enumerate(test_seeds):
            dataset.append(
                {
                    "id": f"seed_test_{seed}",
                    "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
                    "user_prompt_template": "Current state: {observation}. Choose your move.",
                    "environment_context": {"seed": seed, "size": 4, "p": 0.8},
                }
            )

        print(f"📊 Created dataset with seeds: {test_seeds}")

        # Create environments through proxy
        print("🔗 Connecting to proxy server...")
        envs = rk.make(
            "http://localhost:8090/mcp", dataset=dataset, model_id="diagnostic-test"
        )

        print(f"✅ Created {envs.n} environments through proxy")

        # Test initial environment states
        print("🎯 Testing initial environment states...")
        observations, _, _ = await envs.reset()

        print("📋 Initial observations received:")
        for i, obs in enumerate(observations):
            seed = test_seeds[i]
            print(f"  Environment {i} (seed {seed}):")
            print(f"    Observation type: {type(obs)}")
            if isinstance(obs, dict) and "grid" in obs:
                grid_lines = obs["grid"].split("\n")
                print(
                    f"    Grid size: {len(grid_lines)}x{len(grid_lines[0]) if grid_lines else 0}"
                )
                print(f"    Grid preview: {grid_lines[0] if grid_lines else 'N/A'}")
            else:
                print(f"    Observation: {obs}")

        await envs.close()

    async def _test_environment_diversity(self):
        """Test that different seeds create actually different environments."""
        print("\n🎲 Testing environment diversity...")

        # Test individual environments to verify they're different
        test_results = {}

        for seed in [42, 123, 456]:
            print(f"\n  Testing seed {seed}...")

            # Create single environment with specific seed
            dataset = [
                {
                    "id": f"diversity_test_{seed}",
                    "system_prompt": "You are playing FrozenLake, a 4x4 grid game.",
                    "user_prompt_template": "Current state: {observation}",
                    "environment_context": {"seed": seed, "size": 4, "p": 0.8},
                }
            ]

            envs = rk.make(
                "http://localhost:8090/mcp", dataset=dataset, model_id="diversity-test"
            )

            # Get initial state
            observations, _, _ = await envs.reset()
            obs = observations[0]

            if isinstance(obs, dict) and "grid" in obs:
                grid = obs["grid"]
                test_results[seed] = {
                    "grid": grid,
                    "grid_hash": hash(grid),
                    "observation": obs,
                }
                print(f"    ✅ Grid captured for seed {seed}")
                # Show first line for quick comparison
                lines = grid.split("\n")
                print(f"    First line: {lines[0] if lines else 'N/A'}")
            else:
                test_results[seed] = {"error": f"Unexpected observation format: {obs}"}
                print(f"    ❌ Unexpected observation format: {obs}")

            await envs.close()
            await asyncio.sleep(1)  # Small delay between tests

        # Analyze diversity
        print("\n📊 Environment Diversity Analysis:")
        grids = [
            result.get("grid") for result in test_results.values() if "grid" in result
        ]
        unique_grids = len(set(grids))

        print(f"  Total environments tested: {len(test_results)}")
        print(f"  Unique grids generated: {unique_grids}")
        print(f"  Expected unique grids: {len(test_results)}")

        if unique_grids == len(test_results):
            print("  ✅ SUCCESS: All seeds generated different environments!")
        else:
            print("  ❌ ISSUE: Some seeds generated identical environments!")

            # Show detailed comparison
            print("\n  Detailed grid comparison:")
            for seed, result in test_results.items():
                if "grid" in result:
                    print(f"    Seed {seed} grid hash: {result['grid_hash']}")
                    # Show full grid for manual inspection
                    print(f"    Grid:\n{result['grid']}")
                    print()

        return unique_grids == len(test_results)

    async def _cleanup(self):
        """Clean up test resources."""
        print("\n🧹 Cleaning up...")

        if self.proxy_process:
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()
                self.proxy_process.wait()
            print("✅ Proxy server stopped")


async def main():
    """Run the seed propagation diagnostic."""
    diagnostic = SeedPropagationDiagnostic()

    try:
        await diagnostic.run_diagnostic()
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
