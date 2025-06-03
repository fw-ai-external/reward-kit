from setuptools import find_packages, setup

setup(
    name="reward-kit",
    version="0.2.11",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "dataclasses-json>=0.5.7",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "openai==1.78.1",
        "aiosqlite",
        "aiohttp",
        "PyYAML>=5.0",  # Added PyYAML
        "datasets==3.6.0",  # For dataset loading functionality
        "fsspec==2025.3.0",  # For filesystem interactions, pinned for dataset compatibility
        "hydra-core>=1.3.2",  # For configuration management
        "omegaconf>=2.3.0",  # For configuration objects
    ],
    extras_require={
        "dev": [
            "build",
            "twine",
            "pytest>=6.0.0",
            "black>=21.5b2",
            "mypy>=0.812",
            "flake8>=3.9.2",
            "autopep8>=1.5.0",
            "transformers>=4.0.0",
            "pytest-asyncio",
            "types-setuptools",
            "types-requests",
            "types-PyYAML",
            "types-docker",
            "torch>=1.9",
            "trl>=0.7.0",
            "peft>=0.7.0",
            # datasets, hydra-core, omegaconf moved to core dependencies
            "pre-commit",
            "mcp==1.9.2",
            "e2b>=0.15.0",  # Added e2b for E2B environment tests
        ],
        "deepseek": [
            "difflib>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fireworks-reward=reward_kit.cli:main",
            "reward-kit=reward_kit.cli:main",
        ],
    },
    author="Fireworks AI",
    author_email="info@fireworks.ai",
    description="A Python library for defining, testing, and using reward functions",
    long_description="A library for defining, testing, and deploying reward functions",
    url="https://github.com/fireworks-ai/reward-kit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
