from setuptools import setup, find_packages

setup(
    name="reward-kit",
    version="0.2.8",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "dataclasses-json>=0.5.7",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "openai==1.9.0",
        "aiosqlite",
        "aiohttp",
        "PyYAML>=5.0", # Added PyYAML
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
            "pytest-asyncio", # Added pytest-asyncio
        ],
        "deepseek": [
            "datasets>=2.12.0",
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
    description="A Python library for defining, testing, deploying, and using reward functions for LLM fine-tuning",
    long_description="A Python library for defining, testing, deploying, and using reward functions for LLM fine-tuning",
    url="https://github.com/fireworks-ai/reward-kit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
