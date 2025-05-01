from setuptools import setup, find_packages

setup(
    name="reward-kit",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "dataclasses-json>=0.5.7",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "mypy>=0.812",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "fireworks-reward=reward_kit.cli:main",
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