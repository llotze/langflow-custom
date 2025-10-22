"""
Setup script for Flow Builder Agent package.
"""

from setuptools import setup, find_packages

setup(
    name="flow_builder_agent",
    version="0.1.0",
    description="AI agent for building Langflow workflows from natural language",
    author="Flow Builder Team",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
