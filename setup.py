from setuptools import setup, find_packages

setup(
    name="InventoryEnv",
    version="1.0.0",
    description="OpenEnv API for B2B Supply Chain & Inventory Management",
    author="Meta Scaler Hackathon Participant",
    author_email="participant@metascaler.ai",
    url="https://github.com/udaysahu09/InventoryEnv",
    py_modules=["app", "models", "environment", "inference"],
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-core>=2.14.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
