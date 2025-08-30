import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define project structure
list_of_files = [
    "data/policies/.gitkeep",
    "data/product_specs/.gitkeep",
    "data/screenshots/.gitkeep",
    "embeddings/.gitkeep",
    "src/__init__.py",
    "src/ingest.py",
    "src/embedder.py",
    "src/retriever.py",
    "src/agents.py",
    "src/orchestrator.py",
    "src/utils.py",
    "api/__init__.py",
    "api/server.py",
    "logs/.gitkeep",
    "tests/__init__.py",
    "tests/test_ingest.py",
    "requirements.txt",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if not exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating new file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
