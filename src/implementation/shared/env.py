"""Initialisiert die globale Prozessumgebung GANZ ZU BEGINN eines Skripts"""
import os
import sys

from dotenv import load_dotenv

from implementation.shared import paths

ENV_PATH = paths.PROJECT_ROOT / ".env"

if not load_dotenv(ENV_PATH):
    print(f"Warning: {ENV_PATH} not found. Variable not set.", flush=True)

sys.pycache_prefix = str(paths.CACHE_DIR)
os.environ["PYTHONPYCACHEPREFIX"] = str(paths.CACHE_DIR)
