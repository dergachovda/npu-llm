import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/qwen_npu_sym"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Answer concisely.")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
