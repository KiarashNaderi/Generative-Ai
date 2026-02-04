import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API key (if using OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories for uploads and vector DB
UPLOAD_DIR = "data/uploads"
VECTOR_DB_DIR = "data/vectordb"
