import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# OpenAI API key
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY_ZHENG")

# file paths
PDF_FILE_PATH = "data/Haida_bracelet.pdf"

# model name
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-3.5-turbo"

# chunk size
CHUNK_SIZE = 1000

# parameters for RAG model
TEMPERATURE = 0.5
MAX_TOKENS = 100
TOP_P = 0.98
TOP_K = 40


# Paths for evaluation
RAG_RESULTS_PATH = "checking_inputs.json"
