import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# LLM
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))