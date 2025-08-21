import os

# Use absolute path for DATABASE_PATH and KNOWLEDGE_BASE_PATH
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "instance", "support_data.db")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "instance", "knowledge")

AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_API_VERSION_GPT_4"),
    "model": os.getenv("AZURE_OPENAI_GPT_4_TURBO_MODEL"),
    "embedding_model": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
}

LLM_SETTINGS = {
    "temperature": 0.1,
    "max_tokens": 800,
    "timeout": 30
}

IMAP = {
    "host": os.getenv("IMAP_HOST", "imap.gmail.com"),
    "port": int(os.getenv("IMAP_PORT", 993)),
    "email": os.getenv("IMAP_EMAIL"),
    "password": os.getenv("IMAP_PASSWORD")
}

SMTP = {
    "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", 587)),
    "email": os.getenv("SMTP_EMAIL"),
    "password": os.getenv("SMTP_PASSWORD")
}

REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_API_VERSION_GPT_4",
    "AZURE_OPENAI_GPT_4_TURBO_MODEL",
    "IMAP_EMAIL",
    "IMAP_PASSWORD",
    "SMTP_EMAIL",
    "SMTP_PASSWORD"
]