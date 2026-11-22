from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    GROQ_API_KEY: str
    DATABASE_URL: str = "sqlite:///./kanoonsathi.db"
    DOCUMENTS_DIR: str = "./storage/documents"
    DATA_DIR: str = "./data"
    VECTOR_STORE_PATH: str = "./storage/faiss_index"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    LLAMA_CLOUD_API_KEY: str | None = None

    model_config = {"env_file": ".env"}

    def documents_path(self) -> Path:
        p = Path(self.DOCUMENTS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def vector_store_path(self) -> Path:
        p = Path(self.VECTOR_STORE_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def data_path(self) -> Path:
        return Path(self.DATA_DIR)


settings = Settings()
