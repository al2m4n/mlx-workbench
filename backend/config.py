from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    mlx_max_tokens: int = 2048
    mlx_temperature: float = 0.7
    mlx_top_p: float = 0.9

    hf_token: str | None = None

    upload_dir: str = "/tmp/mlx_uploads"
    max_upload_mb: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
