from pydantic import SecretStr
from typing import Literal

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # general settings
    APP_NAME: str | None = "server"
    DEBUG: bool = True
    ENV: Literal["dev", "prod", "test"] = "dev"

    # db settings
    MYSQL_HOST: str
    MYSQL_USER: str
    MYSQL_PASSWORD: SecretStr
    MYSQL_DATABASE: str

    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DATABASE: str | None = None

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str | None = None

    SQLALCHEMY_DATABASE_URI: str = ""

    class Config:
      env_file = ".env"
      case_sensitive = True

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DATABASE}?ssl=require"
        )
    


settings = Settings()
