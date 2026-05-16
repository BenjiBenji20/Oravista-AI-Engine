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

  GEMINI_API_KEY: str | None = None
  GEMINI_MODEL: str | None = None

  SQLALCHEMY_DATABASE_URI: str = ""

  class Config:
    env_file = ".env"
    case_sensitive = True

  
  def db_uri(self):
    self.SQLALCHEMY_DATABASE_URI = (
      f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD.get_secret_value()}"
      f"@{self.MYSQL_HOST}/{self.MYSQL_DATABASE}"
    )
    return self.SQLALCHEMY_DATABASE_URI

settings = Settings()