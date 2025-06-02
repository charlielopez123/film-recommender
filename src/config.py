# config.py
from pydantic.v1 import BaseSettings, AnyHttpUrl

class Settings(BaseSettings):
    api_base_url: AnyHttpUrl
    api_key: str

    class Config:
        env_file = ".env"

settings = Settings()
