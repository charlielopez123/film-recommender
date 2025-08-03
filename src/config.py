from pydantic.v1 import BaseSettings, AnyHttpUrl, DirectoryPath

class Settings(BaseSettings):
    api_base_url: AnyHttpUrl
    api_key: str
    aud_model_dir: DirectoryPath

    class Config:
        env_file = ".env"

settings = Settings()
