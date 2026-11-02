from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from contextshift.dialogue import (
    DEFAULT_HISTORY_MAX_CHARACTERS,
    DEFAULT_HISTORY_MAX_TURNS,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: SecretStr = Field(
        validation_alias="DEEPSEEK_API_KEY",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias="DEEPSEEK_BASE_URL",
    )
    model: str = Field(
        default="deepseek-flash",
        validation_alias="CONTEXTSHIFT_MODEL",
    )
    reasoning_effort: Literal["none", "low", "high", "max"] = Field(
        default="none",
        validation_alias="CONTEXTSHIFT_REASONING_EFFORT",
    )
    max_output_tokens: int = Field(
        default=64,
        ge=1,
        le=256,
        validation_alias="CONTEXTSHIFT_MAX_OUTPUT_TOKENS",
    )
    history_max_turns: int = Field(
        default=DEFAULT_HISTORY_MAX_TURNS,
        ge=1,
        validation_alias="CONTEXTSHIFT_HISTORY_MAX_TURNS",
    )
    history_max_characters: int = Field(
        default=DEFAULT_HISTORY_MAX_CHARACTERS,
        ge=1,
        validation_alias="CONTEXTSHIFT_HISTORY_MAX_CHARACTERS",
    )
