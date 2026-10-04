"""Central settings. Every environment variable the project reads is declared here.

Values come from the process environment or a `.env` file in the working directory.
Unknown keys in `.env` are ignored, so adding variables never crashes start-up.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


def _split(csv: str) -> list[str]:
    return [item.strip() for item in csv.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM / VLM (Anthropic API) ---
    anthropic_api_key: str = ""
    llm_model_id: str = "claude-sonnet-5-5"
    llm_max_retries: int = 5
    llm_provider: str = "anthropic"                  # "anthropic" or "gemini": which API the judge tools call
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.5-flash"

    # --- local models ---
    text_dl_model_path: str = "models/text_dl_v2"              # directory (HF format); v2 = DeBERTa-v3-base on the HC3+MAGE+RAID mix
    slm_observer_model: str = "Qwen/Qwen2.5-0.5B"
    slm_performer_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    image_model_path: str = "models/efficientnet_b4.pt"
    calibration_path: str = "models/calibration.json"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- runtime switches (also used for ablations) ---
    enabled_text_tools: str = "dl,slm,llm"
    enabled_image_tools: str = "dl,clip,general,vlm"
    enable_reflexion: bool = True
    image_tta: bool = False

    # --- storage ---
    chroma_persist_dir: str = "./chroma_data"
    evidence_db_url: str = "sqlite:///evidence_store/forensics.db"
    jobs_dir: str = "./jobs"
    log_level: str = "INFO"

    # --- input limits ---
    min_text_words: int = 20
    max_text_words: int = 20000
    max_image_mb: int = 10
    max_image_side_px: int = 4096

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider.lower() == "gemini" else self.anthropic_api_key

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def judge_model_id(self) -> str:
        return self.gemini_model_id if self.llm_provider.lower() == "gemini" else self.llm_model_id

    @property
    def text_tools(self) -> list[str]:
        return _split(self.enabled_text_tools)

    @property
    def image_tools(self) -> list[str]:
        return _split(self.enabled_image_tools)


settings = Settings()


def require_api_key() -> str:
    """Return the key of the configured judge provider, or raise a clear error. Called only when an LLM tool runs."""
    if not settings.llm_api_key:
        name = "GEMINI_API_KEY" if settings.llm_provider.lower() == "gemini" else "ANTHROPIC_API_KEY"
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example) or disable the "
            "LLM tools via ENABLED_TEXT_TOOLS / ENABLED_IMAGE_TOOLS."
        )
    return settings.llm_api_key
