"""Sistema de configuração usando Pydantic Settings para carregar config.yaml e .env"""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuração do LLM"""
    provider: Literal["openai", "deepseek", "qwen", "anthropic", "groq"] = "openai"
    model: str = "gpt-4o"
    api_key: str = Field(default="")
    temperature: float = 0.7
    max_tokens: int = 4000


class UIConfig(BaseSettings):
    """Configuração da interface"""
    theme: str = "dark"
    background: str = "#0A0E27"
    text: str = "#FFFFFF"
    accent_red: str = "#FF3B3B"
    accent_green: str = "#4CAF50"
    accent_cyan: str = "#00D9FF"
    refresh_interval: int = 30


class DatabaseConfig(BaseSettings):
    """Configuração dos bancos de dados"""
    sqlite_path: str = "data/portfolio.db"
    vector_db_path: str = "data/vectordb"


class CacheConfig(BaseSettings):
    """Configuração do Redis"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    ttl_prices: int = 30
    ttl_portfolio: int = 300
    ttl_analysis: int = 3600


class AgentsConfig(BaseSettings):
    """Configuração dos agentes IA"""
    enable_memory: bool = True
    enable_vector_db: bool = False
    learning_mode: bool = True
    background_jobs: bool = False


class Settings(BaseSettings):
    """Configurações globais do aplicativo"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Variáveis de ambiente diretas
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    encryption_key: str = Field(default="", alias="ENCRYPTION_KEY")
    
    # Configurações carregadas do YAML
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """Carrega configurações do config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            return
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        # Substituir variáveis de ambiente no YAML
        if "llm" in config_data:
            llm_data = config_data["llm"]
            if "api_key" in llm_data and llm_data["api_key"].startswith("${"):
                env_var = llm_data["api_key"].strip("${}")
                llm_data["api_key"] = os.getenv(env_var, "")
            self.llm = LLMConfig(**llm_data)
        
        if "ui" in config_data:
            ui_data = config_data["ui"]
            if "colors" in ui_data:
                ui_data.update(ui_data.pop("colors"))
            self.ui = UIConfig(**ui_data)
        
        if "database" in config_data:
            self.database = DatabaseConfig(**config_data["database"])
        
        if "cache" in config_data:
            self.cache = CacheConfig(**config_data["cache"])
        
        if "agents" in config_data:
            self.agents = AgentsConfig(**config_data["agents"])
    
    def get_api_key_for_provider(self, provider: str) -> str:
        """Retorna a API key para o provider especificado"""
        provider_keys = {
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "qwen": self.qwen_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }
        return provider_keys.get(provider, "")


# Instância global de configuração
settings = Settings()
