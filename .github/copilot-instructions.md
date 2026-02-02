# Copilot Instructions for Crypto Portfolio Agno

Este projeto é um dashboard inteligente para análise de portfolios na Binance, utilizando o framework de agentes **Agno** e uma arquitetura multi-agente.

## 🏗️ Architecture & Big Picture
- **Multi-Agent System**: O sistema é orquestrado por um `Maestro` que coordena agentes especializados (`Portfolio Analyzer`, `Risk Manager`, `Market Analyst`, `News & Sentiment Agent`, `MetaLearningOptimizer`, `CryptoExpertAgent`).
- **Data Flow**: Binance API (real-time) -> Redis Cache -> Agent Analysis -> Qdrant Vector DB (persistent memory).
- **Decision Hub**: Todas as decisões arquiteturais estão registradas em arquivos `.md` na raiz, como [ARQUITETURA_AGENTES_IA_PORTFOLIO.md](ARQUITETURA_AGENTES_IA_PORTFOLIO.md) e [DECISOES_CONVERSA_02FEV2026.md](DECISOES_CONVERSA_02FEV2026.md).

## 🛠️ Tech Stack & Conventions
- **Agent Framework**: Use sempre o framework **Agno** para criar e gerenciar agentes.
- **Configuration**: As configurações devem ser acessadas via classe `Settings` em [app/config.py](app/config.py), que carrega dados de `config.yaml` e variáveis de ambiente em `.env`.
- **Validation**: Utilize `Pydantic` para todos os esquemas de dados e tipos de retorno das ferramentas dos agentes.
- **Security**: Chaves de API devem ser tratadas com criptografia (biblioteca `cryptography` já instalada) e nunca expostas.
- **Language**: O código e comentários devem ser em Inglês (conforme [app/config.py](app/config.py)), mas documentação técnica de suporte pode estar em Português.

## 🔄 AI & Optimization Patterns
- **Reinforcement Learning**: Implementar loops de feedback para os agentes melhorarem continuamente.
- **Optimization Algorithms**: 
  - Phase 2: Heurísticas + **Thompson Sampling** (priorizar exploração/explotação rápida).
  - Phase 3: **Bayesian Optimization** (via Optuna) para calibração fina de parâmetros.
- **Memory**: Use Qdrant para armazenamento vetorial de insights históricos e micro-análises.

## 🚀 Critical Workflows
- **Package Management**: Utilize `uv` para instalar dependências e gerenciar o ambiente virtual.
- **Running the app**: `uv run uvicorn main:app --reload` (após implementar a integração FastAPI em `main.py`).
- **Adding a New Agent**:
  1. Definir responsabilidades no [ARQUITETURA_AGENTES_IA_PORTFOLIO.md](ARQUITETURA_AGENTES_IA_PORTFOLIO.md).
  2. Implementar a classe do agente em `app/agents/`.
  3. Registrar ferramentas (tools) específicas que o agente necessita.
  4. Integrar com o `Maestro` agent.

## 📂 Key Files
- [app/config.py](app/config.py): Ponto central de configurações.
- [config.yaml](config.yaml): Definições de ambiente e parâmetros de agentes.
- [pyproject.toml](pyproject.toml): Gerenciamento de dependências e metadados.
