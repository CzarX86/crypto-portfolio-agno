import uvicorn
from fastapi import FastAPI
from app.config import settings
from app.agents.meta_learning_optimizer import MetaLearningOptimizer
from app.agents.crypto_expert import CryptoExpertAgent

app = FastAPI(title="Crypto Portfolio Agno Dashboard")

@app.on_event("startup")
async def startup_event():
    # Inicializar agentes e conexões
    print(f"Iniciando Dashboard para {len(settings.ACTIVE_ASSETS)} ativos")
    # meta_learning = MetaLearningOptimizer(settings)
    # crypto_expert = CryptoExpertAgent(settings)

@app.get("/")
async def root():
    return {"status": "running", "version": "0.1.0-mvp"}

@app.get("/analysis/{asset}")
async def get_asset_analysis(asset: str):
    # Endpoint para solicitar análise de um ativo
    return {"asset": asset, "analysis": "pending"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
