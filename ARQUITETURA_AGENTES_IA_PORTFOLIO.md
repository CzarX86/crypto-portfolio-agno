# Arquitetura Avançada de Agentes IA para Análise de Portfolio Binance

**Data**: Fevereiro 2, 2026 (ATUALIZADO COM DECISÕES FINAIS)  
**Autor**: Pesquisa em Profundidade - Agentes IA Especializados para Investimentos  
**Escopo**: 1-5 ativos, Real-time com Binance

---

## 📋 Índice Executivo

Esta é uma pesquisa em profundidade sobre como construir uma arquitetura de **agentes IA multi-especializados** que analisam portfolios em tempo real, utilizando **Agno**, **Vector Databases**, **APIs de mercado** e **RAG (Retrieval-Augmented Generation)** para criar um sistema inteligente, adaptável e com memória persistente.

**✅ ATUALIZADO EM 02/02/2026**: Integrado **MetaLearningOptimizer** e **CryptoExpertAgent** como agentes MVP; decisão técnica de usar **Bayesian Optimization + Thompson Sampling** ao invés de Genetic Algorithm.

---

## 1. Arquitetura Multi-Agent Especializada com Agno

### 1.1 Times de Agentes Especializados

```
┌─────────────────────────────────────────────────────────┐
│              PORTFOLIO DASHBOARD SYSTEM                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │           ORCHESTRATOR AGENT (Maestro)           │   │
│  │  - Coordena todos os agentes                     │   │
│  │  - Toma decisões estratégicas                    │   │
│  │  - Sintetiza insights                            │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                 │
│         ┌───────────────┼───────────────┐               │
│         │               │               │               │
│    ┌────▼────┐     ┌────▼────┐    ┌────▼────┐          │
│    │Portfolio │     │  Risk   │    │ Market  │          │
│    │Analyzer  │     │Manager  │    │Analyst  │          │
│    │ Agent    │     │ Agent   │    │ Agent   │          │
│    └────┬─────┘     └────┬────┘    └────┬────┘          │
│         │                │              │               │
│    ┌────▼────────────────┼──────────────┴──┐            │
│    │   News & Sentiment  │                 │            │
│    │      Analyzer       │                 │            │
│    └────┬────────────────┼─────────────────┘            │
│         │                │                              │
│    ┌────▼────────────────▼──────────────┐              │
│    │    VECTOR DATABASE (Qdrant)        │              │
│    │  - Histórico de análises           │              │
│    │  - Notícias/insights embarcados    │              │
│    │  - Micro-análises                  │              │
│    └────────────────────────────────────┘              │
│                                                           │
│    ┌────────────────────────────────────┐              │
│    │    MARKET DATA LAYER               │              │
│    │  - Binance API (real-time)         │              │
│    │  - Finnhub (fundamentals)          │              │
│    │  - NewsAPI (notícias)              │              │
│    └────────────────────────────────────┘              │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Perfis de Agentes

#### **Portfolio Analyzer Agent**
- **Responsabilidade**: Analisar composição, diversificação, rentabilidade
- **Ferramentas**: 
  - Cálculo de Sharpe Ratio, Beta, Duration
  - Análise de correlação entre ativos
  - Detecção de drift (desvio da alocação ideal)
- **Memória**: Histórico de rebalanceamentos e decisões
- **Output**: Relatório de health do portfolio

#### **Risk Manager Agent**
- **Responsabilidade**: Identificar riscos, VaR, cenários de stress
- **Ferramentas**:
  - Cálculo de VaR (Value at Risk)
  - Análise de drawdown
  - Teste de stress (cenários black swan)
  - Alert de limites (stop-loss, volatilidade máxima)
- **Memória**: Eventos de risco históricos, padrões de crises
- **Output**: Alertas, recomendações de hedging

#### **Market Analyst Agent**
- **Responsabilidade**: Análise técnica, tendências, correlações
- **Ferramentas**:
  - Indicadores técnicos (SMA, RSI, MACD, Bollinger Bands)
  - Análise de Volume
  - Pattern recognition (Head & Shoulders, Double Bottom)
  - Correlação com macro-indices
- **Memória**: Padrões históricos, validação de sinais
- **Output**: Sinais de compra/venda, tendências

#### **News & Sentiment Agent**
- **Responsabilidade**: Análise de notícias, sentimento de mercado
- **Ferramentas**:
  - NLP para extração de sentimento
  - Classificação de relevância
  - Detecção de anomalias em volume de notícias
  - Rastreamento de temas emergentes
- **Memória**: News embarcadas em vector DB com timestamps
- **Output**: Sentimento agregado, warning de eventos

### 1.3 Padrões de Comunicação Entre Agentes

#### **Publish-Subscribe Pattern**

```python
# Exemplo com Agno (conceitual)
class AgentHub:
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []
    
    def subscribe(self, event_type: str, agent):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(agent)
    
    def publish(self, event_type: str, data: dict):
        """Agent publica um evento que outros agentes recebem"""
        message = {
            "event_type": event_type,
            "timestamp": datetime.now(),
            "data": data,
            "source": data.get("source_agent")
        }
        self.message_queue.append(message)
        
        for agent in self.subscribers.get(event_type, []):
            agent.receive_message(message)

# Exemplo de uso
hub = AgentHub()
portfolio_analyzer = PortfolioAnalyzer()
risk_manager = RiskManager()

# Risk Manager se inscreve em atualizações de portfolio
hub.subscribe("portfolio_updated", risk_manager)

# Portfolio Analyzer publica um evento
hub.publish("portfolio_updated", {
    "portfolio_data": new_portfolio,
    "source_agent": "PortfolioAnalyzer"
})
```

#### **Request-Response Pattern**

```python
class OrchestrationLayer:
    async def analyze_portfolio(self):
        """Maestro coordena análise sincronizada"""
        
        # Request paralelo para múltiplos agentes
        tasks = [
            portfolio_analyzer.analyze(self.portfolio),
            risk_manager.assess_risk(self.portfolio),
            market_analyst.analyze_technicals(self.prices),
            sentiment_agent.analyze_news(self.assets)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sintetizar respostas
        synthesis = self.synthesize_results(results)
        return synthesis
```

### 1.4 Coordenação e Orquestração

#### **Agente Maestro/Orchestrator**

```python
class PortfolioMaestro:
    def __init__(self):
        self.agents = {
            "portfolio": PortfolioAnalyzer(),
            "risk": RiskManager(),
            "market": MarketAnalyst(),
            "sentiment": NewsAgent()
        }
        self.memory = PersistentMemory()
        self.decision_engine = DecisionEngine()
    
    async def daily_analysis_cycle(self):
        """Ciclo diário de análise"""
        
        # 1. Fetch dados
        market_data = await self.fetch_market_data()
        news_data = await self.fetch_news()
        
        # 2. Análises paralelas
        analyses = {
            "portfolio": await self.agents["portfolio"].analyze(market_data),
            "risk": await self.agents["risk"].assess(market_data),
            "market": await self.agents["market"].analyze(market_data),
            "sentiment": await self.agents["sentiment"].analyze(news_data)
        }
        
        # 3. Consolidar insights
        consolidated = self._consolidate_insights(analyses)
        
        # 4. Tomar decisões
        decisions = await self.decision_engine.recommend(consolidated)
        
        # 5. Armazenar micro-análises para aprendizado
        await self.memory.store_micro_analysis({
            "timestamp": datetime.now(),
            "input": {"market_data": market_data, "news_data": news_data},
            "analyses": analyses,
            "decisions": decisions
        })
        
        return decisions
    
    def _consolidate_insights(self, analyses: dict) -> dict:
        """Sintetizar múltiplas perspectivas"""
        return {
            "portfolio_health": analyses["portfolio"].health_score,
            "risk_level": analyses["risk"].risk_score,
            "market_trend": analyses["market"].trend,
            "sentiment": analyses["sentiment"].overall_sentiment,
            "conflicts": self._detect_conflicts(analyses),
            "consensus_action": self._find_consensus(analyses)
        }
    
    def _detect_conflicts(self, analyses: dict) -> list:
        """Quando agentes têm opiniões conflitantes"""
        conflicts = []
        if analyses["market"].recommendation == "BUY" and \
           analyses["risk"].recommendation == "SELL":
            conflicts.append({
                "type": "trend_vs_risk",
                "severity": "high",
                "description": "Mercado em alta mas risco aumentando"
            })
        return conflicts
```

---

## 2. Memória e Aprendizado em Agentes

### 2.1 Arquitetura de Memória Persistente

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from datetime import datetime
import sqlite3
from sentence_transformers import SentenceTransformer

@dataclass
class MicroAnalysis:
    """Representa uma análise pequena e atômica"""
    id: str
    timestamp: datetime
    asset: str
    analysis_type: str  # "technical", "sentiment", "fundamental"
    context: Dict[str, Any]  # Dados de entrada
    output: Dict[str, Any]  # Resultado da análise
    confidence: float  # 0-1
    was_accurate: bool = None  # Feedback
    accuracy_score: float = None
    
    def to_embedding(self, model):
        """Converter para embedding vetorial"""
        text = f"{self.analysis_type} {self.asset}: {json.dumps(self.output)}"
        return model.encode(text)

class PersistentMemory:
    """Sistema de memória que evolui com o tempo"""
    
    def __init__(self, db_path="portfolio_memory.db", vector_db_path="memories.db"):
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_databases()
    
    def _init_databases(self):
        """Inicializar SQLite e Qdrant"""
        # SQLite para structured data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS micro_analyses (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                asset TEXT,
                analysis_type TEXT,
                context JSON,
                output JSON,
                confidence REAL,
                was_accurate BOOLEAN,
                accuracy_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_insights (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                agent_name TEXT,
                insight TEXT,
                supporting_data JSON,
                reliability_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                decision TEXT,
                reasoning JSON,
                outcome REAL,  # Rentabilidade resultante
                feedback_received BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_micro_analysis(self, analysis: MicroAnalysis):
        """Armazenar análise atômica"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO micro_analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis.id,
            analysis.timestamp,
            analysis.asset,
            analysis.analysis_type,
            json.dumps(analysis.context),
            json.dumps(analysis.output),
            analysis.confidence,
            analysis.was_accurate,
            analysis.accuracy_score
        ))
        
        conn.commit()
        
        # Também armazenar embedding no vector DB
        embedding = analysis.to_embedding(self.embedding_model)
        await self._store_vector(
            analysis.id,
            embedding,
            {
                "asset": analysis.asset,
                "type": analysis.analysis_type,
                "timestamp": analysis.timestamp.isoformat()
            }
        )
    
    async def retrieve_similar_analyses(self, query: str, asset: str = None, top_k: int = 5):
        """Recuperar análises similares via embedding"""
        query_embedding = self.embedding_model.encode(query)
        
        # Qdrant similarity search
        similar = await self._vector_search(query_embedding, top_k, asset)
        return similar
    
    async def get_agent_performance(self, agent_name: str, lookback_days: int = 30):
        """Avaliar performance de um agente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT reliability_score, COUNT(*) as count 
            FROM agent_insights 
            WHERE agent_name = ? AND timestamp > datetime('now', '-' || ? || ' days')
            GROUP BY reliability_score
        ''', (agent_name, lookback_days))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            avg_reliability = sum(r[0] * r[1] for r in results) / sum(r[1] for r in results)
            return {"agent": agent_name, "avg_reliability": avg_reliability, "data_points": results}
        return None

class MemoryEvolutionEngine:
    """Motor que aprende com o histórico"""
    
    def __init__(self, memory: PersistentMemory):
        self.memory = memory
    
    async def feedback_loop(self, decision_id: str, outcome: float):
        """Receber feedback sobre uma decisão"""
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        # Atualizar outcome da decision
        cursor.execute('''
            UPDATE decisions 
            SET outcome = ?, feedback_received = TRUE 
            WHERE id = ?
        ''', (outcome, decision_id))
        
        conn.commit()
        conn.close()
        
        # Usar feedback para melhorar agentes
        await self._retrain_agent_models(decision_id, outcome)
    
    async def identify_patterns(self, asset: str, analysis_type: str):
        """Identificar padrões que funcionam"""
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT output, outcome 
            FROM micro_analyses ma
            JOIN decisions d ON ma.id = d.id
            WHERE ma.asset = ? AND ma.analysis_type = ?
            ORDER BY ma.timestamp DESC LIMIT 100
        ''', (asset, analysis_type))
        
        results = cursor.fetchall()
        conn.close()
        
        # Análise de correlação output -> outcome
        patterns = self._extract_patterns(results)
        return patterns
    
    async def adaptive_prompt_engineering(self, agent_type: str, performance_data: dict):
        """Melhorar prompts dos agentes baseado em performance"""
        if performance_data['avg_reliability'] < 0.7:
            # Sistema detects baixa confiança
            new_prompt = await self._generate_improved_prompt(
                agent_type, 
                performance_data
            )
            return new_prompt
        return None
```

### 2.2 Feedback Loops e Re-training

```python
class ContinuousLearning:
    """Sistema de aprendizado contínuo para agentes"""
    
    def __init__(self, maestro: PortfolioMaestro, memory: PersistentMemory):
        self.maestro = maestro
        self.memory = memory
        self.evolution_engine = MemoryEvolutionEngine(memory)
    
    async def daily_review_cycle(self):
        """Ao final do dia, revisar decisões e aprender"""
        
        # 1. Coletar outcomes do dia
        daily_decisions = await self.memory.get_decisions_from_today()
        
        for decision in daily_decisions:
            # 2. Calcular resultado real
            actual_return = await self.calculate_actual_return(decision)
            
            # 3. Enviar feedback
            await self.evolution_engine.feedback_loop(decision.id, actual_return)
            
            # 4. Atualizar confiança do agente
            await self._update_agent_confidence(decision, actual_return)
        
        # 5. Re-avaliar performance de cada agente
        for agent_name in ["portfolio", "risk", "market", "sentiment"]:
            performance = await self.memory.get_agent_performance(agent_name, lookback_days=7)
            
            if performance and performance["avg_reliability"] < 0.65:
                # Agente precisa melhorar
                await self._intervene_agent_performance(agent_name, performance)
    
    async def _intervene_agent_performance(self, agent_name: str, performance: dict):
        """Intervenção quando agent está com baixa performance"""
        
        # Exemplos de intervenção:
        if agent_name == "market":
            # Market analyst está errando sinais técnicos
            await self._update_market_analyst_thresholds(performance)
        
        elif agent_name == "sentiment":
            # News agent está com false positives
            await self._retune_sentiment_model(performance)
        
        elif agent_name == "risk":
            # Risk manager não está capturando riscos
            await self._increase_risk_sensitivity(performance)
```

---

## 3. Vector Database Local para Portfolio Analysis

### 3.1 Matriz de Comparação: Vector DBs Locais

| **Critério** | **Qdrant** | **Milvus Lite** | **SQLite Vector** |
|---|---|---|---|
| **Instalação** | Docker / Standalone | Pip install | Built-in |
| **Performance (1K vecs)** | 🟢 ~5ms | 🟢 ~8ms | 🔴 ~50ms |
| **Performance (1M vecs)** | 🟢 ~20ms | 🟢 ~30ms | 🔴 Problematic |
| **Memória Base** | ~100MB | ~50MB | ~5MB |
| **Escalabilidade** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **Metadata Filtering** | ✅ Sim (excelente) | ✅ Sim | ✅ Sim |
| **Sync Replication** | ✅ Enterprise | ❌ Não | N/A |
| **Backup/Restore** | ✅ Automático | ✅ Manual | ✅ Standard |
| **Curva Aprendizado** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Ideal Para** | Produção escalável | Prototipagem rápida | Quick & dirty |

### ✅ Recomendação para seu caso (1-5 ativos):
**Qdrant** - Trade-off ótimo entre performance, features e escalabilidade futura

### 3.2 Implementação com Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from typing import List, Dict, Any

class PortfolioVectorDB:
    """Vector DB para portfolio analysis"""
    
    def __init__(self, collection_name: str = "portfolio_analyses"):
        # Conectar a Qdrant (local ou cloud)
        self.client = QdrantClient(":memory:")  # Local in-memory para dev
        # Para produção: QdrantClient(host="localhost", port=6333)
        
        self.collection_name = collection_name
        self.embedding_dim = 384  # all-MiniLM-L6-v2
        self._init_collections()
    
    def _init_collections(self):
        """Criar coleções para diferentes tipos de dados"""
        
        collections = {
            "technical_analyses": "Análises técnicas de ativos",
            "news_embeddings": "Embeddings de notícias",
            "sentiment_data": "Dados de sentimento",
            "historical_patterns": "Padrões históricos"
        }
        
        for col_name, description in collections.items():
            try:
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    ),
                    # Payload config para metadata filtering
                    payload_index_config={
                        "text_index": {"type": "text"},
                        "datetime_index": {"type": "datetime"}
                    }
                )
            except:
                pass  # Collection já existe
    
    async def store_technical_analysis(self, asset: str, analysis: dict, embedding: np.array):
        """Armazenar análise técnica"""
        
        point = PointStruct(
            id=self._generate_id(),
            vector=embedding.tolist(),
            payload={
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "rsi": analysis.get("rsi"),
                "macd": analysis.get("macd"),
                "sma_20": analysis.get("sma_20"),
                "sma_200": analysis.get("sma_200"),
                "trend": analysis.get("trend"),
                "signal_strength": analysis.get("signal_strength"),
                "price": analysis.get("price"),
                "volume": analysis.get("volume")
            }
        )
        
        self.client.upsert(
            collection_name="technical_analyses",
            points=[point]
        )
    
    async def store_news(self, asset: str, news: dict, embedding: np.array):
        """Armazenar notícia com embedding"""
        
        point = PointStruct(
            id=self._generate_id(),
            vector=embedding.tolist(),
            payload={
                "asset": asset,
                "title": news.get("title"),
                "content": news.get("content"),
                "source": news.get("source"),
                "published_at": news.get("published_at"),
                "sentiment": news.get("sentiment"),  # -1 to 1
                "relevance_score": news.get("relevance_score"),
                "url": news.get("url")
            }
        )
        
        self.client.upsert(
            collection_name="news_embeddings",
            points=[point]
        )
    
    async def search_similar_analyses(self, query_embedding: np.array, asset: str = None, top_k: int = 5):
        """Recuperar análises similares com filtering"""
        
        query_filter = None
        if asset:
            query_filter = {
                "must": [
                    {"key": "asset", "match": {"value": asset}}
                ]
            }
        
        results = self.client.search(
            collection_name="technical_analyses",
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k
        )
        
        return results
    
    async def search_news_by_sentiment(self, asset: str, sentiment_min: float = 0.0, top_k: int = 10):
        """Buscar notícias positivas sobre um ativo"""
        
        results = self.client.scroll(
            collection_name="news_embeddings",
            scroll_filter={
                "must": [
                    {"key": "asset", "match": {"value": asset}},
                    {"key": "sentiment", "range": {"gte": sentiment_min}}
                ]
            },
            limit=top_k
        )
        
        return results
    
    async def find_patterns(self, pattern_type: str = "bullish", asset: str = None):
        """Encontrar padrões históricos similares"""
        
        # Buscar por padrões que levaram a resultados positivos
        results = self.client.scroll(
            collection_name="historical_patterns",
            scroll_filter={
                "must": [
                    {"key": "pattern_type", "match": {"value": pattern_type}},
                    {"key": "outcome_positive", "match": {"value": True}}
                ] + ([{"key": "asset", "match": {"value": asset}}] if asset else [])
            },
            limit=20
        )
        
        return results
```

### 3.3 RAG para Contextualização de Agentes

```python
from sentence_transformers import SentenceTransformer

class PortfolioRAG:
    """Retrieval-Augmented Generation para análise de portfolio"""
    
    def __init__(self, vector_db: PortfolioVectorDB, llm_client):
        self.vector_db = vector_db
        self.llm = llm_client
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def contextualize_market_analysis(self, asset: str, current_price: float):
        """Recuperar contexto histórico antes de fazer análise"""
        
        # 1. Gerar embedding da query
        query = f"Análise técnica e padrões históricos para {asset} a {current_price}"
        query_embedding = self.embedder.encode(query)
        
        # 2. Recuperar contexto relevante
        similar_analyses = await self.vector_db.search_similar_analyses(
            query_embedding, 
            asset=asset, 
            top_k=5
        )
        
        relevant_news = await self.vector_db.search_news_by_sentiment(
            asset=asset, 
            sentiment_min=0.0,  # Todas as notícias
            top_k=10
        )
        
        historical_patterns = await self.vector_db.find_patterns(
            pattern_type="bullish",
            asset=asset
        )
        
        # 3. Formatar contexto para LLM
        context = f"""
        ANÁLISE HISTÓRICA PARA {asset}:
        
        ANÁLISES TÉCNICAS SIMILARES:
        {self._format_analyses(similar_analyses)}
        
        NOTÍCIAS RECENTES:
        {self._format_news(relevant_news)}
        
        PADRÕES HISTÓRICOS POSITIVOS:
        {self._format_patterns(historical_patterns)}
        
        PREÇO ATUAL: ${current_price}
        """
        
        # 4. Usar contexto para melhorar LLM response
        prompt = f"""
        {context}
        
        Considerando o histórico acima, forneça uma análise sobre {asset}.
        Quais sinais técnicos históricos se repetem?
        Como o sentimento do mercado comparado ao histórico?
        """
        
        response = await self.llm.generate(prompt)
        return response
    
    async def contextualize_risk_assessment(self, asset: str):
        """Recuperar contexto de risco histórico"""
        
        # Buscar máximos drawdowns históricos
        worst_days = await self.vector_db.client.scroll(
            collection_name="technical_analyses",
            scroll_filter={
                "must": [
                    {"key": "asset", "match": {"value": asset}},
                    {"key": "volume", "range": {"gte": 1000000}}  # Alto volume
                ]
            }
        )
        
        # Buscar notícias negativas
        negative_news = await self.vector_db.search_news_by_sentiment(
            asset=asset,
            sentiment_min=-1.0,
            top_k=10
        )
        
        context = f"""
        CONTEXTO DE RISCO PARA {asset}:
        
        DIAS COM MAIOR MOVIMENTO:
        {self._format_analyses(worst_days)}
        
        NOTÍCIAS NEGATIVAS HISTÓRICAS:
        {self._format_news(negative_news)}
        """
        
        return context
    
    def _format_analyses(self, analyses: List) -> str:
        """Formatar análises para texto legível"""
        return "\n".join([
            f"- {a.payload['timestamp']}: RSI={a.payload['rsi']:.2f}, " \
            f"Trend={a.payload['trend']}, Price={a.payload['price']}"
            for a in analyses
        ])
    
    def _format_news(self, news: List) -> str:
        """Formatar notícias para texto legível"""
        return "\n".join([
            f"- [{n.payload['source']}] {n.payload['title']} " \
            f"(Sentiment: {n.payload['sentiment']:.2f})"
            for n in news
        ])
    
    def _format_patterns(self, patterns: List) -> str:
        """Formatar padrões para texto legível"""
        return "\n".join([
            f"- {p.payload['pattern_type']}: " \
            f"Ocorreu {p.payload['occurrences']} vezes, " \
            f"Sucesso: {p.payload['success_rate']:.1%}"
            for p in patterns
        ])
```

---

## 4. Context Awareness - Tendências, Cotações, Volumes em Tempo Real

### 4.1 Arquitetura de Dados em Tempo Real

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import websocket
import json

class RealtimeMarketListener:
    """Listener para dados de mercado em tempo real"""
    
    def __init__(self, exchange_api):
        self.exchange = exchange_api  # Binance, etc
        self.subscribers = {}
        self.cache = {}
        self.last_update = {}
    
    async def start_listening(self, assets: List[str]):
        """Iniciar websocket para stream de dados"""
        
        # Para Binance
        ws_url = "wss://stream.binance.com:9443/ws"
        
        # Subscribe a multiplos pares
        streams = [f"{asset.lower()}@kline_1m" for asset in assets]
        
        tasks = [
            self._listen_stream(ws_url, stream) 
            for stream in streams
        ]
        
        await asyncio.gather(*tasks)
    
    async def _listen_stream(self, ws_url: str, stream: str):
        """Listener para um stream específico"""
        
        try:
            async with websockets.connect(f"{ws_url}/{stream}") as ws:
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    # Publicar para subscribers
                    await self._publish_update(stream, data)
        except Exception as e:
            print(f"Error in stream {stream}: {e}")
    
    async def _publish_update(self, stream: str, data: dict):
        """Publicar update para interessados"""
        
        for subscriber in self.subscribers.get(stream, []):
            await subscriber.on_market_update(data)
        
        # Atualizar cache
        self.cache[stream] = data
        self.last_update[stream] = datetime.now()
    
    def subscribe(self, stream: str, agent):
        """Um agente se inscreve em atualizações"""
        if stream not in self.subscribers:
            self.subscribers[stream] = []
        self.subscribers[stream].append(agent)

class TechnicalIndicatorCalculator:
    """Calcular indicadores técnicos em tempo real"""
    
    def __init__(self):
        self.cache = {}  # Cache de valores calculados
        self.prices_history = {}  # Histórico para cálculos
    
    async def calculate_indicators(self, asset: str, price_data: dict) -> dict:
        """Calcular SMA, RSI, MACD, Bollinger Bands"""
        
        # Manter histórico
        if asset not in self.prices_history:
            self.prices_history[asset] = []
        
        self.prices_history[asset].append({
            "timestamp": datetime.now(),
            "close": price_data['close'],
            "high": price_data['high'],
            "low": price_data['low'],
            "volume": price_data['volume']
        })
        
        # Manter apenas últimos 200 candles
        if len(self.prices_history[asset]) > 200:
            self.prices_history[asset].pop(0)
        
        closes = [p['close'] for p in self.prices_history[asset]]
        
        indicators = {
            "sma_20": self._calculate_sma(closes, 20),
            "sma_50": self._calculate_sma(closes, 50),
            "sma_200": self._calculate_sma(closes, 200),
            "rsi_14": self._calculate_rsi(closes, 14),
            "macd": self._calculate_macd(closes),
            "bollinger_bands": self._calculate_bollinger_bands(closes, 20),
            "atr": self._calculate_atr(self.prices_history[asset], 14),
            "volume_sma": self._calculate_volume_sma(self.prices_history[asset], 20)
        }
        
        return indicators
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        seed = deltas[:period]
        up = sum([d for d in seed if d > 0]) / period
        down = sum([-d for d in seed if d < 0]) / period
        
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float]):
        """MACD (Moving Average Convergence Divergence)"""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        if ema_12 is None or ema_26 is None:
            return None
        
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema([macd_line], 9)
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line if signal_line else None
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int):
        """Bollinger Bands"""
        if len(prices) < period:
            return None
        
        sma = self._calculate_sma(prices, period)
        variance = sum([(p - sma) ** 2 for p in prices[-period:]]) / period
        std_dev = variance ** 0.5
        
        return {
            "upper": sma + (std_dev * 2),
            "middle": sma,
            "lower": sma - (std_dev * 2),
            "std_dev": std_dev
        }
    
    def _calculate_ema(self, prices: List[float], period: int):
        """Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_atr(self, ohlc_data: List[dict], period: int):
        """Average True Range"""
        if len(ohlc_data) < period:
            return None
        
        true_ranges = []
        for i in range(1, len(ohlc_data)):
            high = ohlc_data[i]['high']
            low = ohlc_data[i]['low']
            close_prev = ohlc_data[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev)
            )
            true_ranges.append(tr)
        
        atr = sum(true_ranges[-period:]) / period
        return atr
    
    def _calculate_volume_sma(self, ohlc_data: List[dict], period: int):
        """Volume Moving Average"""
        if len(ohlc_data) < period:
            return None
        
        volumes = [d['volume'] for d in ohlc_data[-period:]]
        return sum(volumes) / period

class ContextAwarenessEngine:
    """Motor de context awareness que alimenta agentes"""
    
    def __init__(self):
        self.market_listener = None
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.current_context = {}
    
    async def update_context(self, assets: List[str]):
        """Atualizar contexto continuamente"""
        
        while True:
            for asset in assets:
                # Buscar dados mais recentes
                market_data = await self._fetch_market_data(asset)
                
                # Calcular indicadores
                indicators = await self.indicator_calculator.calculate_indicators(
                    asset, 
                    market_data
                )
                
                # Atualizar contexto global
                self.current_context[asset] = {
                    "price": market_data['close'],
                    "volume": market_data['volume'],
                    "indicators": indicators,
                    "timestamp": datetime.now(),
                    "trend": self._determine_trend(indicators)
                }
            
            # Atualizar a cada 1 minuto
            await asyncio.sleep(60)
    
    def _determine_trend(self, indicators: dict) -> str:
        """Determinar trend baseado em indicadores"""
        
        sma_20 = indicators.get("sma_20")
        sma_200 = indicators.get("sma_200")
        rsi = indicators.get("rsi_14")
        
        if sma_20 and sma_200 and sma_20 > sma_200:
            if rsi and rsi > 70:
                return "STRONG_UPTREND"
            return "UPTREND"
        
        elif sma_20 and sma_200 and sma_20 < sma_200:
            if rsi and rsi < 30:
                return "STRONG_DOWNTREND"
            return "DOWNTREND"
        
        else:
            return "NEUTRAL"
    
    def get_context_snapshot(self, asset: str) -> dict:
        """Snapshot do contexto atual para um agente"""
        return self.current_context.get(asset, {})
    
    async def _fetch_market_data(self, asset: str) -> dict:
        """Buscar dados de mercado (implementar com sua exchange API)"""
        # Exemplo com Binance
        pass
```

### 4.2 Integração com APIs de Dados Financeiros

```python
import aiohttp
import pandas as pd
from typing import Optional

class FinancialDataIntegrator:
    """Integração com múltiplas APIs de dados financeiros"""
    
    def __init__(self):
        self.finnhub_key = "YOUR_FINNHUB_KEY"
        self.alpha_vantage_key = "YOUR_ALPHA_VANTAGE_KEY"
        self.newsapi_key = "YOUR_NEWSAPI_KEY"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    # ==================== FINNHUB ====================
    async def get_company_fundamentals(self, symbol: str) -> dict:
        """Obter dados fundamentais via Finnhub"""
        
        url = f"https://finnhub.io/api/v1/quote"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            return {
                "price": data.get('c'),
                "high_day": data.get('h'),
                "low_day": data.get('l'),
                "open": data.get('o'),
                "previous_close": data.get('pc'),
                "pe_ratio": data.get('pe'),  # Se disponível
                "market_cap": data.get('marketCap'),
                "timestamp": pd.Timestamp.now()
            }
    
    async def get_earnings_estimate(self, symbol: str) -> dict:
        """Obter estimativas de earnings"""
        
        url = f"https://finnhub.io/api/v1/stock/earnings"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            return {
                "current_estimate": data[0].get('epsEstimate') if data else None,
                "surprise": data[0].get('surprise') if data else None,
                "report_date": data[0].get('reportDate') if data else None
            }
    
    # ==================== NEWSAPI ====================
    async def fetch_news(self, symbol: str, language: str = "en") -> List[dict]:
        """Buscar notícias sobre um ativo"""
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": language,
            "sort_by": "publishedAt",
            "page_size": 10,
            "apiKey": self.newsapi_key
        }
        
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            return data.get('articles', [])
    
    # ==================== ALPHA VANTAGE ====================
    async def get_technical_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Obter dados técnicos (OHLCV)"""
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full"
        }
        
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            
            # Parse resposta
            time_series_key = f"Time Series ({interval})"
            if time_series_key in data:
                df = pd.DataFrame([
                    {
                        "timestamp": ts,
                        "open": float(v['1. open']),
                        "high": float(v['2. high']),
                        "low": float(v['3. low']),
                        "close": float(v['4. close']),
                        "volume": float(v['5. volume'])
                    }
                    for ts, v in data[time_series_key].items()
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            
            return pd.DataFrame()

# Exemplo de uso
async def example_data_integration():
    async with FinancialDataIntegrator() as integrator:
        # Fundamentals
        btc_data = await integrator.get_company_fundamentals("AAPL")
        print(f"Apple Price: ${btc_data['price']}")
        
        # News
        news = await integrator.fetch_news("Apple")
        for article in news[:3]:
            print(f"- {article['title']}")
        
        # Technical
        technical = await integrator.get_technical_data("AAPL", "1min")
        print(technical.tail())
```

---

## 5. Integração com APIs de Notícias Financeiras

### 5.1 APIs Recomendadas

| **API** | **Preço** | **Latência** | **Cobertura** | **Features** | **Ideal Para** |
|---|---|---|---|---|---|
| **NewsAPI** | Free: $0, Pro: $299/mês | 5-10 min | 40k+ fontes | Filtros básicos | Prototipagem |
| **Finnhub** | Free: $0, Pro: $399/ano | Real-time | 15k+ fontes | News + Market data | Produção |
| **Yahoo Finance** | Free (scraping) | 15-20 min | Web scraping | Notícias + análise | Backup |
| **CoinGecko** | Free: $0, Pro: $499/ano | 30-60 seg | Crypto focus | Market + news | Crypto assets |
| **Sentieo** | Custom pricing | Real-time | Premium sources | Earnings calls | Enterprise |

### ✅ Recomendação: **Finnhub** (balanceado) + **NewsAPI** (backup)

### 5.2 Análise de Sentimento em Notícias

```python
from transformers import pipeline
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

class NewsAnalyzer:
    """Análise de notícias com NLP e sentimento"""
    
    def __init__(self):
        # Modelos de sentimento
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Extração de entidades (para encontrar menções de ativos)
        self.ner_model = pipeline(
            "named-entity-recognition",
            model="dslim/bert-base-uncased-ner"
        )
        
        self.asset_mentions = {}  # Cache de menções
    
    async def analyze_news_article(self, article: dict) -> dict:
        """Analisar um artigo de notícia"""
        
        title = article.get('title', '')
        content = article.get('content', '')
        source = article.get('source', {}).get('name', '')
        published_at = article.get('publishedAt', '')
        
        # 1. Sentimento do headline (mais importante)
        title_sentiment = await self._analyze_sentiment(title)
        
        # 2. Sentimento do conteúdo
        content_sentiment = await self._analyze_sentiment_multi_sentence(content)
        
        # 3. Extrair entidades (empresas mencionadas)
        entities = await self._extract_entities(content)
        
        # 4. Detectar relevância (quão relevante para portfolio)
        relevance_score = await self._calculate_relevance(
            title, 
            content, 
            entities
        )
        
        # 5. Calcular sentimento composto
        overall_sentiment = (
            title_sentiment['score'] * 0.6 +  # 60% peso no headline
            content_sentiment * 0.4  # 40% no conteúdo
        ) if title_sentiment else 0
        
        return {
            "title": title,
            "source": source,
            "published_at": published_at,
            "title_sentiment": title_sentiment,
            "content_sentiment": content_sentiment,
            "entities": entities,
            "relevance_score": relevance_score,
            "overall_sentiment": overall_sentiment,
            "url": article.get('url'),
            "analyzed_at": pd.Timestamp.now()
        }
    
    async def _analyze_sentiment(self, text: str) -> dict:
        """Análise de sentimento de um texto"""
        
        if not text or len(text) < 10:
            return None
        
        # Limitar a 512 tokens
        text = text[:512]
        
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Converter para escala -1 a 1
            if result['label'] == 'POSITIVE':
                score = result['score']
            else:
                score = -result['score']
            
            return {
                "label": result['label'],
                "score": score,
                "confidence": result['score']
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None
    
    async def _analyze_sentiment_multi_sentence(self, text: str) -> float:
        """Sentimento agregado de múltiplas sentenças"""
        
        if not text:
            return 0
        
        sentences = sent_tokenize(text)[:10]  # Primeiras 10 sentenças
        sentiments = []
        
        for sentence in sentences:
            sentiment = await self._analyze_sentiment(sentence)
            if sentiment:
                sentiments.append(sentiment['score'])
        
        return sum(sentiments) / len(sentiments) if sentiments else 0
    
    async def _extract_entities(self, text: str) -> Dict[str, int]:
        """Extrair entidades nomeadas (empresas)"""
        
        try:
            entities = self.ner_model(text)
            
            # Agrupar por entidade
            entity_counts = {}
            for entity in entities:
                word = entity['word']
                entity_type = entity['entity']
                
                if entity_type == "B-ORG":  # Organization
                    entity_counts[word] = entity_counts.get(word, 0) + 1
            
            return entity_counts
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}
    
    async def _calculate_relevance(self, title: str, content: str, entities: dict) -> float:
        """Calcular relevância para o portfolio"""
        
        # Fatores de relevância
        relevance = 0
        
        # 1. Menção de ativos do portfolio (varia)
        for entity in entities:
            if any(ticker in title + content for ticker in ["BTC", "ETH", "STOCKS"]):
                relevance += 0.3
        
        # 2. Palavras-chave financeiras
        financial_keywords = [
            "price", "surge", "crash", "earnings", "profit", "loss",
            "acquisition", "merger", "dividend", "split", "bankruptcy"
        ]
        
        text_combined = (title + " " + content).lower()
        keyword_count = sum(1 for kw in financial_keywords if kw in text_combined)
        relevance += min(keyword_count * 0.1, 0.5)
        
        # 3. Fonte credibilidade
        trusted_sources = ["Reuters", "Bloomberg", "CNBC", "Financial Times", "WSJ"]
        source_boost = 0.1 if any(s in title for s in trusted_sources) else 0
        relevance += source_boost
        
        return min(relevance, 1.0)
    
    async def aggregate_sentiment_by_asset(self, news_list: List[dict], asset: str) -> dict:
        """Agregar sentimento por ativo"""
        
        relevant_news = [n for n in news_list if asset in n.get('entities', {})]
        
        if not relevant_news:
            return {"asset": asset, "sentiment": 0, "count": 0}
        
        avg_sentiment = sum(n['overall_sentiment'] for n in relevant_news) / len(relevant_news)
        
        return {
            "asset": asset,
            "sentiment": avg_sentiment,
            "count": len(relevant_news),
            "range": (
                min(n['overall_sentiment'] for n in relevant_news),
                max(n['overall_sentiment'] for n in relevant_news)
            ),
            "articles": relevant_news
        }
```

---

## 6. Micro-análises para Treinamento

### 6.1 Estrutura de Micro-análise (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MicroAnalysis",
  "description": "Análise atômica pequena para treinamento",
  "type": "object",
  "required": ["id", "timestamp", "asset", "analysis_type", "context", "output", "confidence"],
  "properties": {
    "id": {
      "type": "string",
      "description": "UUID único da análise"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Quando foi gerada"
    },
    "asset": {
      "type": "string",
      "description": "BTC, ETH, AAPL, etc"
    },
    "analysis_type": {
      "type": "string",
      "enum": ["technical", "sentiment", "fundamental", "risk", "pattern"],
      "description": "Tipo de análise"
    },
    "context": {
      "type": "object",
      "description": "Dados de entrada que levaram à análise",
      "properties": {
        "price_action": {
          "type": "object",
          "properties": {
            "current_price": {"type": "number"},
            "sma_20": {"type": "number"},
            "sma_200": {"type": "number"},
            "rsi": {"type": "number"},
            "volume_ratio": {"type": "number"}
          }
        },
        "time_period": {
          "type": "string",
          "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
          "description": "Timeframe da análise"
        },
        "market_conditions": {
          "type": "object",
          "properties": {
            "volatility": {"type": "number", "minimum": 0},
            "trend": {"type": "string", "enum": ["uptrend", "downtrend", "ranging"]},
            "news_sentiment": {"type": "number", "minimum": -1, "maximum": 1}
          }
        }
      }
    },
    "output": {
      "type": "object",
      "description": "Resultado da análise",
      "properties": {
        "signal": {
          "type": "string",
          "enum": ["BUY", "SELL", "HOLD", "WAIT"],
          "description": "Sinal gerado"
        },
        "reasoning": {
          "type": "string",
          "description": "Explicação em natural language"
        },
        "target_price": {"type": ["number", "null"]},
        "stop_loss": {"type": ["number", "null"]},
        "probability": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Confiança do sinal"
        }
      }
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Confiança geral da análise"
    },
    "feedback": {
      "type": "object",
      "description": "Feedback quando o resultado é conhecido",
      "properties": {
        "was_accurate": {"type": "boolean"},
        "accuracy_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "realized_return": {"type": "number"},
        "notes": {"type": "string"}
      }
    }
  },
  "examples": [
    {
      "id": "micro-001-btc-tech",
      "timestamp": "2026-02-01T14:30:00Z",
      "asset": "BTC",
      "analysis_type": "technical",
      "context": {
        "price_action": {
          "current_price": 45230.50,
          "sma_20": 44890.00,
          "sma_200": 42500.00,
          "rsi": 68.5,
          "volume_ratio": 1.8
        },
        "time_period": "1h",
        "market_conditions": {
          "volatility": 0.75,
          "trend": "uptrend",
          "news_sentiment": 0.65
        }
      },
      "output": {
        "signal": "BUY",
        "reasoning": "Price acima de SMA20 e SMA200, RSI em 68.5 (quase sobrecomprado), volume 80% acima da média. Padrão bullish.",
        "target_price": 46500,
        "stop_loss": 44500,
        "probability": 0.72
      },
      "confidence": 0.78,
      "feedback": {
        "was_accurate": true,
        "accuracy_score": 0.85,
        "realized_return": 0.028,
        "notes": "Sinal acionado, atingiu target em 2h"
      }
    }
  ]
}
```

### 6.2 Sistema de Armazenamento e Evaluação

```python
import json
import sqlite3
from datetime import datetime, timedelta
from uuid import uuid4

class MicroAnalysisEngine:
    """Motor para criar, armazenar e avaliar micro-análises"""
    
    def __init__(self, db_path: str = "micro_analyses.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializar banco de micro-análises"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS micro_analyses (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                asset TEXT,
                analysis_type TEXT,
                context JSON,
                output JSON,
                confidence REAL,
                was_accurate BOOLEAN,
                accuracy_score REAL,
                realized_return REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_performance (
                analysis_type TEXT,
                accuracy_rate REAL,
                false_positive_rate REAL,
                avg_return REAL,
                count INTEGER,
                last_updated DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def create_micro_analysis(
        self,
        asset: str,
        analysis_type: str,
        context: dict,
        output: dict,
        confidence: float
    ) -> str:
        """Criar uma micro-análise"""
        
        analysis_id = str(uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO micro_analyses 
            (id, timestamp, asset, analysis_type, context, output, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            datetime.now(),
            asset,
            analysis_type,
            json.dumps(context),
            json.dumps(output),
            confidence
        ))
        
        conn.commit()
        conn.close()
        
        return analysis_id
    
    async def provide_feedback(
        self,
        analysis_id: str,
        was_accurate: bool,
        accuracy_score: float,
        realized_return: float
    ):
        """Fornecer feedback sobre uma análise"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE micro_analyses
            SET was_accurate = ?, accuracy_score = ?, realized_return = ?
            WHERE id = ?
        ''', (was_accurate, accuracy_score, realized_return, analysis_id))
        
        conn.commit()
        conn.close()
    
    async def get_performance_by_type(self, lookback_days: int = 30) -> dict:
        """Performance agregada por tipo de análise"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        cursor.execute('''
            SELECT 
                analysis_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN was_accurate = 1 THEN realized_return ELSE -0.02 END) as avg_return,
                AVG(accuracy_score) as avg_accuracy
            FROM micro_analyses
            WHERE timestamp > ?
            GROUP BY analysis_type
        ''', (cutoff_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        performance = {}
        for row in results:
            analysis_type, total, correct, avg_return, avg_accuracy = row
            performance[analysis_type] = {
                "accuracy_rate": correct / total if total > 0 else 0,
                "count": total,
                "avg_return": avg_return or 0,
                "avg_accuracy_score": avg_accuracy or 0
            }
        
        return performance
    
    async def identify_best_agents(self, analysis_type: str, lookback_days: int = 30):
        """Identificar que tipo de análise está performando melhor"""
        
        performance = await self.get_performance_by_type(lookback_days)
        
        if analysis_type in performance:
            perf = performance[analysis_type]
            return {
                "type": analysis_type,
                "accuracy": perf['accuracy_rate'],
                "recommendation": "INCREASE_WEIGHT" if perf['accuracy_rate'] > 0.65 else "DECREASE_WEIGHT"
            }
        
        return None
    
    async def pattern_analysis(self, asset: str, analysis_type: str):
        """Analisar padrões de sucesso/fracasso"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buscar análises bem-sucedidas
        cursor.execute('''
            SELECT context, output, accuracy_score
            FROM micro_analyses
            WHERE asset = ? AND analysis_type = ? AND was_accurate = 1
            ORDER BY accuracy_score DESC
            LIMIT 50
        ''', (asset, analysis_type))
        
        successful = cursor.fetchall()
        
        # Buscar análises que falharam
        cursor.execute('''
            SELECT context, output, accuracy_score
            FROM micro_analyses
            WHERE asset = ? AND analysis_type = ? AND was_accurate = 0
            ORDER BY accuracy_score ASC
            LIMIT 50
        ''', (asset, analysis_type))
        
        failed = cursor.fetchall()
        conn.close()
        
        # Extrair padrões
        successful_contexts = [json.loads(s[0]) for s in successful]
        failed_contexts = [json.loads(f[0]) for f in failed]
        
        patterns = {
            "successful_pattern": self._extract_common_pattern(successful_contexts),
            "failed_pattern": self._extract_common_pattern(failed_contexts),
            "success_rate": len(successful) / (len(successful) + len(failed)) if (len(successful) + len(failed)) > 0 else 0
        }
        
        return patterns
    
    def _extract_common_pattern(self, contexts: list) -> dict:
        """Extrair padrão comum de contextos"""
        if not contexts:
            return {}
        
        # Encontrar valores comuns
        pattern = {}
        for key in contexts[0].keys() if contexts else []:
            values = [c.get(key) for c in contexts]
            if all(isinstance(v, (int, float)) for v in values):
                pattern[key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return pattern
```

---

## 7. Melhores Práticas de Análise Financeira

### 7.1 Estratégias Recomendadas para Portfolio

#### **Modern Portfolio Theory (Markowitz)**
```python
import numpy as np
from scipy.optimize import minimize

class PortfolioOptimization:
    """Otimização de portfolio segundo Markowitz"""
    
    def __init__(self, returns: np.array, cov_matrix: np.array):
        self.returns = returns  # Retorno esperado de cada ativo
        self.cov = cov_matrix   # Matriz de covariância
    
    def efficient_frontier(self):
        """Calcular fronteira eficiente"""
        
        def portfolio_stats(weights):
            portfolio_return = np.sum(self.returns * weights)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov, weights)))
            return portfolio_return, portfolio_std
        
        # Max Sharpe ratio
        def neg_sharpe(weights):
            p_ret, p_std = portfolio_stats(weights)
            return -(p_ret / p_std)  # Negativo porque minimizamos
        
        n_assets = len(self.returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        opt_return, opt_std = portfolio_stats(optimal_weights)
        
        return {
            "weights": optimal_weights,
            "return": opt_return,
            "volatility": opt_std,
            "sharpe_ratio": opt_return / opt_std
        }
    
    def min_variance_portfolio(self):
        """Portfólio de mínima variância"""
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(self.cov, weights))
        
        n_assets = len(self.returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_variance, init_guess, bounds=bounds, constraints=constraints)
        
        return {"weights": result.x, "variance": result.fun}
```

#### **Risk Parity**
```python
class RiskParityPortfolio:
    """Portfolio em que cada ativo contribui igualmente com risco"""
    
    def __init__(self, cov_matrix: np.array):
        self.cov = cov_matrix
    
    def equal_risk_weights(self) -> np.array:
        """Pesos baseados em risk parity"""
        
        # Volatilidade individual
        volatilities = np.sqrt(np.diag(self.cov))
        
        # Pesos inversamente proporcionais à volatilidade
        inv_vol = 1 / volatilities
        weights = inv_vol / np.sum(inv_vol)
        
        return weights
```

### 7.2 Indicadores Financeiros Críticos

```python
class FinancialIndicators:
    """Indicadores que agentes devem monitorar"""
    
    @staticmethod
    def sharpe_ratio(returns: np.array, risk_free_rate: float = 0.02) -> float:
        """Retorno ajustado ao risco"""
        excess_returns = np.mean(returns) - risk_free_rate
        return excess_returns / np.std(returns)
    
    @staticmethod
    def sortino_ratio(returns: np.array, risk_free_rate: float = 0.02) -> float:
        """Como Sharpe, mas penaliza apenas volatilidade negativa"""
        excess_returns = np.mean(returns) - risk_free_rate
        downside_std = np.std(returns[returns < 0])
        return excess_returns / downside_std if downside_std > 0 else 0
    
    @staticmethod
    def max_drawdown(returns: np.array) -> float:
        """Pior perda acumulada"""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calmar_ratio(returns: np.array, annual_return: float) -> float:
        """Retorno anual / max drawdown (quanto maior, melhor)"""
        md = FinancialIndicators.max_drawdown(returns)
        return annual_return / abs(md) if md != 0 else 0
    
    @staticmethod
    def beta(asset_returns: np.array, market_returns: np.array) -> float:
        """Sensibilidade do ativo ao mercado"""
        cov = np.cov(asset_returns, market_returns)[0, 1]
        market_var = np.var(market_returns)
        return cov / market_var
    
    @staticmethod
    def value_at_risk(returns: np.array, confidence_level: float = 0.95) -> float:
        """VaR - perda máxima esperada com X% confiança"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_var(returns: np.array, confidence_level: float = 0.95) -> float:
        """CVaR - média das perdas piores que VaR"""
        var = FinancialIndicators.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
```

### 7.3 Red Flags Automaticamente Detectadas

```python
class RiskDetection:
    """Detector automático de red flags"""
    
    @staticmethod
    def detect_red_flags(portfolio_data: dict, market_data: dict) -> list:
        """Listar red flags detectadas"""
        
        flags = []
        
        # 1. Concentração excessiva
        largest_position = max(portfolio_data['weights'])
        if largest_position > 0.5:
            flags.append({
                "type": "CONCENTRATION",
                "severity": "HIGH",
                "description": f"Posição maior que 50%: {largest_position:.1%}",
                "action": "Rebalancear portfolio"
            })
        
        # 2. Volatilidade aumentando
        vol_30d = market_data['volatility_30d']
        vol_60d = market_data['volatility_60d']
        if vol_30d > vol_60d * 1.5:
            flags.append({
                "type": "VOLATILITY_SPIKE",
                "severity": "MEDIUM",
                "description": f"Volatilidade aumentou {(vol_30d/vol_60d - 1):.1%}",
                "action": "Revisar posições, considerar hedge"
            })
        
        # 3. Correlação aumentando (risco de não-diversificação)
        correlation_matrix = market_data.get('correlation_matrix', {})
        avg_correlation = np.mean([
            correlation_matrix.get((i, j), 0) 
            for i in range(len(portfolio_data['assets']))
            for j in range(i+1, len(portfolio_data['assets']))
        ])
        
        if avg_correlation > 0.8:
            flags.append({
                "type": "HIGH_CORRELATION",
                "severity": "HIGH",
                "description": f"Ativos altamente correlacionados: {avg_correlation:.2f}",
                "action": "Aumentar diversificação"
            })
        
        # 4. Drift de alocação
        target_weights = portfolio_data['target_weights']
        current_weights = portfolio_data['weights']
        
        for i, (target, current) in enumerate(zip(target_weights, current_weights)):
            drift = abs(current - target)
            if drift > 0.1:  # 10% de drift
                flags.append({
                    "type": "ALLOCATION_DRIFT",
                    "severity": "LOW",
                    "description": f"{portfolio_data['assets'][i]}: " \
                                 f"target {target:.1%}, atual {current:.1%}",
                    "action": "Rebalancear"
                })
        
        return flags
```

---

## 8. Arquitetura Geral Recomendada

### 8.1 Fluxo de Dados Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                   PORTFOLIO ANALYSIS PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │  1. DATA INGESTION LAYER (Tempo Real)               │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  Binance API              Finnhub API                 │
    │  ├─ Preços (1m)          ├─ News                     │
    │  ├─ Volumes              ├─ Fundamentals             │
    │  └─ Trades               └─ Analyst Estimates        │
    │                                                       │
    │  NewsAPI                  Yahoo Finance              │
    │  ├─ News Headlines        └─ Market Data             │
    │  └─ Timestamps                                       │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  2. PREPROCESSING & NORMALIZATION                    │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  - Clean dados (outliers, missing values)            │
    │  - Normalizar timeframes                             │
    │  - Deduplicate notícias                              │
    │  - Converter para formato padrão                     │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  3. VECTOR EMBEDDING & STORAGE (Qdrant)             │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  - Embeddings de notícias (BERT)                     │
    │  - Embeddings de análises técnicas                   │
    │  - Store com metadata (asset, timestamp, tipo)       │
    │  - Indexação para rápido retrieval                   │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  4. AGENT ANALYSIS LAYER (Paralelo)                 │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐   │
    │  │  Portfolio  │  │   Risk       │  │  Market    │   │
    │  │  Analyzer   │  │   Manager    │  │  Analyst   │   │
    │  │  - Health   │  │  - VaR       │  │  - RSI     │   │
    │  │  - Drift    │  │  - Alerts    │  │  - Trends  │   │
    │  │  - Sharpe   │  │  - Stress    │  │  - MA      │   │
    │  └────┬────────┘  └──────┬───────┘  └─────┬──────┘   │
    │       │                  │                │           │
    │  ┌────▼───────────────────▼────────────────▼─────┐    │
    │  │      Sentiment/News Analyzer                 │    │
    │  │  - NLP Analysis                              │    │
    │  │  - Entity extraction                         │    │
    │  │  - Relevance scoring                         │    │
    │  └────┬────────────────────────────────────────┘    │
    │       │                                             │
    └───────┼─────────────────────────────────────────────┘
            │
    ┌───────▼─────────────────────────────────────────────┐
    │  5. ORCHESTRATION & SYNTHESIS                       │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  Maestro Agent:                                      │
    │  - Recebe todos os outputs                          │
    │  - Resolve conflitos entre agentes                  │
    │  - Sintetiza insights em ação                       │
    │  - Scoring consolidado                             │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  6. DECISION ENGINE                                  │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  - Gera recomendações (BUY/SELL/HOLD)               │
    │  - Calcula tamanho posição                          │
    │  - Define stop-loss/take-profit                     │
    │  - Cria micro-análise para histórico                │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  7. EXECUTION & MONITORING                          │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  - Enviar para trade execution (se automático)       │
    │  - Log de decisão                                   │
    │  - Alert ao usuário                                 │
    │  - Real-time monitoring de resultado                │
    │                                                       │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  8. FEEDBACK LOOP & LEARNING                        │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  - Coletar resultado da trade                       │
    │  - Calcular accuracy da análise                     │
    │  - Armazenar micro-análise com feedback             │
    │  - Atualizar performance do agente                  │
    │  - Melhorar próximas análises                       │
    │                                                       │
    └──────────────────────────────────────────────────────┘
```

### 8.2 Implementação de Escalabilidade

```python
class PortfolioSystemArchitecture:
    """Arquitetura escalável para múltiplos ativos"""
    
    def __init__(self, max_assets: int = 5):
        # Layer 1: Data Collection
        self.market_listener = RealtimeMarketListener()
        self.data_integrator = FinancialDataIntegrator()
        
        # Layer 2: Preprocessing
        self.preprocessor = DataPreprocessor()
        
        # Layer 3: Vector DB
        self.vector_db = PortfolioVectorDB()
        
        # Layer 4: Agents
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.risk_manager = RiskManager()
        self.market_analyst = MarketAnalyst()
        self.sentiment_agent = NewsAgent()
        
        # Layer 5: Orchestration
        self.maestro = PortfolioMaestro()
        
        # Layer 6: Memory
        self.memory = PersistentMemory()
        self.learning_engine = ContinuousLearning(self.maestro, self.memory)
        
        # Layer 7: RAG
        self.rag = PortfolioRAG(self.vector_db, llm_client=None)
        
        self.max_assets = max_assets
        self.active_assets = []
    
    async def initialize_system(self, assets: List[str]):
        """Inicializar sistema para novos ativos"""
        
        print(f"Inicializando sistema para: {assets}")
        
        # 1. Validar número de ativos
        if len(assets) > self.max_assets:
            raise ValueError(f"Máximo {self.max_assets} ativos permitidos")
        
        self.active_assets = assets
        
        # 2. Iniciar listeners de mercado
        await self.market_listener.start_listening(assets)
        
        # 3. Carregar dados históricos
        for asset in assets:
            historical_data = await self.data_integrator.get_technical_data(asset)
            await self.preprocessor.process_historical(asset, historical_data)
    
    async def run_analysis_cycle(self):
        """Rodar ciclo de análise continuamente"""
        
        while True:
            try:
                # 1. Coletar dados
                current_data = {}
                for asset in self.active_assets:
                    current_data[asset] = {
                        "market": await self.market_listener.get_latest(asset),
                        "news": await self.data_integrator.fetch_news(asset)
                    }
                
                # 2. Executar análises paralelas
                analyses = await self.maestro.daily_analysis_cycle()
                
                # 3. Gerar recomendações
                recommendations = await self.maestro.decision_engine.recommend(analyses)
                
                # 4. Armazenar para aprendizado
                await self.memory.store_cycle_results({
                    "timestamp": datetime.now(),
                    "data": current_data,
                    "analyses": analyses,
                    "recommendations": recommendations
                })
                
                # 5. Publicar resultados
                await self._publish_results(recommendations)
                
                # Próximo ciclo
                await asyncio.sleep(300)  # 5 minutos
            
            except Exception as e:
                print(f"Error in analysis cycle: {e}")
                await asyncio.sleep(60)
    
    async def run_feedback_cycle(self):
        """Rodar ciclo de feedback/aprendizado diariamente"""
        
        while True:
            # A cada 24h
            await asyncio.sleep(86400)
            
            # Revisar decisões do dia anterior
            await self.learning_engine.daily_review_cycle()
```

### 8.3 Performance e Overhead

Para 1-5 ativos, estimativas de overhead:

| **Componente** | **Latência** | **Memória** | **CPU** |
|---|---|---|---|
| Coleta de dados | 50-100ms | ~50MB | 5-10% |
| Processamento | 100-200ms | ~100MB | 15-20% |
| Vector DB Query | 5-20ms | ~200MB base | 5-10% |
| Análises (paralelo) | 500-2000ms | ~300MB | 30-50% |
| RAG contexto | 100-300ms | ~100MB | 10-15% |
| **TOTAL** | **~2-3 segundos** | **~750MB** | **~70% (picos)** |

✅ **Conclusão**: Sistema é totalmente viável para seu caso. Rode em máquina com 2GB RAM, 2+ cores.

---

## 9. Step 13: Ecosystem de Agentes IA Multi-Especializados (MVP + Phases)

> **Data de Decisão**: Fevereiro 2, 2026  
> **Decisão Principal**: Implementar 7 agentes MVP + 2 novos agentes especializados (MetaLearningOptimizer + CryptoExpertAgent)  
> **Algoritmo de Otimização**: Bayesian Optimization (Optuna) + Thompson Sampling (Multi-Armed Bandit)

### 9.1 Decisões Técnicas Fundamentais

#### **Problema Identificado**
- ❌ Nenhum mecanismo para medir e melhorar a eficiência do reinforcement learning ao longo do tempo
- ❌ Sem calibração automática de parâmetros dos agentes
- ❌ Sem especialista dedicado a análise completa do ecossistema Binance/Crypto

#### **Análise de Abordagens: GA vs Bayesian vs MAB**

| **Critério** | **Genetic Algorithm (GA)** | **Bayesian Optimization** | **Thompson Sampling (MAB)** |
|---|---|---|---|
| **Iterações Necessárias** | 50k-100k (muito lento) | 50-100 (ótimo) | 20-30 por peso |
| **Convergência** | Lenta, não garantida | Rápida, probabilística | Rápida para pesos |
| **Complexidade** | Alta (crossover, mutation) | Média (modelar surface) | Baixa (simples probabilidade) |
| **Overhead Computacional** | 30-50% de CPU | 10-15% de CPU | <5% de CPU |
| **Adaptação a Mudanças** | Lenta (espera próxima geração) | Rápida (contínuo) | Muito rápida (real-time) |
| **Ideal Para** | Problemas muito complexos | Hyperparameter tuning | Dynamic weight adjustment |
| **Nossa Decisão** | ❌ Não (overkill) | ✅ Sim (Phase 3) | ✅ Sim (Phase 2) |

**Conclusão**: Usar **Bayesian Optimization** (Phase 3) para ajuste de hiperparâmetros + **Thompson Sampling** (Phase 2) para pesos de ensemble em tempo real.

### 9.2 Arquitetura de 7 Agentes MVP + 2 Novos

#### **Tabela Consolidada: 21 Agentes Planejados**

| # | **Nome do Agente** | **Fase** | **Status** | **Responsabilidade Principal** | **Prioridade** | **Integração** |
|---|---|---|---|---|---|---|
| **MVP (7 agentes)** |
| 1 | 🎯 **Meta Orchestrator** | MVP | ✅ Especificado | Coordena todos; sintetiza; toma decisões finais | 🔴 Crítica | Hub central |
| 2 | 📊 **Portfolio Analyzer** | MVP | ✅ Especificado | Análise de composição, diversificação, drift | 🔴 Crítica | Pub-Sub |
| 3 | ⚠️ **Risk Manager** | MVP | ✅ Especificado | VaR, drawdown, stress test, alerts | 🔴 Crítica | Pub-Sub |
| 4 | 📈 **Market Analyst** | MVP | ✅ Especificado | Análise técnica, padrões, correlações | 🔴 Crítica | Pub-Sub |
| 5 | 📰 **News & Sentiment** | MVP | ✅ Especificado | NLP, sentimento, anomalias | 🟡 Alta | Pub-Sub |
| 6 | 🧠 **MetaLearningOptimizer** | MVP (Fase 2 primeiro) | 🆕 **NOVO** | Mede efficiency RL; calibra parâmetros; Thompson Sampling | 🔴 Crítica | Request-Response |
| 7 | 🪙 **CryptoExpertAgent** | MVP | 🆕 **NOVO** | Análise completa Binance; oportunidades; on-chain | 🔴 Crítica | Pub-Sub |
| **Phase 2 (4 agentes)** |
| 8 | ⚖️ **Rebalancing Agent** | Phase 2 | 📋 Planejado | Rebalanceamento automático periódico | 🟡 Alta | Pub-Sub |
| 9 | 💰 **Tax Optimizer** | Phase 2 | 📋 Planejado | Otimização de fiscalidade | 🟡 Alta | Request-Response |
| 10 | 📊 **Performance Attribution** | Phase 2 | 📋 Planejado | Análise de quem contribuiu com ganhos | 🟢 Média | Request-Response |
| 11 | 🛑 **Stop Loss Manager** | Phase 2 | 📋 Planejado | Gerência dinâmica de stop-loss | 🟡 Alta | Pub-Sub |
| **Phase 3 (10+ agentes)** |
| 12 | 🌐 **Macro Economist** | Phase 3 | 📋 Planejado | Fed decisions, inflation, macroeconomics | 🟢 Média | Pub-Sub |
| 13 | 🤝 **Correlation Analyzer** | Phase 3 | 📋 Planejado | Cross-asset correlations, hedge strategies | 🟢 Média | Request-Response |
| 14 | 🔄 **Arbitrage Scout** | Phase 3 | 📋 Planejado | Oportunidades de arbitragem | 🟢 Média | Request-Response |
| 15 | 📡 **Social Sentiment Bot** | Phase 3 | 📋 Planejado | Twitter, Reddit, Discord analysis | 🟢 Média | Pub-Sub |
| 16-21 | **6+ Future Agents** | Phase 3+ | 📋 Planejado | DeFi, Options, Derivatives, etc. | 🟢 Média | TBD |

### 9.3 Detalhamento: MetaLearningOptimizer Agent (NOVO)

#### **Objetivo Geral**
Medir continuamente a eficiência do reinforcement learning do sistema, detectar degradação de performance e calibrar automaticamente:
- Hiperparâmetros de cada agente (temperature, confidence thresholds, volatility multipliers)
- Pesos do ensemble (quem contribui mais com accuracy)
- Estratégias quando agentes discordam

#### **Phase 2: Heuristic + Thompson Sampling**

**Período**: Semanas 1-3 de implementação (mais rápido, conservative)

**Algoritmo**:
1. **Calculate Agent Metrics** (diariamente):
   ```
   - accuracy = (correct_predictions / total_predictions)
   - precision = (true_positives / (true_positives + false_positives))
   - recall = (true_positives / (true_positives + false_negatives))
   - false_positive_rate = (false_positives / (false_positives + true_negatives))
   - sharpe_ratio = (avg_return / std_return)
   ```

2. **Detect Performance Trends** (janelas de 7/14/30 dias):
   ```
   - trend = linear_regression(accuracy_last_14_days)
   - if trend < 0: agent está piorando
   - velocity = (current_accuracy - prev_week_accuracy)
   ```

3. **Thompson Sampling Bandit** (cada agente = 1 arm):
   ```
   - Cada agente tem: Alpha (sucessos) e Beta (fracassos)
   - Gerar sample: theta ~ Beta(alpha + 1, beta + 1)
   - Selecionar agente com maior theta (exploitation + exploration)
   - Atualizar Alpha/Beta baseado em resultado real
   ```

4. **Heuristic Rules** (guardas conservadores):
   ```
   - if accuracy < 60%: reduce weight by 10%
   - if accuracy stable 2+ weeks: reduce weight 5% (forced exploration)
   - if accuracy > 75% AND trend positive: increase weight 10%
   - if false_positive_rate > 30%: penalize heavily
   ```

5. **Conflict Detection** (quando 2+ agentes discordam):
   ```
   - Log: {timestamp, agent1, signal1, agent2, signal2, actual_outcome}
   - Track: quem estava certo? (histórico)
   - Usar para calibrar confiança futura
   ```

**Output Diário**:
```json
{
  "timestamp": "2026-02-02T18:00Z",
  "agent_metrics": {
    "market_analyst": {
      "accuracy": 0.72,
      "precision": 0.68,
      "recall": 0.75,
      "false_positive_rate": 0.15,
      "sharpe_ratio": 1.8,
      "current_weight": 0.25,
      "trend": "improving (+2% last 7 days)"
    }
  },
  "recommendations": [
    {
      "agent": "portfolio_analyzer",
      "recommendation": "increase_weight",
      "from_weight": 0.15,
      "to_weight": 0.18,
      "reason": "accuracy improved consistently"
    }
  ],
  "conflicts_detected": [
    {
      "agent1": "market_analyst",
      "agent2": "news_sentiment",
      "disagreement": "bullish vs bearish",
      "outcome": "market_analyst correct",
      "frequency_this_week": 3
    }
  ]
}
```

**Database Tables** (novos):
```sql
CREATE TABLE agent_performance_history (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  accuracy FLOAT,
  precision FLOAT,
  recall FLOAT,
  false_positive_rate FLOAT,
  sharpe_ratio FLOAT,
  weight FLOAT,
  FOREIGN KEY(agent_name) REFERENCES agents(name)
);

CREATE TABLE ensemble_conflicts (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent1_name TEXT,
  agent2_name TEXT,
  signal1 TEXT,
  signal2 TEXT,
  actual_outcome TEXT,
  winner TEXT,
  FOREIGN KEY(agent1_name) REFERENCES agents(name),
  FOREIGN KEY(agent2_name) REFERENCES agents(name)
);

CREATE TABLE parameter_adjustments (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  parameter_name TEXT,
  old_value FLOAT,
  new_value FLOAT,
  reason TEXT,
  performance_result TEXT,
  FOREIGN KEY(agent_name) REFERENCES agents(name)
);

CREATE TABLE agent_weights (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  weight FLOAT,
  is_active BOOLEAN,
  FOREIGN KEY(agent_name) REFERENCES agents(name)
);
```

**Tool Functions**:
```python
# Implementar em app/agents/meta_learning_optimizer_agent.py
async def calculate_agent_metrics(agent_name: str, lookback_days: int = 30) -> dict
async def detect_performance_trends(agent_name: str) -> dict
async def generate_heuristic_recommendations() -> list
async def apply_thompson_sampling() -> dict
async def detect_agent_conflicts() -> list
async def get_performance_report() -> dict
async def adjust_agent_weight(agent_name: str, new_weight: float) -> bool
async def revert_failed_adjustments() -> bool
```

#### **Phase 3: Bayesian Optimization (Futuro)**

**Período**: Semanas 4-6 de implementação (mais complexo, adaptive)

**Usar Optuna para tunar**:
- `temperature_llm` (0.4 a 1.0) - controla criatividade
- `confidence_threshold` (0.3 a 0.9) - quando recomendar vs ficar quieto
- `volatility_multiplier` (0.5 a 2.0) - quanto maior o risco, mais conservador

**Validação**:
- Treinar em 70% do histórico
- Validar em 20% holdout (últimos 7 dias nunca vistos)
- Auto-revert se performance < versão anterior no holdout set

**Job Diário**:
```
15:55h (antes do market close)
  → Coletar últimas 100 análises
  → Executar 50 iterações de Bayesian Opt
  → Testar em holdout (7 dias)
  → Aplicar ou revert
  → Log de todas as mudanças
```

### 9.4 Detalhamento: CryptoExpertAgent (NOVO)

#### **Objetivo Geral**
Análise especializada e contínua de **todas as facetas de investimento em crypto** no ecossistema Binance:
- Spot trading (análise técnica + fundamentals)
- Staking/Savings (yields, APY comparisons)
- Launchpool (oportunidades, padrões históricos)
- Futures (liquidation risk, funding rates)
- Farming/LP (impermanent loss, yields)
- Margin trading (collateral monitoring)

Integrado com **dados multi-source**:
- Binance Announcements Feed
- On-chain data (whale movements, exchange flows)
- Social sentiment (Twitter, Reddit, Discord)
- Macro crypto news (regulations, Fed, Bitcoin miners)

#### **Responsabilidades & Tool Functions**

```python
# Implementar em app/agents/crypto_expert_agent.py

async def fetch_binance_opportunities() -> dict:
    """
    Agregar TODAS as oportunidades Binance em tempo real:
    - Spot: ativos listados, volumes, spreads
    - Futures: liquidation prices, open interest
    - Staking: APY atual vs histórico
    - Launchpool: próximos eventos, histórico ROI
    - Farming: LP pairs, impermanent loss risk
    - Margin: collateral ratios, liquidation prices
    
    Output: {
      "spot_opportunities": [...],
      "futures_opportunities": [...],
      "staking_opportunities": [...],
      "launchpool_upcoming": [...],
      "farming_opportunities": [...]
    }
    """

async def analyze_new_token_listing(token: str) -> dict:
    """
    Quando Binance lista novo token:
    - Análise rápida de fundamentals (website, whitepaper, team)
    - Histórico de listing events (qual foi padrão de retorno?)
    - Pump-dump risk assessment
    - Volatilidade esperada baseado em padrões
    
    Output: {
      "token": "XYZ",
      "fundamentals_score": 0.7,
      "historical_roi": "avg 40% em 1 semana",
      "pump_dump_risk": "moderate",
      "recommendation": "ACCUMULATE" | "MONITOR" | "SKIP",
      "confidence": 0.65
    }
    """

async def calculate_staking_yield(asset: str) -> dict:
    """
    Comparar staking yields:
    - Binance Staking APY
    - Comparação vs histórico médio
    - Comparação vs outros exchanges (Kraken, Lido, etc)
    - Lockup periods e riscos
    
    Output: {
      "asset": "ETH",
      "binance_apy": 0.035,
      "historical_avg_apy": 0.032,
      "recommendation": "ABOVE_AVERAGE - window opportunity",
      "lockup_periods": ["15d", "30d", "90d", "flexible"],
      "risks": ["validator_risk", "smart_contract_risk"]
    }
    """

async def detect_launchpool_pump_dump(token: str) -> dict:
    """
    Análise de risco pump-dump em Launchpool:
    - Padrões históricos de projetos similares
    - Community size vs hype (red flag se desproporcionais)
    - Whitepaper quality vs community
    - Entrada + saída vs volatilidade
    
    Output: {
      "token": "ABC",
      "historical_pattern": "66% pump in first hour, then dump",
      "community_score": 6.2,
      "project_score": 7.8,
      "mismatch_risk": "HIGH - small project, huge hype",
      "recommendation": "SKIP" | "LIGHT_ENTRY" | "ACCUMULATE",
      "confidence": 0.72
    }
    """

async def fetch_crypto_news() -> list:
    """
    Agregar notícias de múltiplas fontes:
    - Binance Announcements (exclusivo)
    - Macro crypto news (regulations, Fed, Bitcoin miners)
    - NewsAPI com filtro crypto
    
    Output: [
      {
        "title": "SEC approves Bitcoin ETF",
        "source": "macro_crypto",
        "sentiment": "bullish",
        "relevance_to_portfolio": 0.85,
        "timestamp": "2026-02-02T10:30Z",
        "impact_on_assets": ["BTC", "ETH"]
      }
    ]
    """

async def analyze_on_chain_data(asset: str) -> dict:
    """
    Análise on-chain usando CoinGecko/on-chain APIs:
    - Whale movements (large transactions)
    - Exchange inflows/outflows (accumulation vs distribution)
    - Dormant addresses activating (signal de interesse)
    - Miner movements (BTC miners vendendo vs acumulando)
    
    Output: {
      "asset": "BTC",
      "whale_transactions_24h": 15,
      "large_tx_direction": "to_exchanges (distribution)", 
      "exchange_flow": "net_outflow 500 BTC (accumulation)",
      "dormant_activation": "3 addresses 10y+ activate",
      "miner_movement": "net_accumulation",
      "aggregate_signal": "ACCUMULATION",
      "confidence": 0.68
    }
    """

async def social_sentiment_analysis(asset: str) -> dict:
    """
    Análise de sentimento de comunidade:
    - Twitter/X trending topics
    - Reddit discussions + upvotes
    - Discord community sentiment
    - Telegram alerts volume
    
    Output: {
      "asset": "SOL",
      "twitter_sentiment": 0.72,
      "twitter_volume": 45000,
      "reddit_upvotes": 2300,
      "discord_online": 12500,
      "telegram_mentions": 890,
      "aggregate_sentiment": "POSITIVE",
      "trend_direction": "increasing",
      "authenticity_score": 0.65
    }
    """

async def get_futures_liquidation_risk(symbol: str) -> dict:
    """
    Análise de risco em Futures:
    - Liquidation price range
    - Open interest distribution
    - Funding rate (positivo = bullish, negativo = bearish)
    - Extreme funding = reversal likely
    
    Output: {
      "symbol": "BTC/USDT",
      "current_price": 42500,
      "liquidation_price_long": 38000,
      "liquidation_price_short": 47800,
      "open_interest": 2.3e9,
      "funding_rate": 0.0035,
      "funding_direction": "bullish",
      "extreme_level": false,
      "recommendation": "CAUTION - high liquidation risk for longs"
    }
    """

async def compare_opportunities() -> list:
    """
    Ranking consolidado de oportunidades:
    - Spot vs Futures vs Staking vs Farming
    - Risk-adjusted returns
    - Volatilidade esperada
    
    Output: [
      {
        "opportunity": "SOL Spot long",
        "expected_return": 0.15,
        "risk_level": "medium",
        "timeframe": "1-2 weeks",
        "confidence": 0.68,
        "factors": ["bullish on-chain", "positive sentiment"],
        "risks": ["macro uncertainty", "funding rate risk"],
        "rank": 1
      }
    ]
    """
```

#### **Multi-Source Data Integration**

```python
class CryptoDataCollector:
    """Integração com múltiplas fontes de dados crypto"""
    
    async def get_binance_announcements(self) -> list:
        # API: https://www.binance.com/en/support/announcement/...
        # Atualizar a cada 1h
        pass
    
    async def get_on_chain_metrics(self, asset: str) -> dict:
        # APIs: CoinGecko Free Tier, Nansen (paid), Glassnode (paid)
        # Whale tracking, exchange flows, etc
        pass
    
    async def get_social_sentiment(self, asset: str) -> dict:
        # Twitter API v2 (gratuito mas limitado)
        # Reddit API (gratuito)
        # Discord scraping (respeitar ToS)
        pass
    
    async def get_macro_crypto_news(self) -> list:
        # NewsAPI com filtro "cryptocurrency"
        # Finnhub crypto endpoints
        # Custom feeds de Regulation changes
        pass
```

#### **Memory & Pattern Recognition**

```python
class CryptoExpertMemory:
    """Memória persistente de padrões crypto"""
    
    async def store_launchpool_history(self, token: str, analysis: dict):
        """Guardar histórico: entrada → saída → ROI"""
        # entry_price, exit_price, predicted_roi, actual_roi
        # Usar para prever padrões futuros
        pass
    
    async def store_staking_yields(self, asset: str, apy: float, timestamp: datetime):
        """Histórico de yields para detecção de anomalias"""
        pass
    
    async def store_binance_listing_patterns(self, token: str, outcome: dict):
        """Padrões: quais tokens crescem, quais pump-and-dump"""
        pass
```

#### **Sample Output - Recomendações do CryptoExpertAgent**

```json
{
  "timestamp": "2026-02-02T14:30Z",
  "recommendations": [
    {
      "asset": "BTC",
      "signal": "LONG_SPOT",
      "reasoning": "100k BTC leaving exchanges (accumulation signal) + dormant whales activating",
      "confidence": 0.70,
      "timeframe": "2-4 weeks",
      "entry_strategy": "DCA over 3 days at support"
    },
    {
      "asset": "ETH",
      "signal": "STAKING_OPPORTUNITY",
      "reasoning": "Current APY 3.5% vs historical avg 3.2% = temporary window",
      "confidence": 0.68,
      "duration": "lock for 90 days to maximize",
      "risk": "validator_risk low"
    },
    {
      "asset": "SOL_LAUNCHPOOL",
      "signal": "MONITOR",
      "reasoning": "Launchpool starts in 2 days, Token fundamentals: solid but small community (risk mismatch)",
      "confidence": 0.40,
      "recommendation": "Skip or light entry only"
    },
    {
      "asset": "BTC_FUTURES",
      "signal": "CAUTION",
      "reasoning": "Extreme funding rate (0.12% daily) favors shorts but liquidation risk for longs = reversal likely",
      "confidence": 0.65,
      "entry_level": "if enters, only with tight stop-loss"
    }
  ],
  "macro_signals": [
    "Bitcoin miners accumulating (bullish)",
    "SEC likely approving Ethereum ETF in March (bullish for ETH)",
    "Fed decision next week (macro uncertainty)"
  ]
}
```

### 9.5 Integração Entre MetaLearningOptimizer + CryptoExpertAgent

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATION FLOW                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  CryptoExpertAgent                                      │
│  ├─ Fetch opportunities (Binance + on-chain + news)    │
│  ├─ Publish: "crypto_opportunities_updated"            │
│  └─ Output: [BTC long 70%, ETH stake 68%, ...]         │
│                                                           │
│  MetaLearningOptimizer                                 │
│  ├─ Subscribe to agent metrics                         │
│  ├─ Calculate: accuracy, precision, recall per agent   │
│  ├─ Thompson Sampling: ajustar weights                 │
│  ├─ Detect conflicts: quando agentes discordam         │
│  └─ Output: new weights {BTC: 0.25, ETH: 0.15, ...}   │
│                                                           │
│  MetaOrchestrator                                      │
│  ├─ Receive recommendations from all agents            │
│  ├─ Apply weights from MetaLearningOptimizer           │
│  ├─ Synthesize final signal (weighted ensemble)        │
│  └─ Output: "BTC LONG 68% confidence (ensemble)"       │
│                                                           │
│  User Dashboard                                         │
│  ├─ "Calibração IA" tab showing:                       │
│  │  - Per-agent performance + trends                   │
│  │  - Conflicts & resolutions                          │
│  │  - Suggested parameter adjustments                  │
│  └─ Final recommendations with confidence              │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 9.6 Configuration (config.yaml)

```yaml
# config.yaml
agents:
  meta_orchestrator:
    llm_provider: "openai"  # ou deepseek, qwen, anthropic
    model: "gpt-4o"
    temperature: 0.7
  
  meta_learning_optimizer:
    phase: "heuristic"  # heuristic (Phase 2) → bayesian (Phase 3)
    strategy: "thompson_sampling"
    auto_apply: false  # require user approval for changes
    min_data_points: 100  # minimum analyses before optimizing
    revert_if_worse: true
    hold_out_period: 7  # days for validation
    heuristic_rules:
      accuracy_threshold_down: 0.60
      weight_penalty_if_poor: 0.10
      weight_boost_if_good: 0.10
      trend_lookback_days: 14
  
  crypto_expert_agent:
    enabled: true
    data_sources:
      - "binance_announcements"
      - "on_chain_data"
      - "social_sentiment"
      - "macro_crypto_news"
    update_frequency: "1h"
    confidence_min: 0.40  # minimum confidence to recommend
    
  portfolio_analyzer:
    enabled: true
    max_assets: 5
    rebalance_threshold: 0.05  # 5% drift triggers rebalance

database:
  sqlite_path: "portfolio_memory.db"
  vector_db: "qdrant"
  qdrant_url: "localhost:6333"

api_keys:
  binance: "${BINANCE_API_KEY}"
  finnhub: "${FINNHUB_KEY}"
  newsapi: "${NEWSAPI_KEY}"
  coingecko: "${COINGECKO_KEY}"
  openai: "${OPENAI_API_KEY}"
```

### 9.7 Frontend Requirements

**New Tab: "Calibração IA"**

```
├─ Agent Status Cards (per agent):
│  ├─ Accuracy trend chart (7/14/30 days)
│  ├─ Current weight (Thompson Sampling value)
│  ├─ W/L record (wins vs losses)
│  └─ Last updated timestamp
│
├─ Conflict Analysis Table:
│  ├─ When agents disagreed
│  ├─ Which was correct
│  ├─ Frequency tracking
│  └─ Impact on portfolio
│
├─ Calibration Suggestions:
│  ├─ "Increase XXX Agent weight" (with confidence)
│  ├─ "Reduce YYY temperature parameter"
│  ├─ Apply button (with user approval)
│  └─ Revert button if went wrong
│
└─ Historical Adjustments Log:
   ├─ What changed
   ├─ When
   ├─ Why
   └─ Result (positive/negative/pending)
```

### 9.8 Implementation Roadmap

| **Semana** | **Tarefas** | **Ownership** | **Deliverables** |
|---|---|---|---|
| **1** | Database schema (agent_performance_history, conflicts, adjustments) | Backend | ✅ 4 new tables |
| **1-2** | MetaLearningOptimizer Phase 2 (Thompson Sampling + heuristics) | Backend | ✅ Agent + daily job |
| **2-3** | CryptoExpertAgent core (fetch opportunities, basic analysis) | Backend | ✅ 9 tool functions |
| **3** | Integrate data sources (Binance API, on-chain, social) | Data | ✅ Multi-source collector |
| **3-4** | Frontend: "Calibração IA" dashboard tab | Frontend | ✅ Charts + controls |
| **4-5** | MetaLearningOptimizer Phase 3 (Bayesian Optimization with Optuna) | ML | ✅ Hyperparameter tuning |
| **5** | Testing + validation on backtested data | QA | ✅ Test suite |
| **6** | Deploy to production + monitor | DevOps | ✅ Live system |

---

## 📚 URLs de Recursos Recomendados

### Documentação Oficial

- **Agno** (Framework multi-agent):https://github.com/aqora-io/agno (⚠️ Projeto novo, verifique versão atual)
- **LangChain** (Alternativa madura): https://python.langchain.com/docs/
- **LangGraph** (Orquestração): https://docs.langchain.com/oss/python/langgraph/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Milvus Docs**: https://milvus.io/docs

### APIs de Dados Financeiros

- **Finnhub**: https://finnhub.io/docs/api
- **NewsAPI**: https://newsapi.org/docs
- **Alpha Vantage**: https://www.alphavantage.co/documentation/
- **CoinGecko**: https://www.coingecko.com/en/api
- **Binance API**: https://binance-docs.github.io/apidocs/

### Vector Embeddings & NLP

- **Sentence Transformers**: https://www.sbert.net/
- **Transformers (Hugging Face)**: https://huggingface.co/docs/transformers/
- **BERT Models**: https://huggingface.co/dslim/ (NER), https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english (Sentiment)

### Bibliotecas Python Úteis

```bash
pip install agno langchain langgraph qdrant-client sentence-transformers transformers
pip install pandas numpy scipy scikit-learn
pip install aiohttp websockets
pip install finnhub-python newsapi
python-binance pycoingecko
```

### Artigos & Papers

- **Efficient Frontier**: https://en.wikipedia.org/wiki/Efficient_frontier
- **Multi-Agent Systems**: https://arxiv.org/abs/2406.14462 (Survey 2024)
- **RAG Architecture**: https://arxiv.org/abs/2005.11401
- **Vector Search Performance**: https://qdrant.tech/articles/vector-database-benchmark/

---

## 🎯 Próximos Passos Recomendados

### Fase 1: Prototipagem (1-2 semanas)
1. Configurar Qdrant local
2. Implementar primeira análise técnica simples
3. Integrar NewsAPI para notícias
4. Criar estrutura básica de micro-análises

### Fase 2: Multi-Agent (2-3 semanas)
1. Implementar 3 agentes principais (Portfolio, Risk, Market)
2. Criar camada de orquestração (Maestro)
3. Integrar com Finnhub
4. Sistema de feedback loops

### Fase 3: Produção (2-4 semanas)
1. Deploy em servidor
2. Histórico completo de micro-análises
3. Aprendizado contínuo ativo
4. Dashboard de monitoring

---

**Documento finalizado em 01/02/2026**  
**Próxima revisão recomendada**: Quando houver novo modelo Agno LTS ou mudanças significativas em APIs
