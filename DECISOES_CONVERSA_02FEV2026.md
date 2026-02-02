# Decisões Técnicas - Conversa 02/02/2026

## 🎯 Resumo Executivo

**Data**: 2 de Fevereiro de 2026  
**Participantes**: Desenvolvedor + GitHub Copilot (Claude Haiku 4.5)  
**Status**: ✅ TODAS AS DECISÕES REGISTRADAS NO ARQUITETURA_AGENTES_IA_PORTFOLIO.md

---

## 1️⃣ Decisão Principal: MetaLearningOptimizer Agent

### Problema Identificado
- ❌ Nenhum mecanismo para medir eficiência do reinforcement learning ao longo do tempo
- ❌ Sem calibração automática de parâmetros dos agentes
- ❌ Sem forma de detectar quando um agente está piorando

### Solução Aprovada
**Criar um agente especializado**: `MetaLearningOptimizer`

**Responsabilidades**:
- ✅ Medir performance de cada agente (accuracy, precision, recall, Sharpe ratio)
- ✅ Detectar tendências de degradação em janelas de 7/14/30 dias
- ✅ Calibrar automaticamente pesos e hiperparâmetros
- ✅ Detectar conflitos quando 2+ agentes discordam
- ✅ Manter audit trail completo (parameter_adjustments table)

**Integração no MVP**: ✅ Agent #6 (7 agentes totais)

---

## 2️⃣ Decisão Técnica: Algoritmo de Otimização

### Questão: GA vs Bayesian vs MAB?
*"é possível, viavel, vale a pena, usar um algoritmo genetico?"*

### Análise Comparativa

| Critério | Genetic Algorithm | Bayesian Optimization | Thompson Sampling |
|---|---|---|---|
| **Iterações** | 50k-100k ❌ | 50-100 ✅ | 20-30 ✅ |
| **Convergência** | Lenta | Rápida | Muito rápida |
| **CPU Overhead** | 30-50% | 10-15% | <5% |
| **Adaptação** | Lenta | Rápida | Muito rápida |
| **Nosso Uso** | ❌ Não (overkill) | ✅ Phase 3 | ✅ Phase 2 |

### Decisão Final: ✅ APROVADO
- **Phase 2** (semanas 1-3): Thompson Sampling + Heuristics
- **Phase 3** (semanas 4-6): Bayesian Optimization (Optuna) para tuning fino

### Por quê não GA?
- GA converge em 50k+ iterações vs Bayesian em 50-100
- GA não se adapta bem a mudanças de regime de mercado
- Overhead computacional muito alto (30-50% CPU)
- Para 10-15 hiperparâmetros, Bayesian é 80% mais eficiente

---

## 3️⃣ Decisão: CryptoExpertAgent

### Problema Identificado
- ❌ Nenhum agente especializado em análise completa do ecossistema Binance/Crypto
- ❌ Falta expertise em Launchpool, Futures, Farming, Staking
- ❌ Sem integração de dados on-chain e social sentiment

### Solução Aprovada
**Criar agente especializado**: `CryptoExpertAgent`

**Cobertura Completa**:
- ✅ Spot trading (análise técnica + fundamentals)
- ✅ Staking/Savings (APY comparison vs histórico)
- ✅ Launchpool (novo token analysis + pump-dump detection)
- ✅ Futures (liquidation risk, funding rates)
- ✅ Farming/LP (impermanent loss, yields)
- ✅ Margin trading (collateral monitoring)

**Data Sources Multi-Source**:
- ✅ Binance Announcements Feed
- ✅ On-chain data (whale movements, exchange flows, CoinGecko API)
- ✅ Social sentiment (Twitter, Reddit, Discord)
- ✅ Macro crypto news (Fed, regulations, Bitcoin miners)

**9 Tool Functions**:
1. `fetch_binance_opportunities()` - agregador de oportunidades
2. `analyze_new_token_listing()` - análise de novos tokens
3. `calculate_staking_yield()` - comparação de yields
4. `detect_launchpool_pump_dump()` - detecção de riscos
5. `fetch_crypto_news()` - agregação de notícias
6. `analyze_on_chain_data()` - métricas on-chain
7. `social_sentiment_analysis()` - análise de comunidade
8. `get_futures_liquidation_risk()` - risco em derivados
9. `compare_opportunities()` - ranking consolidado

**Integração no MVP**: ✅ Agent #7 (7 agentes totais)

---

## 4️⃣ Estrutura Final: 7 Agentes MVP

### Tabela Consolidada

| # | Nome | Novo | Fase | Responsabilidade |
|---|---|---|---|---|
| 1 | 🎯 Meta Orchestrator | - | MVP | Coordena todos os agentes |
| 2 | 📊 Portfolio Analyzer | - | MVP | Análise de composição e diversificação |
| 3 | ⚠️ Risk Manager | - | MVP | VaR, drawdown, stress tests |
| 4 | 📈 Market Analyst | - | MVP | Análise técnica e padrões |
| 5 | 📰 News & Sentiment | - | MVP | NLP, sentimento, anomalias |
| **6** | **🧠 MetaLearningOptimizer** | **✅ NOVO** | **MVP** | **Mede efficiency e calibra parâmetros** |
| **7** | **🪙 CryptoExpertAgent** | **✅ NOVO** | **MVP** | **Análise completa Binance + on-chain** |

**Plus**: 4 agentes Phase 2 + 10+ agentes Phase 3 (mapeados no Step 13)

---

## 5️⃣ Database Tables (Novos)

```sql
-- Armazenar performance de cada agente
CREATE TABLE agent_performance_history (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  accuracy FLOAT,
  precision FLOAT,
  recall FLOAT,
  false_positive_rate FLOAT,
  sharpe_ratio FLOAT,
  weight FLOAT
);

-- Rastrear quando agentes discordam
CREATE TABLE ensemble_conflicts (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent1_name TEXT,
  agent2_name TEXT,
  signal1 TEXT,
  signal2 TEXT,
  actual_outcome TEXT,
  winner TEXT  -- quem estava certo?
);

-- Audit trail de ajustes
CREATE TABLE parameter_adjustments (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  parameter_name TEXT,
  old_value FLOAT,
  new_value FLOAT,
  reason TEXT,
  performance_result TEXT
);

-- Evolução de pesos do ensemble
CREATE TABLE agent_weights (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  agent_name TEXT,
  weight FLOAT,
  is_active BOOLEAN
);
```

---

## 6️⃣ Configuration (config.yaml - Seção Nova)

```yaml
meta_learning:
  phase: "heuristic"  # heuristic (Phase 2) → bayesian (Phase 3)
  strategy: "thompson_sampling"
  auto_apply: false  # require user approval
  min_data_points: 100  # minimum antes de otimizar
  revert_if_worse: true  # auto-revert se performance piora
  hold_out_period: 7  # dias para validação
```

---

## 7️⃣ Frontend Requirement

### Nova Aba: "Calibração IA"

```
├─ Agent Status Cards
│  ├─ Accuracy trend (7/14/30 days)
│  ├─ Current weight (Thompson Sampling)
│  ├─ W/L record
│  └─ Last updated
│
├─ Conflict Analysis Table
│  ├─ Quando agentes discordaram
│  ├─ Qual estava certo
│  └─ Frequência
│
├─ Calibration Suggestions
│  ├─ "Increase XXX weight"
│  ├─ "Reduce YYY temperature"
│  └─ Apply/Revert buttons
│
└─ Adjustments Log
   └─ Histórico completo de mudanças
```

---

## 8️⃣ Implementation Roadmap

| Semana | Tarefas | Status |
|---|---|---|
| **1** | Database schema (4 novas tables) | 🔄 Ready to code |
| **1-2** | MetaLearningOptimizer Phase 2 (Thompson Sampling) | 🔄 Ready to code |
| **2-3** | CryptoExpertAgent core (9 tool functions) | 🔄 Ready to code |
| **3** | Data integrations (Binance, on-chain, social) | 🔄 Ready to code |
| **3-4** | Frontend: Calibração IA dashboard | 🔄 Ready to code |
| **4-5** | MetaLearningOptimizer Phase 3 (Bayesian Opt) | 🔄 Ready to code |
| **5** | Testing + validation | 🔄 Ready to code |
| **6** | Deploy production | 🔄 Ready to code |

---

## 9️⃣ Justificativas Técnicas

### Por quê Thompson Sampling para Phase 2?
1. ✅ Cada agente = 1 "arm" (braço) do bandit
2. ✅ Beta distribution rastreia sucesso/fracasso
3. ✅ Exploration natural vs exploitation
4. ✅ CPU overhead < 5%
5. ✅ Implementação simples e rápida (1 semana)

### Por quê Bayesian Optimization para Phase 3?
1. ✅ Modela a superfície de objetivo eficientemente
2. ✅ Requer apenas 50-100 iterações vs GA's 50k+
3. ✅ Proporciona uncertainty estimates
4. ✅ Validação em hold-out set (últimos 7 dias)
5. ✅ Auto-revert se performance piora

### Por quê CryptoExpertAgent é crítico?
1. ✅ Binance tem 6 tipos de investimento (spot, futures, staking, launchpool, farming, margin)
2. ✅ Cada tipo tem dinâmica totalmente diferente
3. ✅ On-chain data + social sentiment = edge competitivo
4. ✅ Padrões históricos (launchpool ROI patterns) são previsíveis
5. ✅ Integra dados que outros agentes não têm acesso

---

## 🔟 Arquivos Atualizados

✅ **ARQUITETURA_AGENTES_IA_PORTFOLIO.md**
- Seção "## 9. Step 13: Ecosystem de Agentes IA Multi-Especializados"
- Decisão principal: 7 agentes MVP + 2 novos
- Análise técnica: GA vs Bayesian vs MAB
- Detalhamento completo de MetaLearningOptimizer (Phase 2 + Phase 3)
- Detalhamento completo de CryptoExpertAgent (9 tool functions)
- Database schema (4 novas tables)
- Configuration extensions (config.yaml)
- Frontend requirements (Calibração IA tab)
- Implementation roadmap (8 semanas)

✅ **DECISOES_CONVERSA_02FEV2026.md** (este arquivo)
- Sumário executivo de todas as decisões
- Justificativas técnicas
- Status de implementação

---

## ✅ Checklist de Confirmação

- [x] MetaLearningOptimizer Agent especificado em Step 13
- [x] CryptoExpertAgent especificado em Step 13
- [x] Thompson Sampling decidido para Phase 2
- [x] Bayesian Optimization decidido para Phase 3
- [x] 4 novas database tables especificadas
- [x] config.yaml extensions documentadas
- [x] Frontend requirements (Calibração IA) especificados
- [x] Implementation roadmap (8 semanas) definido
- [x] Data sources integradas (Binance, on-chain, social)
- [x] 9 CryptoExpert tool functions especificadas
- [x] Tabela de 21 agentes (MVP + Phase 2 + Phase 3) criada
- [x] Todas as decisões registradas no ARQUITETURA_AGENTES_IA_PORTFOLIO.md

---

## 🚀 Próximos Passos

1. **Clonar repositório** da arquitetura
2. **Criar banco de dados** (SQLite + 4 novas tables)
3. **Implementar MetaLearningOptimizer Phase 2** (Thompson Sampling)
4. **Implementar CryptoExpertAgent** (9 tool functions)
5. **Testar em ambiente de staging**
6. **Deploy em produção**

---

**Documento de referência criado em 02/02/2026**  
**Atualizar ARQUITETURA_AGENTES_IA_PORTFOLIO.md quando mudanças forem aprovadas**
