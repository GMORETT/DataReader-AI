# Data Reader AI Agent

Agente de IA que analisa dados de vendas a partir de um arquivo CSV e responde perguntas em linguagem natural, utilizando **LangChain**, **LangGraph** e **OpenAI**.

---

## Arquitetura

```
sales-ai-agent/
├── main.py                  # Entry-point (CLI / API / Streamlit)
├── config/
│   └── settings.py          # Pydantic Settings (env vars / .env)
├── models/
│   └── schemas.py           # Pydantic data models
├── services/
│   ├── data_loader.py       # Loader sales + loader genérico
│   ├── dataset_profiler.py  # DatasetProfile e inferência de schema
│   └── analytics.py         # Serviço de analytics pré-construído
├── agent/
│   ├── csv_agent.py         # SalesAgent runtime (preset + dynamic hybrid)
│   ├── tools.py             # Tools fixas do sales preset
│   ├── generic_tools.py     # Tools genéricas para qualquer dataset
│   ├── dynamic_tools.py     # Tools específicas inferidas por profiling
│   ├── tool_builder.py      # Composição híbrida de tools
│   ├── prompt_builder.py    # Prompt dinâmico baseado em schema
│   └── prompts.py           # System prompt do modo sales preset
├── api/
│   ├── app.py               # FastAPI application factory
│   └── routes.py            # REST endpoints (/chat, /analytics/*)
├── observability/
│   └── tracker.py           # QueryTracker – traces, tokens, custo por query
├── ui/
│   └── streamlit_app.py     # Streamlit com SalesPreset + DynamicHybrid
├── sales.csv                # Dataset de vendas
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

### Decisões Arquiteturais

| Decisão | Justificativa |
|---------|---------------|
| **LangGraph ReAct Agent** | Agente com raciocínio iterativo (Thought → Action → Observation) para queries complexas |
| **Custom Tools + Python REPL** | 12 ferramentas especializadas para consultas comuns + REPL para queries ad-hoc |
| **3 interfaces** (CLI, API, Streamlit) | Flexibilidade: terminal para devs, API para integrações, Streamlit para stakeholders |
| **Pydantic Settings** | Configuração type-safe via env vars, ideal para containers |
| **Colunas derivadas** | `actual_revenue`, `quantity_diff`, etc. pré-computadas no load para performance |
| **Separação de camadas** | Services ↔ Agent ↔ API desacoplados para testabilidade |
| **Observabilidade** | Trace completo por query: tools usadas, tokens, tempo, custo estimado |
| **Modo híbrido dinâmico** | Upload de CSV + tools genéricas e específicas inferidas automaticamente |

---

## Setup

### Pré-requisitos

- Python 3.11+
- Uma chave de API da OpenAI

### Instalação Local

```bash
# 1. Clone / entre na pasta do projeto
cd sales-ai-agent

# 2. Crie um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite o .env e insira sua OPENAI_API_KEY
```

### Executando

#### CLI Interativo (padrão)
```bash
python main.py
```

#### API REST (FastAPI)
```bash
python main.py --api
# Acesse: http://localhost:8000/docs
```

#### Interface Web (Streamlit)
```bash
python main.py --streamlit
# Acesse: http://localhost:8501
```

### Modos no Streamlit

- **SalesPreset**: usa `sales.csv` e tools especializadas de vendas.
- **DynamicHybrid**: permite upload de qualquer CSV e cria:
  - tools genéricas (`aggregate`, `top_n`, `describe`, `filter`)
  - tools dinâmicas inferidas do schema (ranking por categoria, médias por grupo, trends temporais)
  - fallback `python_repl`

---

## Docker

### Build e execução com Docker Compose

```bash
# Copie e configure o .env
cp .env.example .env
# Edite o .env com sua OPENAI_API_KEY

# API + Streamlit
docker compose up api streamlit

# Apenas CLI interativo
docker compose run --rm cli

# Apenas a API
docker compose up api
```

### Build manual

```bash
docker build -t sales-ai-agent .

# CLI
docker run -it --env-file .env -v ./sales.csv:/app/sales.csv:ro sales-ai-agent

# API
docker run -p 8000:8000 --env-file .env -v ./sales.csv:/app/sales.csv:ro sales-ai-agent --api

# Streamlit
docker run -p 8501:8501 --env-file .env -v ./sales.csv:/app/sales.csv:ro sales-ai-agent --streamlit
```

---

## API Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/api/v1/chat` | Enviar pergunta ao agente |
| `GET`  | `/api/v1/analytics/overview` | Visão geral do dataset |
| `GET`  | `/api/v1/analytics/top-products?n=10&by=quantity` | Top produtos |
| `GET`  | `/api/v1/analytics/top-locations?n=10&by=revenue` | Top locais |
| `GET`  | `/api/v1/analytics/promotions` | Impacto das promoções |
| `GET`  | `/api/v1/analytics/planned-vs-actual` | Planejado vs realizado |
| `GET`  | `/api/v1/analytics/service-level` | Estatísticas de nível de serviço |
| `GET`  | `/health` | Health check |

### Exemplo de chamada

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual produto foi mais vendido?"}'
```

---

## Ferramentas do Agente

O agente possui **13 ferramentas** disponíveis:

| Ferramenta | Descrição |
|-----------|-----------|
| `dataset_overview` | Visão geral: linhas, produtos, locais, período |
| `top_products_by_quantity` | Top N produtos por quantidade vendida |
| `top_locations_by_quantity` | Top N locais por quantidade vendida |
| `top_products_by_revenue` | Top N produtos por receita |
| `top_locations_by_revenue` | Top N locais por receita |
| `total_sales_in_period` | Total de vendas em um período |
| `monthly_sales_summary` | Resumo mensal de vendas |
| `planned_vs_actual_summary` | Comparação planejado vs realizado |
| `planned_vs_actual_by_product` | Diferença planejado vs realizado por produto |
| `promotion_impact` | Impacto das promoções no preço e volume |
| `service_level_stats` | Estatísticas do nível de serviço |
| `service_level_by_location` | Nível de serviço por local |
| `python_repl` | REPL Python/pandas para queries customizadas |

---

## Exemplos de Perguntas

- "Qual produto foi mais vendido?"
- "Qual local teve maior volume de vendas?"
- "Qual foi o total de vendas em 2012?"
- "Qual a diferença entre quantidade planejada e realizada?"
- "Qual o impacto das promoções no preço e volume vendido?"
- "Mostre o top 5 produtos por receita"
- "Qual o nível de serviço médio por warehouse?"
- "Compare as vendas do primeiro e segundo semestre de 2012"

---

## Observabilidade

Cada interação com o agente gera um **trace** completo com:

| Dado | Descrição |
|------|-----------|
| **Tools chamadas** | Nome, input e output de cada ferramenta utilizada |
| **Tokens** | Prompt tokens + completion tokens por chamada |
| **Tempo** | Wall-clock time total da query |
| **Custo estimado** | Estimativa em USD baseada no pricing do modelo |
| **Passos do agente** | Quantas iterações ReAct o agente executou |
| **Origem da tool** | Se cada tool foi `sales_specific`, `dynamic`, `generic` ou `python_repl` |
| **Dataset ID** | Hash curto do dataset ativo para rastrear sessões multi-dataset |

### Onde visualizar

- **Streamlit** → aba "Tracking" com detalhes de cada query + resumo da sessão na sidebar
- **CLI** → trace resumido abaixo de cada resposta (tempo, tools, tokens, custo)
- **API** → campo `metadata` na response do `/chat` com o trace completo em JSON

### Exemplo de response da API com trace

```json
{
  "answer": "O produto mais vendido foi...",
  "conversation_id": "abc-123",
  "metadata": {
    "total_duration_ms": 3200,
    "tokens": { "prompt": 1500, "completion": 200, "total": 1700 },
    "estimated_cost_usd": 0.000345,
    "tool_calls": [
      { "name": "top_products_by_quantity", "input": "10", "duration_ms": 50 }
    ]
  }
}
```

---

## Testes

O projeto possui uma suite de testes automatizados com **pytest**.

### Executar testes unitários (sem LLM, sem custo)

```bash
pytest
```

### Executar testes de integração (usa LLM, consome tokens)

```bash
pytest -m integration
```

### Estrutura dos testes

| Arquivo | O que testa | LLM? |
|---------|-------------|------|
| `test_data_loader.py` | Carregamento e limpeza do CSV, colunas derivadas, erros | Não |
| `test_analytics.py` | Todos os métodos do AnalyticsService com dados controlados | Não |
| `test_tools.py` | 13 LangChain tools: formato JSON, parâmetros, python_repl | Não |
| `test_dataset_profiler.py` | Profiling de schema (tipos, colunas, dataset_id) | Não |
| `test_generic_tools.py` | Tools genéricas para datasets arbitrários | Não |
| `test_dynamic_tools.py` | Geração de tools dinâmicas por dataset | Não |
| `test_streamlit_dynamic_mode.py` | Fluxo base do modo dinâmico (loader + criação do agente) | Não |
| `test_api.py` | Endpoints FastAPI com agent mockado | Não |
| `test_agent.py` | Integração real com LLM (memória, traces, tool usage) | Sim |

---

## Dataset

O arquivo `sales.csv` contém ~203k registros com as colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `product_id` | string | Identificador único do produto |
| `local` | string | Local/warehouse da venda |
| `date` | date | Data da venda (DD/MM/YYYY) |
| `planned_quantity` | int | Quantidade planejada |
| `actual_quantity` | int | Quantidade realmente vendida |
| `planned_price` | float | Preço planejado |
| `promotion_type` | string | Tipo de promoção (None, 1, 2, ...) |
| `actual_price` | float | Preço real praticado |
| `service_level` | float | Nível de serviço (0-1) |
