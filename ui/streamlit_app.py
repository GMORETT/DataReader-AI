"""Streamlit chat interface for the Sales AI Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from services.data_loader import load_sales_data
from agent.csv_agent import SalesAgent


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Reader AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal clean CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
        margin: 4px 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="rgba(128,128,128,0.15)", gridwidth=1),
    colorway=["#228be6", "#40c057", "#fab005", "#fa5252", "#7950f2",
              "#20c997", "#fd7e14", "#e64980", "#15aabf", "#82c91e"],
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando dados de vendas...")
def get_dataframe() -> pd.DataFrame:
    return load_sales_data()


@st.cache_resource(show_spinner="Inicializando agente de IA...")
def get_agent() -> SalesAgent:
    df = get_dataframe()
    return SalesAgent(df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Data Reader AI Agent")
    st.caption("Analise com IA")
    st.divider()

    df = get_dataframe()

    st.markdown(f"**{len(df):,}** registros")
    st.markdown(f"**{df['product_id'].nunique():,}** produtos")
    st.markdown(f"**{df['local'].nunique():,}** locais")
    st.markdown(f"**{df['date'].min().date()}** → **{df['date'].max().date()}**")

    st.divider()
    st.markdown("##### Perguntas sugeridas")
    suggestions = [
        "Qual produto foi mais vendido?",
        "Qual local teve maior volume de vendas?",
        "Qual o total de vendas em 2012?",
        "Qual a diferença entre quantidade planejada e realizada?",
        "Qual o impacto das promoções no preço?",
        "Qual o produto com maior receita?",
        "Mostre o resumo mensal de vendas",
        "Qual o nível de serviço médio por local?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", type="tertiary"):
            st.session_state["pending_question"] = s

    st.divider()

    st.markdown("##### Sessão")
    agent = get_agent()
    tracker = agent.tracker
    st.markdown(f"Perguntas: **{tracker.total_queries}**")
    st.markdown(f"Tokens: **{tracker.total_tokens_used:,}**")
    st.markdown(f"Custo: **${tracker.total_cost_usd:.4f}**")

    st.divider()
    st.caption("Powered by LangChain + OpenAI")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Data Reader AI Agent")
st.caption("Faça perguntas sobre os dados em linguagem natural")

tab_chat, tab_dashboard, tab_trace = st.tabs([
    "####  Chat",
    "####  Dashboard",
    "####  Tracking",
])

# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "traces" not in st.session_state:
        st.session_state.traces = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                t = msg["trace"]
                st.caption(
                    f"⏱ {t['total_duration_ms']:.0f}ms · "
                    f"🔧 {len(t['tool_calls'])} tools · "
                    f"📊 {t['tokens']['total']:,} tokens · "
                    f"💰 ${t['estimated_cost_usd']:.4f}"
                )

    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                agent = get_agent()
                result = agent.ask(
                    question=query,
                    conversation_id=st.session_state.conversation_id,
                )
                answer = result["answer"]
                trace = result["trace"]
                trace_dict = trace.to_dict()
                st.session_state.conversation_id = result["conversation_id"]
            st.markdown(answer)
            st.caption(
                f"⏱ {trace_dict['total_duration_ms']:.0f}ms · "
                f"🔧 {len(trace_dict['tool_calls'])} tools · "
                f"📊 {trace_dict['tokens']['total']:,} tokens · "
                f"💰 ${trace_dict['estimated_cost_usd']:.4f}"
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "trace": trace_dict,
        })
        st.session_state.traces.append(trace_dict)

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Faça uma pergunta sobre os dados...") or pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_query = prompt
        st.rerun()

# ---------------------------------------------------------------------------
# Tab: Dashboard
# ---------------------------------------------------------------------------
with tab_dashboard:
    df = get_dataframe()

    # ── Metric cards ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-box"><p class="label">Total Vendido</p>'
            f'<p class="value">{df["actual_quantity"].sum():,.0f}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-box"><p class="label">Receita Total</p>'
            f'<p class="value">R$ {df["actual_revenue"].sum():,.0f}</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-box"><p class="label">Preço Médio</p>'
            f'<p class="value">R$ {df["actual_price"].mean():,.2f}</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-box"><p class="label">Nível de Serviço</p>'
            f'<p class="value">{df["service_level"].mean():.1%}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Top Products ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Top 10 Produtos por Quantidade")
        top_prod = (
            df.groupby("product_id")["actual_quantity"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
            .reset_index()
        )
        fig1 = px.bar(
            top_prod, x="actual_quantity", y="product_id",
            orientation="h", color_discrete_sequence=["#228be6"],
        )
        fig1.update_layout(**PLOTLY_LAYOUT, yaxis_title="", xaxis_title="Quantidade")
        st.plotly_chart(fig1, width="stretch")

    with col_r:
        st.markdown("##### Top 10 Locais por Receita")
        top_loc = (
            df.groupby("local")["actual_revenue"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
            .reset_index()
        )
        fig2 = px.bar(
            top_loc, x="actual_revenue", y="local",
            orientation="h", color_discrete_sequence=["#40c057"],
        )
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="", xaxis_title="Receita (R$)")
        st.plotly_chart(fig2, width="stretch")

    # ── Time series + Promos ──────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("##### Receita Mensal")
        monthly = (
            df.groupby(df["date"].dt.to_period("M"))
            .agg(total_rev=("actual_revenue", "sum"))
            .reset_index()
        )
        monthly["date"] = monthly["date"].astype(str)
        fig3 = px.line(
            monthly, x="date", y="total_rev",
            markers=True, color_discrete_sequence=["#228be6"],
        )
        fig3.update_layout(**PLOTLY_LAYOUT, xaxis_title="Mês", yaxis_title="Receita (R$)")
        st.plotly_chart(fig3, width="stretch")

    with col_r2:
        st.markdown("##### Impacto das Promoções")
        promo = (
            df.groupby("promotion_type")
            .agg(avg_qty=("actual_quantity", "mean"), count=("product_id", "count"))
            .reset_index()
        )
        fig4 = px.bar(
            promo, x="promotion_type", y="avg_qty",
            text="count", color_discrete_sequence=["#fab005"],
        )
        fig4.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Tipo de Promoção",
            yaxis_title="Qtd Média por Transação",
        )
        fig4.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig4, width="stretch")

    # ── Planned vs Actual ─────────────────────────────────────────
    st.markdown("##### Planejado vs Realizado por Local")
    pvsa = (
        df.groupby("local")
        .agg(Planejado=("planned_quantity", "sum"), Realizado=("actual_quantity", "sum"))
        .reset_index()
        .melt(id_vars="local", var_name="Tipo", value_name="Quantidade")
    )
    fig5 = px.bar(
        pvsa, x="local", y="Quantidade", color="Tipo",
        barmode="group", color_discrete_map={"Planejado": "#adb5bd", "Realizado": "#228be6"},
    )
    fig5.update_layout(**PLOTLY_LAYOUT, xaxis_title="Local", yaxis_title="Quantidade")
    st.plotly_chart(fig5, width="stretch")

# ---------------------------------------------------------------------------
# Tab: Traces (Observability)
# ---------------------------------------------------------------------------
with tab_trace:
    traces = st.session_state.get("traces", [])

    if not traces:
        st.info("Nenhum trace ainda. Faça uma pergunta no Chat para ver os dados de observabilidade.")
    else:
        agent = get_agent()
        tk = agent.tracker

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric("Total de Perguntas", tk.total_queries)
        with tc2:
            st.metric("Total de Tokens", f"{tk.total_tokens_used:,}")
        with tc3:
            st.metric("Custo Estimado", f"${tk.total_cost_usd:.4f}")

        st.divider()

        for i, t in enumerate(reversed(traces)):
            idx = len(traces) - i
            with st.expander(f"#{idx} — {t['question'][:80]}", expanded=(i == 0)):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tempo", f"{t['total_duration_ms']:.0f}ms")
                m2.metric("Tokens", f"{t['tokens']['total']:,}")
                m3.metric("Custo", f"${t['estimated_cost_usd']:.4f}")
                m4.metric("Modelo", t["model"])

                if t["tool_calls"]:
                    st.markdown("**Tools utilizadas:**")
                    for tc in t["tool_calls"]:
                        st.markdown(
                            f"- `{tc['name']}` — input: `{tc['input'][:100]}` "
                            f"→ output: `{tc['output'][:100]}…`"
                        )
                else:
                    st.markdown("*Nenhuma tool utilizada (resposta direta do LLM)*")

                st.markdown("**Breakdown de tokens:**")
                st.markdown(
                    f"- Prompt: {t['tokens']['prompt']:,} · "
                    f"Completion: {t['tokens']['completion']:,}"
                )
