"""Streamlit UI supporting sales preset and dynamic hybrid dataset mode."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.csv_agent import SalesAgent, create_dynamic_agent
from services.data_loader import load_csv_generic_from_bytes, load_sales_data
from services.dataset_profiler import profile_dataframe


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


def _dataset_hash(file_bytes: bytes, mode: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:12] if file_bytes else "sales_default"
    return f"{mode}:{digest}"


@st.cache_data(show_spinner=False)
def get_sales_dataframe() -> pd.DataFrame:
    return load_sales_data()


@st.cache_data(show_spinner=False)
def get_dynamic_dataframe(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return load_csv_generic_from_bytes(file_bytes, filename=file_name)


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "conversation_id": None,
        "pending_query": None,
        "traces": [],
        "dataset_key": "",
        "active_mode": "SalesPreset",
        "agent": None,
        "df": None,
        "profile": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_chat_state() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.pending_query = None
    st.session_state.traces = []


def _build_agent_for_mode(mode: str, df: pd.DataFrame, dataset_name: str) -> SalesAgent:
    if mode == "DynamicHybrid":
        return create_dynamic_agent(df, dataset_name=dataset_name)
    return SalesAgent(df, dataset_id="sales_default")


def ensure_runtime(mode: str, uploaded_file) -> None:
    if mode == "DynamicHybrid":
        if uploaded_file is None:
            st.session_state.df = None
            st.session_state.profile = None
            st.session_state.agent = None
            st.session_state.dataset_key = ""
            return
        file_bytes = uploaded_file.getvalue()
        dataset_key = _dataset_hash(file_bytes, mode)
        if st.session_state.dataset_key != dataset_key:
            df = get_dynamic_dataframe(file_bytes, uploaded_file.name)
            agent = _build_agent_for_mode(mode, df, uploaded_file.name)
            st.session_state.df = df
            st.session_state.agent = agent
            st.session_state.profile = profile_dataframe(df, dataset_name=uploaded_file.name)
            st.session_state.dataset_key = dataset_key
            reset_chat_state()
    else:
        dataset_key = "SalesPreset:sales_default"
        if st.session_state.dataset_key != dataset_key:
            df = get_sales_dataframe()
            agent = _build_agent_for_mode(mode, df, "sales.csv")
            st.session_state.df = df
            st.session_state.agent = agent
            st.session_state.profile = None
            st.session_state.dataset_key = dataset_key
            reset_chat_state()


init_session_state()

with st.sidebar:
    st.title("Data Reader AI Agent")
    st.caption("Analise com IA")
    st.divider()
    mode = st.radio(
        "Modo",
        ["SalesPreset", "DynamicHybrid"],
        index=0 if st.session_state.active_mode == "SalesPreset" else 1,
        help="SalesPreset usa o schema fixo de vendas; DynamicHybrid aceita qualquer CSV.",
    )
    st.session_state.active_mode = mode

    uploaded_file = None
    if mode == "DynamicHybrid":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Faça upload de qualquer CSV para gerar tools dinâmicas.")

    ensure_runtime(mode, uploaded_file)

    df = st.session_state.df
    agent = st.session_state.agent

    if df is None:
        st.warning("Faça upload de um CSV para começar no modo DynamicHybrid.")
        st.stop()

    st.markdown(f"**{len(df):,}** registros")
    st.markdown(f"**{len(df.columns):,}** colunas")
    if mode == "SalesPreset":
        if "product_id" in df.columns:
            st.markdown(f"**{df['product_id'].nunique():,}** produtos")
        if "local" in df.columns:
            st.markdown(f"**{df['local'].nunique():,}** locais")
    else:
        profile = st.session_state.profile
        st.markdown(f"Numéricas: **{len(profile.numeric_columns)}**")
        st.markdown(f"Categóricas: **{len(profile.categorical_columns)}**")
        st.markdown(f"Datetime: **{len(profile.datetime_columns)}**")

    st.divider()
    st.markdown("##### Perguntas sugeridas")
    if mode == "SalesPreset":
        suggestions = [
            "Qual produto foi mais vendido?",
            "Qual local teve maior volume de vendas?",
            "Qual o total de vendas em 2012?",
            "Qual a diferença entre quantidade planejada e realizada?",
            "Qual o impacto das promoções no preço?",
            "Qual o produto com maior receita?",
        ]
    else:
        suggestions = [
            "Me dê um overview do dataset",
            "Quais colunas têm mais nulos?",
            "Qual o top 5 por uma métrica importante?",
            "Existe tendência temporal nos dados?",
            "Compare médias entre categorias principais",
            "Quais outliers aparecem nas colunas numéricas?",
        ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", type="tertiary"):
            st.session_state["pending_question"] = s

    st.divider()

    st.markdown("##### Sessão")
    tracker = agent.tracker
    st.markdown(f"Perguntas: **{tracker.total_queries}**")
    st.markdown(f"Tokens: **{tracker.total_tokens_used:,}**")
    st.markdown(f"Custo: **${tracker.total_cost_usd:.4f}**")
    if st.session_state.profile:
        st.markdown(f"Dataset ID: **{st.session_state.profile.dataset_id}**")
    else:
        st.markdown("Dataset ID: **sales_default**")

    st.divider()
    st.caption("Powered by LangChain + OpenAI")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Data Reader AI Agent")
st.caption("Faça perguntas sobre os dados em linguagem natural (modo fixo ou dinâmico)")

tab_chat, tab_dashboard, tab_trace = st.tabs([
    "####  Chat",
    "####  Dashboard",
    "####  Tracking",
])

# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------
with tab_chat:
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
    df = st.session_state.df
    if st.session_state.active_mode == "SalesPreset":
        c1, c2, c3, c4 = st.columns(4)
        if "actual_quantity" in df.columns:
            with c1:
                st.markdown(
                    f'<div class="metric-box"><p class="label">Total Vendido</p>'
                    f'<p class="value">{df["actual_quantity"].sum():,.0f}</p></div>',
                    unsafe_allow_html=True,
                )
        if "actual_revenue" in df.columns:
            with c2:
                st.markdown(
                    f'<div class="metric-box"><p class="label">Receita Total</p>'
                    f'<p class="value">R$ {df["actual_revenue"].sum():,.0f}</p></div>',
                    unsafe_allow_html=True,
                )
        if "actual_price" in df.columns:
            with c3:
                st.markdown(
                    f'<div class="metric-box"><p class="label">Preço Médio</p>'
                    f'<p class="value">R$ {df["actual_price"].mean():,.2f}</p></div>',
                    unsafe_allow_html=True,
                )
        if "service_level" in df.columns:
            with c4:
                st.markdown(
                    f'<div class="metric-box"><p class="label">Nível de Serviço</p>'
                    f'<p class="value">{df["service_level"].mean():.1%}</p></div>',
                    unsafe_allow_html=True,
                )
    else:
        profile = st.session_state.profile
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{profile.row_count:,}")
        m2.metric("Columns", f"{profile.column_count:,}")
        m3.metric("Numeric", f"{len(profile.numeric_columns)}")
        m4.metric("Categorical", f"{len(profile.categorical_columns)}")

        numeric_cols = profile.numeric_columns
        cat_cols = profile.categorical_columns
        dt_cols = profile.datetime_columns

        if numeric_cols:
            col = numeric_cols[0]
            st.markdown(f"##### Distribution: `{col}`")
            fig = px.histogram(df, x=col, nbins=40)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        if cat_cols:
            cat = cat_cols[0]
            st.markdown(f"##### Top Categories: `{cat}`")
            top_counts = df[cat].astype(str).value_counts().head(15).reset_index()
            top_counts.columns = [cat, "count"]
            fig = px.bar(top_counts, x="count", y=cat, orientation="h")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        if dt_cols and numeric_cols:
            date_col = dt_cols[0]
            value_col = numeric_cols[0]
            st.markdown(f"##### Trend: `{value_col}` over `{date_col}`")
            tmp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(tmp[date_col]):
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True)
            tmp = tmp.dropna(subset=[date_col])
            tmp["_period"] = tmp[date_col].dt.to_period("M").astype(str)
            trend = tmp.groupby("_period")[value_col].sum().reset_index()
            fig = px.line(trend, x="_period", y=value_col, markers=True)
            fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Period")
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab: Traces (Observability)
# ---------------------------------------------------------------------------
with tab_trace:
    traces = st.session_state.get("traces", [])

    if not traces:
        st.info("Nenhum trace ainda. Faça uma pergunta no Chat para ver os dados de observabilidade.")
    else:
        agent = st.session_state.agent
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
                            f"- `{tc['name']}` ({tc.get('origin', 'unknown')}) — input: `{tc['input'][:100]}` "
                            f"→ output: `{tc['output'][:100]}…`"
                        )
                else:
                    st.markdown("*Nenhuma tool utilizada (resposta direta do LLM)*")

                st.markdown("**Breakdown de tokens:**")
                st.markdown(
                    f"- Prompt: {t['tokens']['prompt']:,} · "
                    f"Completion: {t['tokens']['completion']:,}"
                )
                st.markdown(f"- Dataset ID: `{t.get('dataset_id', 'unknown')}`")
