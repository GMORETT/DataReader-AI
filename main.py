"""CLI entry-point for the Sales AI Agent.

Usage:
    python main.py              # interactive chat
    python main.py --api        # start FastAPI server
    python main.py --streamlit  # start Streamlit UI
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
})
console = Console(theme=custom_theme)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def run_cli() -> None:
    """Interactive terminal chat loop."""
    from config.settings import get_settings
    from services.data_loader import load_sales_data
    from agent.csv_agent import SalesAgent

    settings = get_settings()
    setup_logging(settings.log_level)

    console.print(
        Panel(
            "[bold cyan]Sales AI Agent[/bold cyan]\n"
            "Faça perguntas sobre os dados de vendas em linguagem natural.\n"
            "Digite [bold]sair[/bold] ou [bold]exit[/bold] para encerrar.",
            title="🤖 Bem-vindo",
            border_style="cyan",
        )
    )

    console.print("[info]Carregando dados...[/info]")
    df = load_sales_data()
    console.print(
        f"[success]✓ {len(df):,} registros carregados "
        f"({df['product_id'].nunique()} produtos, {df['local'].nunique()} locais)[/success]"
    )

    console.print("[info]Inicializando agente de IA...[/info]")
    agent = SalesAgent(df)
    console.print("[success]✓ Agente pronto![/success]\n")

    conversation_id: str | None = None

    while True:
        try:
            question = console.input("[bold cyan]Você >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[warning]Encerrando...[/warning]")
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit", "q"}:
            console.print("[warning]Até logo! 👋[/warning]")
            break

        with console.status("[cyan]Pensando...[/cyan]", spinner="dots"):
            try:
                result = agent.ask(question, conversation_id=conversation_id)
                conversation_id = result["conversation_id"]
            except Exception as e:
                console.print(f"[error]Erro: {e}[/error]")
                continue

        console.print()
        console.print(Panel(Markdown(result["answer"]), title="🤖 Agente", border_style="green"))

        trace = result["trace"]
        tools_used = ", ".join(tc.name for tc in trace.tool_calls) or "nenhuma"
        console.print(
            f"  [dim]⏱ {trace.total_duration_ms:.0f}ms · "
            f"🔧 {tools_used} · "
            f"📊 {trace.total_tokens:,} tokens · "
            f"💰 ${trace.estimated_cost_usd:.4f}[/dim]"
        )
        console.print()


def run_api() -> None:
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import get_settings

    settings = get_settings()
    setup_logging(settings.log_level)

    console.print(
        Panel(
            f"[bold cyan]API Server[/bold cyan]\n"
            f"http://{settings.api_host}:{settings.api_port}\n"
            f"Docs: http://localhost:{settings.api_port}/docs",
            title="🚀 Starting",
            border_style="cyan",
        )
    )

    uvicorn.run(
        "api.app:create_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        factory=True,
    )


def run_streamlit() -> None:
    """Launch the Streamlit UI."""
    from config.settings import get_settings

    settings = get_settings()
    setup_logging(settings.log_level)

    ui_path = os.path.join(os.path.dirname(__file__), "ui", "streamlit_app.py")

    console.print(
        Panel(
            f"[bold cyan]Streamlit UI[/bold cyan]\n"
            f"http://localhost:{settings.streamlit_port}",
            title="🎨 Starting",
            border_style="cyan",
        )
    )

    import subprocess
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", ui_path,
            "--server.port", str(settings.streamlit_port),
            "--server.headless", "true",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sales AI Agent - Analyze sales data with AI",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--api", action="store_true", help="Start FastAPI REST API server")
    group.add_argument("--streamlit", action="store_true", help="Start Streamlit web UI")
    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.streamlit:
        run_streamlit()
    else:
        run_cli()


if __name__ == "__main__":
    main()
