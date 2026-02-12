import sys
from pathlib import Path

import typer
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_settings
from router import DatasetRouter, build_prompt
from llm_client import OpenRouterClient, LLMResponse

app = typer.Typer()
console = Console()


@app.command()
def hello():
    """
    Simple environment check.
    """
    console.print(Panel.fit("Semantic Dataset Router", style="bold magenta"))
    console.print(f"[bold]Python Executable:[/bold] {sys.executable}")
    console.print(f"[bold]Python Version:[/bold] [green]{sys.version.split()[0]}[/green]")
    console.print(f"[bold]Pandas Version:[/bold] [cyan]{pd.__version__}[/cyan]")
    console.print(f"[bold]Numpy Version:[/bold] [cyan]{np.__version__}[/cyan]")
    console.print("\n[bold green]Environment is ready.[/bold green]")


@app.command()
def route(
    query: str = typer.Argument(..., help="User question to route to a dataset"),
    threshold: float | None = typer.Option(None, "--threshold", "-t", help="Min cosine similarity to attach a dataset"),
    model: str | None = typer.Option(None, "--model", "-m", help="Sentence transformer model name"),
):
    """
    Route a query to the best matching dataset by semantic similarity.
    Prints similarity scores and whether a dataset was selected (above threshold).
    """
    app_settings = get_settings()
    threshold = threshold if threshold is not None else app_settings.routing_threshold
    model = model or app_settings.embedding_model
    router = DatasetRouter(model_name=model)
    result = router.route(query=query, threshold=threshold)

    table = Table(title="Similarity scores")
    table.add_column("Dataset", style="cyan")
    table.add_column("Score", justify="right", style="green")
    for did, score in result.all_scores.items():
        marker = " *" if did == result.best_dataset_id else ""
        table.add_row(did + marker, f"{score:.4f}")

    console.print(table)
    console.print(f"Threshold: [bold]{result.threshold}[/bold]")
    if result.best_dataset_id:
        console.print(f"Selected: [bold green]{result.best_dataset_id}[/bold green] (score {result.best_score:.4f} >= {result.threshold})")
    else:
        console.print(f"Selected: [dim]none[/dim] (best score {result.best_score:.4f} < {result.threshold})")


def _print_llm_response(resp: LLMResponse) -> None:
    """Format and print LLM response with usage info."""
    console.print(Panel(resp.content.strip(), title="Answer", border_style="green"))
    usage_table = Table(show_header=False)
    usage_table.add_column(style="dim")
    usage_table.add_column(style="cyan")
    usage_table.add_row("Model", resp.model)
    usage_table.add_row("Finish", resp.finish_reason)
    usage_table.add_row("Tokens", f"{resp.prompt_tokens} prompt + {resp.completion_tokens} completion = {resp.total_tokens} total")
    console.print(Panel(usage_table, title="Usage", border_style="dim"))


@app.command()
def query(
    user_query: str = typer.Argument(..., help="User question"),
    threshold: float | None = typer.Option(None, "--threshold", "-t", help="Min cosine similarity to include dataset context"),
    instruction: str | None = typer.Option(
        None,
        "--instruction", "-i",
        help="General instruction text (overrides versioned instruction)",
    ),
    instruction_version: str | None = typer.Option(
        None,
        "--instruction-version",
        help="Version key from general_instructions.py (e.g. 1). Default when -i not set.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Sentence transformer model name"),
    show_scores: bool = typer.Option(False, "--scores", "-s", help="Print routing scores before the prompt"),
    call_llm: bool = typer.Option(False, "--call", "-c", help="Send prompt to OpenRouter LLM and show the answer"),
    llm_model: str | None = typer.Option(None, "--llm-model", "-l", help="OpenRouter model id (e.g. openai/gpt-4o-mini)"),
    temperature: float = typer.Option(0.7, "--temperature", help="LLM sampling temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max completion tokens"),
):
    """
    Build a prompt with optional dataset context. Use --call to send it to OpenRouter and print the LLM answer.
    """
    app_settings = get_settings()
    threshold = threshold if threshold is not None else app_settings.routing_threshold
    model = model or app_settings.embedding_model
    llm_model = llm_model or app_settings.default_llm_model
    router = DatasetRouter(model_name=model)
    result = router.route(query=user_query, threshold=threshold)
    prompt = build_prompt(
        user_query,
        result,
        general_instruction=instruction,
        instruction_version=instruction_version,
    )

    if show_scores:
        console.print(Panel(f"Best: {result.best_dataset_id or 'none'} (score={result.best_score:.4f}, threshold={result.threshold})", title="Routing"))
    if not call_llm:
        console.print(Panel(prompt, title="Prompt", border_style="blue"))
        return

    try:
        client = OpenRouterClient(default_model=llm_model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(Panel(prompt, title="Prompt (sent to LLM)", border_style="blue"))
    with console.status("Calling OpenRouter..."):
        resp = client.chat(prompt, model=llm_model, temperature=temperature, max_tokens=max_tokens)
    _print_llm_response(resp)


if __name__ == "__main__":
    app()
