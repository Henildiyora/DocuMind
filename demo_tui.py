#!/usr/bin/env python3
"""
Demo script to showcase the DocuMind TUI components.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
import time

console = Console()

def demo_tui():
    """Demonstrate the TUI components."""
    console.clear()

    # Header
    header = Panel.fit(
        "[bold blue]DocuMind AI Agent[/bold blue] | [green]LOCAL MODE[/green]",
        title="DocuMind TUI",
        border_style="blue"
    )
    console.print(header)

    # Status bar
    status = Panel.fit(
        "Status: [yellow]Ready[/yellow] | Mode: [blue]LOCAL[/blue] | Press Ctrl+C to exit",
        border_style="green"
    )
    console.print(status)

    # Chat area demo
    chat_panel = Panel.fit(
        "[bold cyan]You:[/bold cyan] Hello DocuMind!\n\n"
        "[green]DocuMind:[/green] Hello! I'm your AI coding assistant. How can I help you today?\n\n"
        "[bold cyan]You:[/bold cyan] Can you explain this project's architecture?\n\n"
        "[green]DocuMind:[/green] This is a RAG (Retrieval-Augmented Generation) system with dual modes...\n\n"
        "[dim italic yellow]System: Welcome to DocuMind! Type your questions below.[/dim italic yellow]",
        title="Chat History",
        border_style="purple",
        height=15
    )
    console.print(chat_panel)

    # Input area demo
    input_panel = Panel.fit(
        "[dim]Type your message here... (or 'exit' to quit)[/dim]",
        title="Message Input",
        border_style="cyan"
    )
    console.print(input_panel)

    # Footer
    footer = Panel.fit(
        "[bold]Key Bindings:[/bold] Enter=Send | Ctrl+L=Clear | Ctrl+C=Quit",
        border_style="magenta"
    )
    console.print(footer)

    console.print("\n[bold green] TUI Demo Complete![/bold green]")
    console.print("Run [bold cyan]python main.py --tui[/bold cyan] to start the interactive TUI!")

if __name__ == "__main__":
    demo_tui()