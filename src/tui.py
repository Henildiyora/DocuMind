"""
DocuMind Terminal User Interface

A rich, interactive TUI for DocuMind using Textual and Rich libraries.
"""

import asyncio
from typing import Optional
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Input, RichLog, Button, Static
from textual.widget import Widget
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from src.reg import RAGEngine
from src.config import logger


class ChatArea(RichLog):
    """Widget for displaying chat messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "DocuMind Chat"

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat."""
        user_text = Text(f"You: {message}", style="bold cyan")
        self.write(user_text)

    def add_bot_message(self, message: str) -> None:
        """Add a bot response to the chat."""
        bot_text = Text(f"DocuMind: {message}", style="green")
        self.write(bot_text)

    def add_system_message(self, message: str) -> None:
        """Add a system message to the chat."""
        system_text = Text(message, style="yellow italic")
        self.write(system_text)

    def add_error_message(self, message: str) -> None:
        """Add an error message to the chat."""
        error_text = Text(f"Error: {message}", style="bold red")
        self.write(error_text)


class StatusBar(Static):
    """Status bar showing current mode and status."""

    def __init__(self, mode: str = "LOCAL"):
        super().__init__()
        self.mode = mode
        self.update_status("Ready")

    def update_status(self, status: str) -> None:
        """Update the status message."""
        mode_color = "green" if self.mode == "ONLINE" else "blue"
        self.update(f"Mode: [{mode_color}]{self.mode}[/] | Status: {status}")


class DocuMindApp(App):
    """Main DocuMind TUI Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
    }

    Footer {
        background: $primary-darken-2;
        color: $text-muted;
    }

    ChatArea {
        border: solid $primary;
        height: 3fr;
        margin: 1;
    }

    Input {
        border: solid $accent;
        margin: 1;
        height: 3;
    }

    StatusBar {
        background: $panel;
        color: $text;
        padding: 0 1;
        border: solid $primary-darken-1;
        margin: 1;
        height: 1;
    }

    Button {
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("enter", "submit_message", "Send Message"),
    ]

    def __init__(self, rag_engine: RAGEngine):
        super().__init__()
        self.rag_engine = rag_engine
        self.processing = False

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        with Container():
            yield StatusBar(self.rag_engine.mode)
            yield ChatArea(id="chat_area")
            with Horizontal():
                yield Input(placeholder="Type your message here... (or 'exit' to quit)", id="message_input")
                yield Button("Send", id="send_button", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_system_message("Welcome to DocuMind! ")
        chat_area.add_system_message("Type your questions or commands below.")
        chat_area.add_system_message("Commands: 'exit' to quit, 'clear' to clear chat")

        # Focus on input
        self.query_one("#message_input").focus()

    @on(Button.Pressed, "#send_button")
    def handle_send_button(self) -> None:
        """Handle send button press."""
        self.submit_message()

    @on(Input.Submitted, "#message_input")
    def handle_input_submitted(self) -> None:
        """Handle input submission."""
        self.submit_message()

    def action_submit_message(self) -> None:
        """Submit the current message."""
        if self.processing:
            return
        self.submit_message()

    def submit_message(self) -> None:
        """Process and send the message."""
        if self.processing:
            return

        input_widget = self.query_one("#message_input", Input)
        message = input_widget.value.strip()

        if not message:
            return

        # Clear input
        input_widget.value = ""

        # Handle special commands
        if message.lower() in ["exit", "quit"]:
            self.exit()
            return
        elif message.lower() == "clear":
            self.action_clear_chat()
            return

        # Add user message to chat
        chat_area = self.query_one(ChatArea)
        chat_area.add_user_message(message)

        # Start processing
        self.processing = True
        status_bar = self.query_one(StatusBar)
        status_bar.update_status("Thinking...")

        # Process in background
        asyncio.create_task(self.process_message(message))

    async def process_message(self, message: str) -> None:
        """Process a message asynchronously."""
        try:
            # Show thinking indicator
            chat_area = self.query_one(ChatArea)
            status_bar = self.query_one(StatusBar)
            status_bar.update_status("Thinking... ")

            # Get response from RAG engine (run in thread pool to avoid blocking)
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.rag_engine.ask, message
            )

            # Add bot response
            chat_area.add_bot_message(response)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            chat_area = self.query_one(ChatArea)
            chat_area.add_error_message(str(e))

        finally:
            # Reset processing state
            self.processing = False
            status_bar = self.query_one(StatusBar)
            status_bar.update_status("Ready ")

    def action_clear_chat(self) -> None:
        """Clear the chat area."""
        chat_area = self.query_one(ChatArea)
        chat_area.clear()
        chat_area.add_system_message("Chat cleared! ")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def run_tui(rag_engine: RAGEngine) -> None:
    """Run the DocuMind TUI application."""
    app = DocuMindApp(rag_engine)
    app.run()