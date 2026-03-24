#!/usr/bin/env python3
"""
setup_wizard.py
===============
Interactive first-run setup wizard for the Matrix LLM Bot.

Guides the user through:
  1. Choosing a model / backend
  2. Entering Matrix server credentials
  3. Selecting rooms to listen in
  4. Setting basic bot personality options
  5. Writing config/config.yaml and downloading the model if needed

Run with:
    python setup_wizard.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Try to import rich / questionary; gracefully degrade to plain input()
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class _FallbackConsole:  # type: ignore[no-redef]
        def print(self, *args: object, **kwargs: object) -> None:
            print(*args)

        def rule(self, *args: object, **kwargs: object) -> None:
            print("─" * 60)

    console = _FallbackConsole()

try:
    import questionary  # type: ignore[import-untyped]

    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


# ---------------------------------------------------------------------------
# Helper wrappers — use questionary when available, else plain input()
# ---------------------------------------------------------------------------


def ask_text(prompt: str, default: str = "") -> str:
    if HAS_QUESTIONARY:
        result = questionary.text(prompt, default=default).ask()
        return result if result is not None else default
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def ask_select(prompt: str, choices: list) -> str:
    if HAS_QUESTIONARY:
        result = questionary.select(prompt, choices=choices).ask()
        return result if result is not None else choices[0]
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("Invalid choice.")


def ask_confirm(prompt: str, default: bool = True) -> bool:
    if HAS_QUESTIONARY:
        result = questionary.confirm(prompt, default=default).ask()
        return result if result is not None else default
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def ask_password(prompt: str) -> str:
    import getpass

    return getpass.getpass(f"{prompt}: ").strip()


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODELS = {
    "Tiny — Phi-3 Mini 3.8B Q4 (≈ 2.2 GB VRAM / RAM)  — great on CPUs": {
        "description": "Microsoft Phi-3 Mini 3.8B — fast, smart, tiny.",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "ram_gb": 3,
    },
    "Small — Mistral 7B Instruct Q4_K_M (≈ 4.8 GB VRAM / RAM) — recommended default": {
        "description": "Mistral 7B Instruct v0.2 — excellent quality/speed balance.",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "ram_gb": 5,
    },
    "Medium — LLaMA-3 8B Instruct Q4_K_M (≈ 5.5 GB VRAM / RAM)": {
        "description": "Meta LLaMA-3 8B — very capable, good for GPU.",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "ram_gb": 6,
    },
    "Large — Mistral 7B Instruct Q8_0 (≈ 7.7 GB VRAM / RAM) — higher quality": {
        "description": "Mistral 7B Q8 — highest quality at 7B size, needs 8 GB RAM.",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "ram_gb": 8,
    },
    "I already have a model / will download it myself": {
        "description": "Skip auto-download — you supply the path.",
        "url": None,
        "filename": None,
        "ram_gb": 0,
    },
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_model(url: str, dest: Path) -> bool:
    """Download a file with progress using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"\n⬇  Downloading model to {dest} …")
    console.print("   This may take several minutes depending on your connection.\n")

    if shutil.which("wget"):
        cmd = ["wget", "-c", "-q", "--show-progress", url, "-O", str(dest)]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "--progress-bar", url, "-o", str(dest)]
    else:
        console.print(
            "[red]Neither wget nor curl found.  Please install one of them "
            "or download the model manually.[/red]"
            if HAS_RICH
            else "ERROR: Neither wget nor curl found."
        )
        return False

    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# YAML writer (no external dependency)
# ---------------------------------------------------------------------------


def write_yaml(path: Path, data: dict) -> None:
    """Write a simple (non-nested-list) YAML file without PyYAML."""
    lines = []

    def _write_section(section_name: str, d: dict, indent: int = 0):
        lines.append(f"{'  ' * indent}{section_name}:")
        for k, v in d.items():
            prefix = "  " * (indent + 1)
            if isinstance(v, list):
                lines.append(f"{prefix}{k}:")
                for item in v:
                    lines.append(f"{prefix}  - \"{item}\"")
            elif isinstance(v, bool):
                lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{prefix}{k}: {v}")
            elif v is None or v == "":
                lines.append(f"{prefix}{k}: \"\"")
            else:
                # Escape quotes
                safe = str(v).replace('"', '\\"')
                lines.append(f"{prefix}{k}: \"{safe}\"")
        lines.append("")

    for section, content in data.items():
        _write_section(section, content)

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main wizard flow
# ---------------------------------------------------------------------------


def main() -> None:
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]Matrix Local LLM Bot — Setup Wizard[/bold cyan]\n"
                "This wizard will guide you through configuring your bot.\n"
                "Press [bold]Ctrl-C[/bold] at any time to quit.",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("  Matrix Local LLM Bot — Setup Wizard")
        print("=" * 60)

    console.print()

    # ------------------------------------------------------------------ #
    # 1. Matrix credentials
    # ------------------------------------------------------------------ #
    console.rule("Step 1: Matrix Server")
    console.print(
        "Enter your Matrix homeserver URL and bot account credentials.\n"
        "Example homeserver: https://matrix.example.org\n"
    )

    homeserver = ask_text("Matrix homeserver URL", "https://matrix.example.org")
    username = ask_text(
        "Bot Matrix username (full ID, e.g. @mybot:example.org)",
        f"@mybot:{homeserver.replace('https://', '')}",
    )
    password = ask_password("Bot account password")
    device_name = ask_text("Device name (shown in session list)", "LLM-Bot")


    console.print("\nEnter the room aliases or IDs the bot should listen to.")
    console.print("  Examples: #general:example.org  or  !abc123:example.org")
    console.print("  Leave blank and press Enter when done.\n")

    rooms: list[str] = []
    while True:
        room = ask_text("Room (or leave blank to finish)", "").strip()
        if not room:
            break
        rooms.append(room)

    # Passive channels prompt
    console.print("\n(Optional) Enter room aliases or IDs where the bot should only respond passively (e.g., when mentioned or based on probability).\n  Leave blank and press Enter when done.")
    passive_channels: list[str] = []
    while True:
        pchan = ask_text("Passive room (or leave blank to finish)", "").strip()
        if not pchan:
            break
        passive_channels.append(pchan)


    # ------------------------------------------------------------------ #
    # 2. Bot personality
    # ------------------------------------------------------------------ #
    console.rule("Step 2: Bot Display Name & Triggers")
    display_name = ask_text("Bot display name (shown in chat)", "GPBot")
    console.print(
        "\nTrigger names are words/phrases that always make the bot respond.\n"
        "Separate multiple triggers with commas.\n"
    )
    triggers_raw = ask_text("Trigger names (comma-separated)", display_name.lower())
    triggers = [t.strip().lower() for t in triggers_raw.split(",") if t.strip()]

    # Respond on 'maybe' in passive channels
    console.print(
        "\n[Optional] In passive channels, the bot uses the LLM to decide if it should reply.\n"
        "If the LLM is unsure (answers 'maybe'), should the bot reply anyway?\n"
        "[bold]No[/bold] = only reply when the LLM is confident.\n"
        "[bold]Yes[/bold] = reply even when the LLM is unsure/ambiguous.\n"
    )
    respond_on_maybe = ask_confirm("Reply in passive channels when LLM is unsure ('maybe')?", default=False)

    # ------------------------------------------------------------------ #
    # 3. Model selection
    # ------------------------------------------------------------------ #
    console.rule("Step 3: Choose an LLM Model")

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan")
        table.add_column("RAM needed")
        for label, info in MODELS.items():
            table.add_row(label, f"{info['ram_gb']} GB" if info["ram_gb"] else "—")
        console.print(table)
        console.print()

    model_choice_label = ask_select("Which model do you want to use?", list(MODELS.keys()))
    model_info = MODELS[model_choice_label]

    model_path = ""
    if model_info["url"] is None:
        # User supplies path
        model_path = ask_text(
            "Full path to your GGUF model file",
            "models/your-model.gguf",
        )
    else:
        models_dir = Path("models")
        dest = models_dir / model_info["filename"]
        model_path = str(dest)

        if dest.exists():
            console.print(f"\n✅ Model file already exists at {dest} — skipping download.")
        else:
            do_download = ask_confirm(
                f"\nDownload {model_info['filename']} now?", default=True
            )
            if do_download:
                success = download_model(model_info["url"], dest)
                if not success:
                    console.print(
                        "Download failed.  Please download the file manually and\n"
                        f"place it at: {dest}"
                    )

    # Hardware mode
    hw_mode = ask_select(
        "Hardware mode",
        ["cpu  — works on any machine (slower)", "gpu  — NVIDIA GPU required (much faster)"],
    )
    hardware_mode = "gpu" if hw_mode.startswith("gpu") else "cpu"
    n_gpu_layers = 0
    if hardware_mode == "gpu":
        layers_raw = ask_text(
            "GPU layers to offload (-1 = all, start with 20 if unsure)", "-1"
        )
        try:
            n_gpu_layers = int(layers_raw)
        except ValueError:
            n_gpu_layers = -1

    # ------------------------------------------------------------------ #
    # 4. Chime-in behaviour
    # ------------------------------------------------------------------ #
    console.rule("Step 4: Chime-In Behaviour")
    console.print(
        "The bot can spontaneously join conversations without being directly asked.\n"
        "0% = never butts in.  20% = joins about 1 in 5 random messages.\n"
    )
    chime_raw = ask_text("Starting chime-in chance (0-100%)", "8")
    try:
        chime_pct = max(0.0, min(100.0, float(chime_raw.strip("%")))) / 100.0
    except ValueError:
        chime_pct = 0.08

    # ------------------------------------------------------------------ #
    # 5. Write config
    # ------------------------------------------------------------------ #
    console.rule("Step 5: Writing Configuration")

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.yaml"

    config_data = {
        "matrix": {
            "homeserver": homeserver,
            "username": username,
            "password": password,
            "device_name": device_name,
            "allowed_rooms": rooms,
            "passive_channels": passive_channels,
            "store_path": "data/matrix_store",
        },
        "llm": {
            "backend": "llamacpp",
            "model_path": model_path,
            "hf_model_id": "",
            "hf_cache_dir": "models/hf_cache",
            "hardware_mode": hardware_mode,
            "n_gpu_layers": n_gpu_layers,
            "context_length": 4096,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
        "bot": {
            "display_name": display_name,
            "trigger_names": triggers,
            "persona_file": "config/bot_persona.txt",
            "chime_in_probability": chime_pct,
            "chime_in_cooldown_messages": 5,
            "chime_in_cooldown_seconds": 60,
            "conversation_history_limit": 12,
            "dossier_token_budget": 512,
            "respond_on_maybe": respond_on_maybe,
        },
        "temperature_controller": {
            "enabled": True,
            "change_interval_minutes": 30,
            "min_temperature": 0.3,
            "max_temperature": 1.15,
            "randomise_chime_in": True,
            "min_chime_in": 0.02,
            "max_chime_in": 0.20,
        },
        "memory": {
            "dossier_dir": "data/dossiers",
            "archive_dir": "data/archives",
            "max_active_entries": 50,
            "compaction_interval_hours": 6,
            "max_summary_chars": 800,
        },
        "logging": {
            "level": "INFO",
            "file": "data/bot.log",
        },
    }

    write_yaml(config_path, config_data)
    console.print(f"\n✅ Configuration written to [bold]{config_path}[/bold]")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    console.print()
    if HAS_RICH:
        console.print(
            Panel(
                "[bold green]Setup complete![/bold green]\n\n"
                "To start the bot, run:\n\n"
                "    [cyan]python -m src.main[/cyan]\n\n"
                "To customise the bot's personality, edit:\n\n"
                "    [cyan]config/bot_persona.txt[/cyan]",
                border_style="green",
            )
        )
    else:
        print("\n=== Setup complete! ===")
        print("\nTo start the bot:")
        print("    python -m src.main")
        print("\nTo customise the bot's personality:")
        print("    edit config/bot_persona.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
