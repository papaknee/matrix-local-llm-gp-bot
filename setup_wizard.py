#!/usr/bin/env python3
"""
setup_wizard.py
===============
Interactive first-run setup wizard for the Matrix LLM Bot.

Supports two modes:
  - **Single bot** — the original setup flow (writes ``config/config.yaml``).
  - **Multi-bot fleet** — sets up a route-bot + multiple child bots under
    ``bots/`` and writes ``bots/fleet_config.yaml``.

Additional CLI flags:
    python setup_wizard.py --add-bot        Add a child bot to an existing fleet
    python setup_wizard.py --remove-bot <name>   Remove a child bot from a fleet

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
    "Small — Dolphin Mistral 7B Q4_K_M (≈ 4.8 GB VRAM / RAM) — creative, expressive": {
        "description": "Dolphin 2.6 Mistral 7B — fine-tuned for expressive, uncensored chat.",
        "url": "https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "filename": "dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "ram_gb": 5,
    },
    "Medium — LLaMA-3 8B Instruct Q4_K_M (≈ 5.5 GB VRAM / RAM)": {
        "description": "Meta LLaMA-3 8B — very capable, good for GPU.",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "ram_gb": 6,
    },
    "Medium — Hermes 3 LLaMA 3.1 8B Q4_K_M (≈ 5.5 GB VRAM / RAM) — strong reasoning": {
        "description": "NousResearch Hermes 3 Llama 3.1 8B — excellent instruction following and reasoning.",
        "url": "https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-GGUF/resolve/main/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
        "filename": "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
        "ram_gb": 6,
    },
    "Large — Mistral 7B Instruct Q8_0 (≈ 7.7 GB VRAM / RAM) — higher quality": {
        "description": "Mistral 7B Q8 — highest quality at 7B size, needs 8 GB RAM.",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "ram_gb": 8,
    },
    "Enthusiast — Mixtral 8x7B Instruct Q4_K_M (≈ 26 GB VRAM / RAM) — workstation": {
        "description": "Mixtral 8x7B MoE — flagship open-source quality, needs 32 GB RAM.",
        "url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "ram_gb": 32,
    },
    "I already have a model / will download it myself": {
        "description": "Skip auto-download — you supply the path.",
        "url": None,
        "filename": None,
        "ram_gb": 0,
    },
}

# Route-bot models — smaller models recommended for fast classification
ROUTE_BOT_MODELS = {
    "Tiny — Phi-3 Mini 3.8B Q4 (≈ 2.2 GB) — recommended for route-bot": {
        "description": "Microsoft Phi-3 Mini 3.8B — fast classification.",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "ram_gb": 3,
    },
    "Small — Mistral 7B Instruct Q4_K_M (≈ 4.8 GB)": {
        "description": "Mistral 7B — better classification quality, needs more RAM.",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "ram_gb": 5,
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

    def _write_list_section(section_name: str, lst: list, indent: int = 0):
        lines.append(f"{'  ' * indent}{section_name}:")
        for item in lst:
            if isinstance(item, dict):
                first = True
                for k, v in item.items():
                    prefix = "  " * (indent + 1)
                    bullet = "- " if first else "  "
                    first = False
                    if isinstance(v, bool):
                        lines.append(f"{prefix}{bullet}{k}: {'true' if v else 'false'}")
                    elif isinstance(v, (int, float)):
                        lines.append(f"{prefix}{bullet}{k}: {v}")
                    elif v is None or v == "":
                        lines.append(f"{prefix}{bullet}{k}: \"\"")
                    else:
                        safe = str(v).replace('"', '\\"')
                        lines.append(f"{prefix}{bullet}{k}: \"{safe}\"")
            else:
                prefix = "  " * (indent + 1)
                lines.append(f"{prefix}- \"{item}\"")
        lines.append("")

    for section, content in data.items():
        if isinstance(content, list):
            _write_list_section(section, content)
        else:
            _write_section(section, content)

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared: collect a single child bot's config (reused by single + multi-bot)
# ---------------------------------------------------------------------------


def collect_bot_config(
    bot_name_hint: str = "GPBot",
    config_base_dir: str = "config",
    data_base_dir: str = "data",
    models_dir: str = "models",
    model_catalogue: Optional[dict] = None,
) -> dict:
    """Run the interactive prompts for a single bot and return a config dict.

    This is the shared core used by both single-bot ``main()`` and multi-bot
    ``setup_multi_bot()``.  The caller is responsible for writing the dict to
    disk.
    """
    if model_catalogue is None:
        model_catalogue = MODELS

    # ------------------------------------------------------------------ #
    # Matrix credentials
    # ------------------------------------------------------------------ #
    console.rule("Matrix Server")
    console.print(
        "Enter the Matrix homeserver URL and bot account credentials.\n"
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

    console.print(
        "\n(Optional) Enter room aliases or IDs where the bot should only respond "
        "passively (e.g., when mentioned or based on probability).\n"
        "  Leave blank and press Enter when done."
    )
    passive_channels: list[str] = []
    while True:
        pchan = ask_text("Passive room (or leave blank to finish)", "").strip()
        if not pchan:
            break
        passive_channels.append(pchan)

    # ------------------------------------------------------------------ #
    # Bot personality
    # ------------------------------------------------------------------ #
    console.rule("Bot Display Name & Triggers")
    display_name = ask_text("Bot display name (shown in chat)", bot_name_hint)
    console.print(
        "\nTrigger names are words/phrases that always make the bot respond.\n"
        "Separate multiple triggers with commas.\n"
    )
    triggers_raw = ask_text("Trigger names (comma-separated)", display_name.lower())
    triggers = [t.strip().lower() for t in triggers_raw.split(",") if t.strip()]

    console.print(
        "\n[Optional] In passive channels, the bot uses the LLM to decide if it should reply.\n"
        "If the LLM is unsure (answers 'maybe'), should the bot reply anyway?\n"
    )
    respond_on_maybe = ask_confirm(
        "Reply in passive channels when LLM is unsure ('maybe')?", default=False
    )

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #
    console.rule("Choose an LLM Model")

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan")
        table.add_column("RAM needed")
        for label, info in model_catalogue.items():
            table.add_row(label, f"{info['ram_gb']} GB" if info["ram_gb"] else "—")
        console.print(table)
        console.print()

    model_choice_label = ask_select(
        "Which model do you want to use?", list(model_catalogue.keys())
    )
    model_info = model_catalogue[model_choice_label]

    model_path = ""
    if model_info["url"] is None:
        model_path = ask_text(
            "Full path to your GGUF model file",
            f"{models_dir}/your-model.gguf",
        )
    else:
        dest = Path(models_dir) / model_info["filename"]
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
    # Chime-in behaviour
    # ------------------------------------------------------------------ #
    console.rule("Chime-In Behaviour")
    console.print(
        "The bot can spontaneously join conversations without being directly asked.\n"
        "0% = never butts in.  20% = joins about 1 in 5 random messages.\n"
    )
    chime_raw = ask_text("Starting chime-in chance (0-100%)", "8")
    try:
        chime_pct = max(0.0, min(100.0, float(chime_raw.strip("%")))) / 100.0
    except ValueError:
        chime_pct = 0.08

    persona_file = f"{config_base_dir}/bot_persona.txt"

    return {
        "matrix": {
            "homeserver": homeserver,
            "username": username,
            "password": password,
            "device_name": device_name,
            "allowed_rooms": rooms,
            "passive_channels": passive_channels,
            "store_path": f"{data_base_dir}/matrix_store",
        },
        "llm": {
            "backend": "llamacpp",
            "model_path": model_path,
            "hf_model_id": "",
            "hf_cache_dir": f"{models_dir}/hf_cache",
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
            "persona_file": persona_file,
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
            "dossier_dir": f"{data_base_dir}/dossiers",
            "archive_dir": f"{data_base_dir}/archives",
            "max_active_entries": 50,
            "compaction_interval_hours": 6,
            "max_summary_chars": 800,
        },
        "logging": {
            "level": "INFO",
            "file": f"{data_base_dir}/bot.log",
        },
    }


# ---------------------------------------------------------------------------
# Route-bot config collection (LLM section only)
# ---------------------------------------------------------------------------


def collect_route_bot_config(models_dir: str = "models") -> dict:
    """Collect the route-bot's LLM configuration (no Matrix account needed)."""
    console.rule("Route-Bot Model")
    console.print(
        "The route-bot is a small, fast model that monitors conversations and\n"
        "decides which child bot should respond.  It never speaks in chat.\n"
        "A smaller model is recommended — it only needs to classify messages.\n"
    )

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan")
        table.add_column("RAM needed")
        for label, info in ROUTE_BOT_MODELS.items():
            table.add_row(label, f"{info['ram_gb']} GB" if info["ram_gb"] else "—")
        console.print(table)
        console.print()

    model_choice_label = ask_select(
        "Which model for the route-bot?", list(ROUTE_BOT_MODELS.keys())
    )
    model_info = ROUTE_BOT_MODELS[model_choice_label]

    model_path = ""
    if model_info["url"] is None:
        model_path = ask_text(
            "Full path to route-bot GGUF model file",
            f"{models_dir}/your-route-model.gguf",
        )
    else:
        dest = Path(models_dir) / model_info["filename"]
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

    hw_mode = ask_select(
        "Hardware mode for route-bot",
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

    return {
        "llm": {
            "backend": "llamacpp",
            "model_path": model_path,
            "hardware_mode": hardware_mode,
            "n_gpu_layers": n_gpu_layers,
            "context_length": 4096,
            "max_new_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }


# ---------------------------------------------------------------------------
# Multi-bot setup flow
# ---------------------------------------------------------------------------


def setup_multi_bot() -> None:
    """Interactive multi-bot fleet setup wizard."""
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]Multi-Bot Fleet Setup[/bold cyan]\n"
                "This will configure a route-bot and one or more child bots.\n"
                "Each child bot gets its own Matrix account, persona, and model.\n"
                "The route-bot monitors conversations and routes messages to the\n"
                "appropriate child bot.  Only one LLM is in memory at a time.",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("  Multi-Bot Fleet Setup")
        print("=" * 60)

    console.print()

    # --- Route-bot ---
    route_cfg = collect_route_bot_config()

    route_dir = Path("bots/route_bot/config")
    route_dir.mkdir(parents=True, exist_ok=True)
    route_config_path = route_dir / "config.yaml"
    write_yaml(route_config_path, route_cfg)
    console.print(f"\n✅ Route-bot config written to {route_config_path}")

    # --- Child bots ---
    child_bots: list[dict] = []  # list of {"name": ..., "config_path": ...}

    while True:
        console.print()
        console.rule(f"Child Bot #{len(child_bots) + 1}")
        bot_name = ask_text("Short name for this bot (e.g. alice, bob)", "").strip()
        if not bot_name:
            if not child_bots:
                console.print("You need at least one child bot.")
                continue
            break

        # Sanitise name
        safe_name = bot_name.lower().replace(" ", "_")
        bot_base = f"bots/{safe_name}"
        config_base = f"{bot_base}/config"
        data_base = f"{bot_base}/data"

        console.print(f"\nConfiguring child bot: [bold]{safe_name}[/bold]\n")

        bot_cfg = collect_bot_config(
            bot_name_hint=bot_name,
            config_base_dir=config_base,
            data_base_dir=data_base,
        )

        # Write child bot config
        cfg_dir = Path(config_base)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        Path(data_base).mkdir(parents=True, exist_ok=True)
        child_config_path = cfg_dir / "config.yaml"
        write_yaml(child_config_path, bot_cfg)

        # Create a default persona file if it doesn't exist
        persona_path = Path(bot_cfg["bot"]["persona_file"])
        if not persona_path.exists():
            persona_path.parent.mkdir(parents=True, exist_ok=True)
            persona_path.write_text(
                f"You are {bot_name}, a witty bot living in a Matrix chat server.\n"
                "Keep responses short and punchy. Be yourself.\n",
                encoding="utf-8",
            )
            console.print(f"✅ Default persona written to {persona_path}")

        child_bots.append({
            "name": safe_name,
            "config_path": str(child_config_path),
        })
        console.print(f"\n✅ Child bot {safe_name!r} configured at {child_config_path}")

        if not ask_confirm("\nAdd another child bot?", default=False):
            break

    # --- Write fleet config ---
    fleet_data = {
        "route_bot": {"config_path": str(route_config_path)},
        "bots": child_bots,
    }
    fleet_path = Path("bots/fleet_config.yaml")
    fleet_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(fleet_path, fleet_data)

    console.print(f"\n✅ Fleet config written to {fleet_path}")
    console.print()

    if HAS_RICH:
        bot_list = "\n".join(f"    • {b['name']}" for b in child_bots)
        console.print(
            Panel(
                "[bold green]Multi-bot setup complete![/bold green]\n\n"
                f"Route-bot config: [cyan]{route_config_path}[/cyan]\n"
                f"Child bots:\n{bot_list}\n\n"
                "To start the fleet, run:\n\n"
                "    [cyan]python -m src.main --multi[/cyan]\n\n"
                "Or just [cyan]python -m src.main[/cyan] — it auto-detects the fleet config.\n\n"
                "To add another bot later:\n\n"
                "    [cyan]python setup_wizard.py --add-bot[/cyan]",
                border_style="green",
            )
        )
    else:
        print("\n=== Multi-bot setup complete! ===")
        print(f"\nRoute-bot config: {route_config_path}")
        print("Child bots:", [b["name"] for b in child_bots])
        print("\nTo start the fleet:")
        print("    python -m src.main --multi")
        print("\nTo add another bot later:")
        print("    python setup_wizard.py --add-bot")


# ---------------------------------------------------------------------------
# --add-bot: add a child bot to an existing fleet
# ---------------------------------------------------------------------------


def add_bot_to_fleet() -> None:
    """Add a new child bot to an existing fleet config."""
    fleet_path = "bots/fleet_config.yaml"
    if not Path(fleet_path).exists():
        print(f"\n[ERROR] Fleet config not found at {fleet_path}")
        print("Run the multi-bot setup wizard first.\n")
        sys.exit(1)

    # Load existing fleet config
    import yaml
    with open(fleet_path, "r", encoding="utf-8") as fh:
        fleet_data = yaml.safe_load(fh) or {}

    existing_names = {b["name"] for b in fleet_data.get("bots", [])}

    console.print()
    console.rule("Add Child Bot to Fleet")
    bot_name = ask_text("Short name for the new bot (e.g. charlie)", "").strip()
    if not bot_name:
        print("No name provided — aborting.")
        return

    safe_name = bot_name.lower().replace(" ", "_")
    if safe_name in existing_names:
        print(f"\n[ERROR] Bot {safe_name!r} already exists in the fleet.")
        sys.exit(1)

    bot_base = f"bots/{safe_name}"
    config_base = f"{bot_base}/config"
    data_base = f"{bot_base}/data"

    console.print(f"\nConfiguring child bot: [bold]{safe_name}[/bold]\n")
    bot_cfg = collect_bot_config(
        bot_name_hint=bot_name,
        config_base_dir=config_base,
        data_base_dir=data_base,
    )

    cfg_dir = Path(config_base)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    Path(data_base).mkdir(parents=True, exist_ok=True)
    child_config_path = cfg_dir / "config.yaml"
    write_yaml(child_config_path, bot_cfg)

    # Create default persona
    persona_path = Path(bot_cfg["bot"]["persona_file"])
    if not persona_path.exists():
        persona_path.parent.mkdir(parents=True, exist_ok=True)
        persona_path.write_text(
            f"You are {bot_name}, a witty bot living in a Matrix chat server.\n"
            "Keep responses short and punchy. Be yourself.\n",
            encoding="utf-8",
        )

    # Update fleet config
    if "bots" not in fleet_data:
        fleet_data["bots"] = []
    fleet_data["bots"].append({
        "name": safe_name,
        "config_path": str(child_config_path),
    })
    write_yaml(Path(fleet_path), fleet_data)

    console.print(f"\n✅ Bot {safe_name!r} added to fleet.")
    console.print(f"   Config: {child_config_path}")
    console.print(f"   Persona: {persona_path}")
    console.print(f"\nRestart the fleet to pick up the new bot:")
    console.print("    python -m src.main --multi\n")


# ---------------------------------------------------------------------------
# --remove-bot: remove a child bot from an existing fleet
# ---------------------------------------------------------------------------


def remove_bot_from_fleet(name: str) -> None:
    """Remove a child bot from the fleet config by name."""
    fleet_path = "bots/fleet_config.yaml"
    if not Path(fleet_path).exists():
        print(f"\n[ERROR] Fleet config not found at {fleet_path}")
        sys.exit(1)

    import yaml
    with open(fleet_path, "r", encoding="utf-8") as fh:
        fleet_data = yaml.safe_load(fh) or {}

    bots = fleet_data.get("bots", [])
    new_bots = [b for b in bots if b.get("name") != name]

    if len(new_bots) == len(bots):
        print(f"\n[ERROR] Bot {name!r} not found in fleet.")
        sys.exit(1)

    if not new_bots:
        print("\n[ERROR] Cannot remove the last bot — fleet needs at least one.")
        sys.exit(1)

    fleet_data["bots"] = new_bots
    write_yaml(Path(fleet_path), fleet_data)

    console.print(f"\n✅ Bot {name!r} removed from fleet config.")
    console.print(f"   Note: bot files under bots/{name}/ were NOT deleted.")
    console.print(f"   Remove them manually if desired.\n")


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

    # Ask deployment mode
    mode = ask_select(
        "How would you like to deploy?",
        [
            "Single bot  — one bot, one model (current default)",
            "Multi-bot fleet  — multiple bots sharing one GPU via a route-bot",
        ],
    )

    if mode.startswith("Multi"):
        setup_multi_bot()
        return

    # ------------------------------------------------------------------ #
    # Single-bot mode (original flow, using shared collector)
    # ------------------------------------------------------------------ #
    config_data = collect_bot_config()

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.yaml"
    write_yaml(config_path, config_data)
    console.print(f"\n✅ Configuration written to [bold]{config_path}[/bold]")

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
        # Handle CLI flags
        if "--add-bot" in sys.argv:
            add_bot_to_fleet()
        elif "--remove-bot" in sys.argv:
            idx = sys.argv.index("--remove-bot")
            if idx + 1 >= len(sys.argv):
                print("Usage: python setup_wizard.py --remove-bot <name>")
                sys.exit(1)
            remove_bot_from_fleet(sys.argv[idx + 1])
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
