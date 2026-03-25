"""
thoughts.py
===========
Periodically summarizes recent events and discussions from the bot's perspective, logs these "thoughts" with timestamps, manages log size, and can be used as context for future sessions.

Usage:
    python thoughts.py
    python thoughts.py --data-dir bots/alice/data --config bots/alice/config/config.yaml

This script should be scheduled to run every hour or after every 12 bot responses.
"""

import argparse
import os
import time
import json
from datetime import datetime
from pathlib import Path

from src.config_manager import ConfigManager
from src.llm import LLMBackend
from src.memory_manager import MemoryManager

# --- Defaults ---
DEFAULT_DATA_DIR = "data"
DEFAULT_CONFIG = "config/config.yaml"
CONTEXT_WINDOW = 8192  # Adjust to match your LLM's context window
MAX_LOG_TOKENS = int(CONTEXT_WINDOW * 0.3)  # 30% of context window
HISTORY_LIMIT = 100  # Number of recent messages to consider

# --- Helpers ---
def load_state(state_path):
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {"last_run": 0, "last_msg_idx": 0}

def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f)

def get_conversation_history(memory_mgr, last_msg_idx):
    # This assumes memory_mgr exposes a method to get all messages as a list
    # Each message: {"timestamp": ..., "sender": ..., "text": ...}
    history = memory_mgr.get_all_messages()
    return history[last_msg_idx:]

def summarize_events(llm, history):
    if not history:
        return None
    # Format history for prompt
    formatted = "\n".join([
        f"{msg['timestamp']} {msg['sender']}: {msg['text']}" for msg in history
    ])
    prompt = (
        "You are a valued and important member of this chat server. Summarize, in your own words, what happened in the chat since the last summary. "
        "Include factual events, interesting discussions, and your thoughts or opinions on the users. "
        "Be insightful, how did the topic make you feel? How did you like interacting with the various participants? "
        "What did they say that surprised you or challenged your beliefs?\n\nRecent conversation:\n" + formatted
    )
    summary = llm.generate(system_prompt="You are a valued and important member of this chat server.", user_message=prompt)
    return summary.strip()

def log_thought(summary, log_path):
    timestamp = datetime.now().isoformat()
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}]\n{summary}\n\n")

def compact_log(log_path, config_path):
    # If log is too long, summarize older entries
    with open(log_path, "r") as f:
        lines = f.readlines()
    # Simple token estimation: 1 token ≈ 4 chars
    total_tokens = sum(len(line) // 4 for line in lines)
    if total_tokens <= MAX_LOG_TOKENS:
        return
    # Compact: summarize the first 2/3 of the log
    split_idx = int(len(lines) * 2 / 3)
    old = "".join(lines[:split_idx])
    recent = "".join(lines[split_idx:])
    llm = LLMBackend(ConfigManager(config_path).llm)
    summary = llm.generate(system_prompt="You are a Matrix bot.", user_message=f"Summarize the following log entries for memory:\n{old}")
    with open(log_path, "w") as f:
        f.write(f"[COMPACTED {datetime.now().isoformat()}]\n{summary.strip()}\n\n")
        f.write(recent)

def main():
    parser = argparse.ArgumentParser(description="Generate bot thoughts/reflections")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory for this bot")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to bot config.yaml")
    parser.add_argument("--compact", action="store_true", help="Force log compaction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = args.config
    log_path = data_dir / "thoughts.log"
    state_path = data_dir / "thoughts_state.json"

    cfg = ConfigManager(config_path)
    memory_mgr = MemoryManager(cfg.memory)
    llm = LLMBackend(cfg.llm)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.compact:
        if log_path.exists():
            compact_log(log_path, config_path)
            print("Compaction complete.")
        return

    state = load_state(state_path)
    history = get_conversation_history(memory_mgr, state.get("last_msg_idx", 0))
    if not history:
        print("No new messages to summarize.")
        return
    summary = summarize_events(llm, history)
    if summary:
        log_thought(summary, log_path)
        state["last_run"] = int(time.time())
        state["last_msg_idx"] = state.get("last_msg_idx", 0) + len(history)
        save_state(state, state_path)
        compact_log(log_path, config_path)
        print("Thoughts updated.")
    else:
        print("No summary generated.")

if __name__ == "__main__":
    main()
