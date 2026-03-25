# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Multi-Bot Orchestration

### Added

- **Multi-bot fleet mode** — run multiple personality bots behind a single route-bot
  that decides which bot should respond, ensuring only one LLM is in GPU/RAM at a
  time. Ideal for single-GPU machines that want several distinct bot personalities.

- **Route-bot** — a lightweight, always-loaded LLM that silently monitors all rooms
  and classifies incoming messages to decide which child bot (if any) should respond.
  The route-bot never speaks in chat.

- **`src/orchestrator.py`** — core multi-bot orchestrator:
  - Connects all child bot Matrix clients concurrently.
  - Maintains a priority queue ordered by `server_timestamp` so the oldest
    unanswered message is always processed first (FIFO guarantee).
  - Deduplicates messages across overlapping bot room memberships.
  - Three-phase routing: trigger-name match → chime-in probability → route-bot
    LLM classification.
  - Atomic LLM swap: unloads route-bot → loads child bot → generates response →
    unloads child bot → reloads route-bot. Protected by `asyncio.Lock` so only
    one model is ever in memory.
  - Shared room-history buffer (50 messages per room) ensures context is never
    lost during LLM swaps.

- **`src/fleet_config.py`** — fleet configuration loader/validator:
  - Parses `bots/fleet_config.yaml` referencing the route-bot config and one or
    more child bot configs.
  - Validates file existence, uniqueness of bot names, and required fields.
  - `add_bot()` / `remove_bot()` / `save()` methods for programmatic fleet
    management.

- **`src/child_bot_context.py`** — lightweight per-bot state container:
  - Holds config, persona, memory manager, temperature controller, and Matrix
    client for each child bot — everything except a loaded LLM.
  - Per-room cooldown tracking (`last_post_time`, `messages_since_post`).
  - Trigger detection, chime-in probability, and room eligibility checks.

- **`config/fleet_config.example.yaml`** — annotated example fleet configuration.

- **Setup wizard multi-bot flow** (`setup_wizard.py`):
  - Top-level choice: "Single bot" or "Multi-bot fleet".
  - Multi-bot flow walks through route-bot LLM selection, then iteratively
    configures child bots (each gets its own `bots/<name>/config/` directory).
  - `--add-bot` flag to append a new bot to an existing fleet.
  - `--remove-bot <name>` flag to remove a bot from the fleet.

- **New test suites** (39 tests across 3 files):
  - `tests/test_fleet_config.py` — loading, validation, mutation, save/load
    round-trip.
  - `tests/test_child_bot_context.py` — persona loading, triggers, chime-in
    logic, cooldowns, room eligibility.
  - `tests/test_orchestrator.py` — PendingMessage ordering, PriorityQueue FIFO
    guarantee, field defaults.

### Changed

- **`src/llm.py`** — `unload()` now performs a thorough cleanup: deletes model
  references, calls `gc.collect()`, and clears CUDA cache with
  `torch.cuda.empty_cache()` when available. Added `is_loaded` property.

- **`src/config_manager.py`** — `ConfigManager.__init__()` accepts an optional
  `route_bot_only=True` keyword argument that skips Matrix credential validation
  (the route-bot has an LLM config but no Matrix account).

- **`src/main.py`** — new `--multi` flag and auto-detection of
  `bots/fleet_config.yaml`. When detected, launches the orchestrator instead of
  the single-bot loop. Single-bot mode is completely unchanged.

- **`setup_wizard.py`** — refactored to share `collect_bot_config()` between
  single and multi-bot flows. Route-bot setup recommends smaller models
  (Phi-3 3.8B, Qwen2 1.5B, TinyLlama 1.1B).

- **`thoughts.py`** — accepts `--data-dir`, `--config`, and `--compact` CLI
  arguments via `argparse` so the orchestrator can run it per-bot with
  different data directories. Fully backwards-compatible (defaults to `data/`
  and `config/config.yaml`).

### Fixed

- **`src/orchestrator.py`** — replaced fragile `set`-based event deduplication
  (which evicted in arbitrary order) with `OrderedDict` for correct
  insertion-order (oldest-first) eviction.

- **`src/orchestrator.py`** — removed buggy/redundant passive-channel check in
  `_route_bot_classify()` that contained an incorrect boolean expression.
  Now uses the existing `ctx.passive_channels` property which already handles
  this correctly.
