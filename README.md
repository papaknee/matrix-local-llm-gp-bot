# Matrix Local LLM Bot 🤖

A self-hosted, personality-rich Matrix chat bot that runs a local Large Language Model (LLM) entirely on **your own hardware** — no cloud API keys required. The bot knows who it's talking to, has mood swings, and occasionally barges into conversations uninvited like a real community member.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Pre-requisites](#pre-requisites)
3. [Quick Start](#quick-start)
4. [Choosing a Model](#choosing-a-model)
5. [Project Structure](#project-structure)
6. [Configuration Reference](#configuration-reference)
7. [Customising the Bot's Personality](#customising-the-bots-personality)
8. [User Dossiers — How the Bot Remembers People](#user-dossiers--how-the-bot-remembers-people)
9. [Mood Swings — The Temperature Controller](#mood-swings--the-temperature-controller)
10. [Running the Bot](#running-the-bot)
11. [Troubleshooting](#troubleshooting)
12. [Ideas to Make the Bot Even Better](#ideas-to-make-the-bot-even-better)
13. [Security Notes](#security-notes)

---

## What It Does

- **Local LLM inference** — runs a Hugging Face or GGUF model on your machine. Zero data sent to third parties.
- **Matrix integration** — connects to any Matrix homeserver (Element, Synapse, Conduit, etc.).
- **Configurable rooms** — specify exactly which channels the bot watches and posts in.
- **User memory (dossiers)** — maintains a per-user JSON file recording who it has spoken to, what they said, and a growing personality sketch. Memory is automatically compacted so it never blows the model's context window.
- **Mood swings** — the bot's LLM temperature (creativity/feistiness dial) changes randomly on a schedule. One minute it's calm and helpful; thirty minutes later it's chaotic and opinionated.
- **Spontaneous chime-ins** — configurable probability that the bot inserts itself into a conversation even when not directly addressed.
- **Trigger names** — the bot *always* responds when its name or configured nicknames appear in a message.
- **Interactive setup wizard** — no code editing required to get started.

---

## Pre-requisites

| Requirement | Notes |
|---|---|
| A Matrix homeserver | Self-hosted (Synapse/Conduit) or a public server like `matrix.org` |
| A bot Matrix account | Create a separate account for the bot — don't use your own |
| Python 3.10 or newer | Check with `python --version` |
| 4 GB+ RAM (CPU mode) | More is better; see [model table](#choosing-a-model) |
| NVIDIA GPU (optional) | Required only for GPU mode; dramatically faster |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/papaknee/matrix-local-llm-gp-bot.git
cd matrix-local-llm-gp-bot
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (NVIDIA):** After the above, reinstall llama-cpp-python with CUDA support:
> ```bash
> CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
> ```
>
> **Apple Silicon (M1/M2/M3):** Metal (GPU) is enabled by default — no extra steps needed.

### 4. Run the setup wizard

```bash
python setup_wizard.py
```

The wizard will ask you for:
- Your Matrix server URL, bot username, and password
- Which rooms the bot should listen in
- Which model to download (or where your model already is)
- Whether to use CPU or GPU
- How often the bot should spontaneously join conversations

It will then write `config/config.yaml` and optionally download your chosen model.

### 5. Start the bot

```bash
python -m src.main
```

That's it! The bot will log in, join its allowed rooms, and start listening.

---

## Choosing a Model

All models below work with the **llama-cpp** backend (GGUF format). They are quantised to use much less memory than full-precision models.

| Label | Model | File Size | Min RAM | Best For |
|---|---|---|---|---|
| 🟢 Tiny | [Phi-3 Mini 3.8B Q4](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | ~2.2 GB | 4 GB | Old laptops, Raspberry Pi 5, anything with 4 GB RAM |
| 🟡 Small *(recommended)* | [Mistral 7B Instruct Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | ~4.8 GB | 6 GB | Most modern laptops and desktops |
| 🟠 Medium | [LLaMA-3 8B Instruct Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) | ~5.5 GB | 8 GB | Desktop with 8 GB RAM or GPU with 6 GB VRAM |
| 🔴 Large | [Mistral 7B Q8_0](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | ~7.7 GB | 10 GB | Machines with 16 GB RAM or GPU with 8 GB VRAM |
| 🚀 Enthusiast | [Mixtral 8x7B Q4_K_M](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF) | ~26 GB | 32 GB | Workstation / server with lots of RAM |

**Rule of thumb:** Pick the largest model your machine can fit in RAM with a few GB to spare for the OS and other processes.

### GPU mode

Set `hardware_mode: gpu` in `config/config.yaml` and adjust `n_gpu_layers`:
- Start with `n_gpu_layers: 20` and increase until you run out of VRAM.
- Set to `-1` to offload everything to the GPU (fastest, but requires enough VRAM for the whole model).

A GPU with **6 GB VRAM** can run the Mistral 7B Q4 model entirely on-GPU at ~50 tokens/second — much more responsive than CPU.

---

## Project Structure

```
matrix-local-llm-gp-bot/
├── config/
│   ├── config.example.yaml    ← Template — copy to config.yaml
│   └── bot_persona.txt        ← The bot's personality instructions
├── data/
│   ├── dossiers/              ← Per-user memory files (auto-created)
│   ├── archives/              ← Older memory snapshots (auto-created)
│   └── matrix_store/          ← Matrix sync state (auto-created)
├── models/                    ← All large model files live here
│   ├── *.gguf                 ← GGUF model files (llama-cpp backend)
│   └── hf_cache/             ← HuggingFace model downloads (transformers backend)
├── src/
│   ├── main.py                ← Entry point
│   ├── bot.py                 ← Core logic (trigger detection, prompt building)
│   ├── config_manager.py      ← Config loader & validator
│   ├── llm.py                 ← LLM backend (llama-cpp / transformers)
│   ├── matrix_client.py       ← Matrix-nio wrapper
│   ├── memory_manager.py      ← User dossier read/write/compact
│   ├── temperature_controller.py ← Mood/temperature scheduling
│   └── scheduler.py           ← Background async jobs
├── tests/                     ← Unit tests
├── setup_wizard.py            ← Interactive first-run setup
└── requirements.txt
```

> **All large files stay inside the repo directory.** GGUF model files are
> saved to `models/`, and HuggingFace model downloads (transformers backend)
> are cached in `models/hf_cache/`. The Python virtual environment is created
> as `.venv/` inside the repo (see [Quick Start](#quick-start)). Every
> large artifact is therefore co-located with the project — no hunting through
> `~/.cache/huggingface/` or other hidden directories when you want to free
> up disk space. The entire `models/` tree and `.venv/` are already listed in
> `.gitignore` so they are never accidentally committed.

---

## Configuration Reference

Copy `config/config.example.yaml` to `config/config.yaml` and edit it. Every option has comments explaining what it does.

### Key settings at a glance

| Section | Key | What it does |
|---|---|---|
| `matrix` | `homeserver` | URL of your Matrix server |
| `matrix` | `allowed_rooms` | List of room aliases/IDs to watch. Empty = all rooms. |
| `matrix` | `passive_channels` | List of room aliases/IDs where the bot only responds passively (e.g., when mentioned or based on probability). |
| `llm` | `backend` | `llamacpp` (GGUF) or `transformers` (HuggingFace) |
| `llm` | `hardware_mode` | `cpu` or `gpu` |
| `llm` | `n_gpu_layers` | How many layers to offload to GPU (0 = CPU only) |
| `bot` | `trigger_names` | Words that always trigger a response |
| `bot` | `chime_in_probability` | 0.0–1.0 chance of spontaneous reply |
| `bot` | `chime_in_cooldown_seconds` | Min gap between unsolicited posts |
| `temperature_controller` | `change_interval_minutes` | How often to roll a new mood |
| `temperature_controller` | `min_temperature` / `max_temperature` | Mood swing range |
| `memory` | `max_active_entries` | Entries per user before compaction |
| `memory` | `compaction_interval_hours` | How often the compaction job runs |

---

### Passive Channels

You can configure `passive_channels` under the `matrix:` section in your config file. In these rooms, the bot will only respond passively (for example, when mentioned or based on its chime-in probability), rather than actively replying to all messages. This is useful for rooms where you want the bot to be less intrusive.

Example:

```yaml
matrix:
  ...
  allowed_rooms:
    - "#general:matrix.example.org"
    - "#random:matrix.example.org"
  passive_channels:
    - "#lurkers:matrix.example.org"
    - "!passiveRoomId:matrix.example.org"
```

---

## Customising the Bot's Personality

Open `config/bot_persona.txt` and rewrite it however you like. This file is loaded as the **system prompt** for every LLM call, so it defines everything about how the bot behaves.

Tips:
- Be specific. "You are a sarcastic film critic who secretly loves romantic comedies" produces better results than "be funny."
- Include example exchanges — models respond well to few-shot prompting.
- Tell the bot what it should *not* do, not just what it should do.
- Keep it under ~800 characters to leave room for the conversation and dossier context.

---

## User Dossiers — How the Bot Remembers People

Every time the bot interacts with someone, it appends a record to `data/dossiers/<username>.json`:

```json
{
  "user_id": "@alice:example.org",
  "display_name": "Alice",
  "first_seen": "2025-01-15T14:23:01+00:00",
  "last_seen": "2025-03-20T09:11:44+00:00",
  "summary": "Alice is a Python developer who loves cats and hates Mondays...",
  "entries": [
    {
      "timestamp": "2025-03-20T09:11:44+00:00",
      "room_id": "#general:example.org",
      "user_message": "ugh I hate Mondays",
      "bot_response": "Classic Alice. Should I put 'hates Mondays' in your permanent record?"
    }
  ]
}
```

### Memory compaction

When the number of `entries` exceeds `max_active_entries` (default: 50), the oldest half is automatically compressed into the `summary` text and a full snapshot is saved to `data/archives/`. This keeps the dossier small enough to fit in the LLM's context window.

You can edit dossier files manually — they're just JSON. Add notes like `"notes": "loves spicy food, afraid of birds"` inside any entry and the bot will see it next time it talks to that user.

---

## Mood Swings — The Temperature Controller

The **temperature** is the main creativity dial for LLMs:

| Temperature | Behaviour |
|---|---|
| 0.1–0.3 | Very focused, almost robotic |
| 0.4–0.6 | Calm and sensible |
| 0.7–0.9 | Creative and conversational (default sweet spot) |
| 1.0–1.1 | Opinionated and a bit wild |
| 1.1–1.2 | Chaotic, surprising, occasionally unhinged |

Every `change_interval_minutes` minutes (default: 30), the bot picks a random temperature within your configured range and logs the new mood:

```
🎭 Mood swing! 0.650 → 1.087 (feisty) | chime_in=0.14
```

The chime-in probability also changes with each mood swing (if `randomise_chime_in: true`), so on a feisty day the bot will butt into conversations more aggressively.

---

## Running the Bot

### Foreground (development)

```bash
python -m src.main
```

### As a background service (Linux — systemd)

Create `/etc/systemd/system/matrix-bot.service`:

```ini
[Unit]
Description=Matrix Local LLM Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/matrix-local-llm-gp-bot
ExecStart=/path/to/matrix-local-llm-gp-bot/.venv/bin/python -m src.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now matrix-bot
sudo journalctl -fu matrix-bot    # watch logs
```

### As a Docker container

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "src.main"]
```

```bash
docker build -t matrix-llm-bot .
docker run -d \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name matrix-bot \
  matrix-llm-bot
```

---

## Troubleshooting

### "Config file not found"

Run the setup wizard: `python setup_wizard.py`

### "Matrix login failed"

- Double-check your homeserver URL (include `https://`)
- Verify username format: `@botname:yourdomain.org`
- Make sure the account exists and the password is correct
- If the server uses SSO, you'll need to create a password for the bot account in Element settings

### The bot is not responding

1. Check `data/bot.log` for errors
2. Verify the room is listed in `allowed_rooms` (or leave the list empty to allow all rooms)
3. Make sure the bot account has been invited to / joined the room
4. Confirm `trigger_names` includes the name you're using to call the bot

### The bot is too slow

- Switch to GPU mode if you have an NVIDIA card
- Use a smaller/more quantised model (Q4 vs Q8)
- Lower `max_new_tokens` (e.g. 256 instead of 512) for shorter responses
- Increase `chime_in_cooldown_seconds` so the bot posts less often

### Memory compaction not running

Make sure the bot is left running continuously — the compaction job fires on the interval you've configured, so it needs to be awake long enough to trigger.

---

## Ideas to Make the Bot Even Better

Here are some features you could add to extend the bot:

### 🎭 Persona switching
Give the bot multiple persona files and let users vote on which personality to activate: `!bot persona grumpy` or `!bot persona helpful`.

### 📊 User stats command
Add a `!dossier` command that lets a user see what the bot remembers about them, and a `!forget me` command to delete their dossier.

### 🗓️ Scheduled announcements
Use the scheduler to post daily messages — "Good morning!", trivia of the day, weather summaries (via a local API), or a random fun fact.

### 🧠 Shared room memory
In addition to per-user dossiers, maintain a per-room "topic log" that summarises recent discussions. The bot can reference it to stay on-topic.

### 🎲 Random events
Occasionally have the bot post an unprompted observation — "Just had a weird thought: [LLM-generated musing]" — at random intervals to feel more alive.

### 🔊 Voice channel summaries
If your Matrix setup includes Element Call, transcribe audio (via Whisper) and have the bot summarise the meeting.

### 📰 RSS feed reader
Have the bot monitor RSS feeds and post summaries of interesting articles to designated rooms, with its own commentary.

### 🔗 Web search
Add a tool call layer (function calling) that lets the bot decide to do a local web search (via SearXNG or DuckDuckGo) before answering factual questions.

### 🏆 Points and reputation
Track how many times users interact with the bot and give out titles: "Conversation Enthusiast", "Bot Whisperer", "Asking For a Friend".

### 🕹️ Text games
Implement a dungeon crawler or trivia game that runs inside a Matrix room, with the bot as the game master.

### 🌡️ Manual mood control
Add room commands like `!bot calm` or `!bot unhinged` to manually set the temperature and let room members influence the bot's vibe.

### 🧪 A/B persona testing
Run two bot accounts with different personas in the same room and let users rate their responses to discover which personality your community prefers.

---

## Security Notes

- **Never commit `config/config.yaml`** — it contains your Matrix password. It is already in `.gitignore`.
- The dossier files in `data/dossiers/` contain conversation history. Keep the `data/` directory private.
- The bot does not enable Matrix end-to-end encryption by default. If your rooms require E2E, set `encryption_enabled: True` in the MatrixClient config and install `matrix-nio[e2e]`.
- Run the bot as a dedicated low-privilege OS user, not as root.

---

*Built with [matrix-nio](https://github.com/poljar/matrix-nio) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Personality sold separately.*