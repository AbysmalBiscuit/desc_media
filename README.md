# desc-media

A Python CLI tool that uses a local [LLaVA](https://ollama.com/library/llava) vision model (via [Ollama](https://ollama.com)) to generate keyword/tag descriptions for image and video files.

## How it works

- **Images** are sent directly to LLaVA, which returns 10–50 comma-separated keywords/tags.
- **Videos** are split into keyframes with `ffmpeg`, then keyframes are sent to LLaVA in sliding-window batches. Keywords/Tags are aggregated across all batches using a frequency counter.
- Results are either logged to the console (`describe`) or persisted to a JSON file (`save`).

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com) running locally on `http://127.0.0.1:11434` with the `llava` model pulled
- [ffmpeg](https://ffmpeg.org) available on `PATH` (required for video processing)
- [fd](https://github.com/sharkdp/fd) available on `PATH` (used for fast recursive file discovery)

### Pull the LLaVA model

```bash
ollama pull llava
```

### Serve the LLaVA model

```sh
ollama serve
```

## Installation

Install with [uv](https://github.com/astral-sh/uv):

```bash
# Install directly from GitHub
uv tool install git+https://www.github.com/AbysmalBiscuit/desc_media

# By cloning the repo first
git clone https://www.github.com/AbysmalBiscuit/desc_media
cd desc_media
uv tool install .
```

Or install into a virtual environment:

```bash
git clone https://www.github.com/AbysmalBiscuit/desc_media
cd desc_media
uv sync
source .venv/bin/activate
```

This registers the `descmedia` command.

## Usage

```
descmedia [OPTIONS] COMMAND [ARGS]...
```

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--verbose` | — | Increase log verbosity (repeatable: `-vv`, `-vvv`) |
| `-q` / `--quiet` | — | Decrease log verbosity (repeatable) |
| `-a` / `--address TEXT` | `127.0.0.1` | IP address of the Ollama server |
| `-p` / `--port INT` | `11434` | Port of the Ollama server |
| `-r` / `--protocol {http,https}` | `http` | Protocol used to connect to Ollama |
| `--version` | — | Show version and exit |

To connect to a remote Ollama instance:

```bash
descmedia -a 192.168.1.50 -p 11434 describe ~/Photos/
descmedia --protocol https --address ollama.example.com save ~/Photos/ out.json
```

---

### `describe`: print keywords/tags to the console

```
descmedia describe [OPTIONS] PATH
```

Processes a single file or all media files found recursively under `PATH` and logs the resulting keywords. Output is not saved to disk.

| Option | Default | Description |
|--------|---------|-------------|
| `-e` / `--extra TEXT` |   | Extra text appended to the model prompt |
| `-vbs` / `--video-batch-size INT` | `5` | Keyframes per batch for video processing |

**Examples**

```bash
# Describe a single image
descmedia describe photo.jpg

# Describe all media in a folder
descmedia describe ~/Photos/2024/

# Add context to the prompt
descmedia describe -e "These are product photos for an online store." ~/catalog/
```

---

### `save`: describe and persist results to JSON

```
descmedia save [OPTIONS] PATH DESCFILE
```

Same as `describe`, but writes results to `DESCFILE` as a JSON object mapping absolute file paths to keyword/tag lists. A log file (`descmedia.log`) is written alongside `DESCFILE`. Progress is checkpointed every 50 files.

| Option | Default | Description |
|--------|---------|-------------|
| `-e` / `--extra TEXT` |   | Extra text appended to the model prompt |
| `-vbs` / `--video-batch-size INT` | `5` | Keyframes per batch for video processing |
| `-t` / `--timeout INT` | `5` | Ollama client timeout in seconds |
| `-i` / `--incremental` | `False` | Skip files already present in `DESCFILE` |

**Examples**

```bash
# Describe a folder and save results
descmedia save ~/Photos/ descriptions.json

# Resume an interrupted run (skip already-described files)
descmedia save --incremental ~/Photos/ descriptions.json

# Save to a directory (creates descmedia.json inside it)
descmedia save ~/Photos/ ~/output/

# Increase timeout for slow hardware
descmedia save -t 30 ~/Videos/ descriptions.json
```

### Output format

`DESCFILE` is a JSON object where each key is an absolute file path and each value is a list of keywords/tags sorted by frequency:

```json
{
  "/home/user/Photos/beach.jpg": ["ocean", "sunset", "waves", "sand", "blue sky"],
  "/home/user/Videos/holiday.mp4": ["family", "outdoor", "celebration", "garden"]
}
```

## License

See [LICENSE](LICENSE).
