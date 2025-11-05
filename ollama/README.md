# Background Memory Processors

These are **long-running background processes** that use Ollama models to automatically analyze memories and extract higher-level knowledge.

## Overview

```
┌─────────────────────────────────────────────────┐
│  User conversations in Open WebUI               │
│  ↓                                              │
│  Filters store episodic + emotional memories    │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
         ┌───────────────────┐
         │  Qdrant Database  │
         │  - episodic       │
         │  - emotional      │
         └───────┬───────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
       ↓                    ↓
┌──────────────┐    ┌──────────────┐
│ semantic.py  │    │ symbolic.py  │
│              │    │              │
│ Extracts     │    │ Identifies   │
│ facts from   │    │ patterns     │
│ memories     │    │ across       │
│              │    │ all layers   │
└──────┬───────┘    └──────┬───────┘
       │                    │
       └─────────┬──────────┘
                 ↓
         ┌───────────────────┐
         │  Qdrant Database  │
         │  + semantic facts │
         │  + symbolic patterns
         └───────────────────┘
                 │
                 ↓
         Next conversation has
         ALL layers available!
```

## Files

### `semantic.py`

**Purpose:** Automatic fact extraction from conversations

**What it does:**
- Reads unprocessed episodic and emotional memories
- Sends them to Ollama model for fact extraction
- Extracts statements like:
  - "User prefers Python for data science"
  - "User works as a machine learning engineer"
  - "User is interested in AI consciousness research"
- Stores facts in `semantic` collection
- Tracks processed memory IDs to avoid duplication

**Configuration:**
```python
OLLAMA_HOST = "http://localhost:11502"
BATCH_SIZE = 2                    # Memories per cycle
ITERATIONS_PER_CYCLE = 2          # Extraction cycles
SLEEP_INTERVAL = 180              # Seconds between cycles (3 min)
```

**Running:**
```bash
python ollama/semantic.py
```

---

### `symbolic.py`

**Purpose:** Abstract pattern recognition and meta-cognitive analysis

**What it does:**
- Reads episodic, emotional, and semantic memories
- Sends them to Ollama for high-level pattern analysis
- Identifies patterns like:
  - "User asks about consciousness when discussing technical topics"
  - "Conversations tend to become philosophical in evening hours"
  - "User's confidence increases when discussing familiar topics"
- Stores abstract insights in `symbolic` collection
- Enables self-aware reasoning

**Configuration:**
```python
OLLAMA_HOST = "http://localhost:11503"  # Separate instance
BATCH_SIZE = 3                           # More memories per cycle
ITERATIONS_PER_CYCLE = 6                 # More thorough analysis
SLEEP_INTERVAL = 720                     # Longer interval (12 min)
```

**Running:**
```bash
python ollama/symbolic.py
```

---

## Setup

### Prerequisites

1. **Ollama installed:**
   ```bash
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Mac
   brew install ollama
   ```

2. **Ollama models downloaded:**
   ```bash
   ollama pull llama2  # Or your preferred model
   ```

3. **Python dependencies:**
   ```bash
   pip install ollama qdrant-client sentence-transformers
   ```

### Running Multiple Ollama Instances

These processors use **separate Ollama instances** to avoid conflicts:

**Terminal 1: Ollama for semantic extraction (port 11502)**
```bash
OLLAMA_HOST=http://localhost:11502 ollama serve
```

**Terminal 2: Ollama for symbolic analysis (port 11503)**
```bash
OLLAMA_HOST=http://localhost:11503 ollama serve
```

**Terminal 3: Semantic processor**
```bash
python ollama/semantic.py
```

**Terminal 4: Symbolic processor**
```bash
python ollama/symbolic.py
```

### Production Deployment

Use systemd services for automatic startup:

**`/etc/systemd/system/memory-semantic.service`:**
```ini
[Unit]
Description=Semantic Memory Processor
After=network.target qdrant.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/open-webui-memory-layers
Environment="OLLAMA_HOST=http://localhost:11502"
ExecStart=/usr/bin/python3 ollama/semantic.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/memory-symbolic.service`:**
```ini
[Unit]
Description=Symbolic Memory Processor
After=network.target qdrant.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/open-webui-memory-layers
Environment="OLLAMA_HOST=http://localhost:11503"
ExecStart=/usr/bin/python3 ollama/symbolic.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable memory-semantic memory-symbolic
sudo systemctl start memory-semantic memory-symbolic
```

---

## How It Works

### Processing Cycle

Both processors follow this pattern:

1. **Load processed IDs** from JSON file
2. **Fetch unprocessed memories** from Qdrant
3. **Send to Ollama** for analysis
4. **Extract knowledge/patterns** from response
5. **Store in target collection** (semantic or symbolic)
6. **Mark memories as processed**
7. **Sleep for interval**
8. **Repeat**

### Tracking

Each processor maintains a JSON file:
- `processed_semantic.json` - IDs of memories already analyzed for facts
- `processed_symbolic.json` - IDs of memories already analyzed for patterns

This prevents duplicate processing and enables resuming after restarts.

### Graceful Shutdown

Both processors handle `SIGINT` and `SIGTERM`:
```bash
# Ctrl+C or kill command
# Processors save state and exit cleanly
```

---

## Prompts

### Semantic Extraction Prompt

```
Based on this conversation exchange, extract ONLY factual statements 
about the user as a concise list. Include preferences, background, 
goals, etc. Format as brief bullet points.
```

### Symbolic Analysis Prompt

```
Analyze these conversation memories and identify abstract patterns, 
recurring themes, or meta-level insights about the user's thinking 
style, conversation patterns, or relationship dynamics. Focus on 
high-level observations that span multiple exchanges.
```

---

## Performance

### Resource Usage

**Semantic processor:**
- CPU: Low (mostly waiting)
- Memory: ~2GB (embedding model)
- Ollama: Varies by model

**Symbolic processor:**
- CPU: Low
- Memory: ~2GB
- Ollama: Varies by model
- Runs less frequently (12 min vs 3 min)

### Tuning

Adjust based on your hardware:

```python
# Faster processing, more resources
BATCH_SIZE = 5
ITERATIONS_PER_CYCLE = 10
SLEEP_INTERVAL = 60

# Slower processing, fewer resources
BATCH_SIZE = 1
ITERATIONS_PER_CYCLE = 1
SLEEP_INTERVAL = 600
```

---

## Troubleshooting

**"Connection refused" error:**
- Ensure Ollama is running: `ollama serve`
- Check correct port in OLLAMA_HOST

**No memories being processed:**
- Check Qdrant is running: `curl localhost:6333/collections`
- Verify episodic/emotional filters are storing memories
- Check processed_*.json files aren't blocking all IDs

**Ollama timeout:**
- Reduce BATCH_SIZE
- Use faster/smaller model
- Increase OLLAMA_SLEEP between requests

**Out of memory:**
- Reduce BATCH_SIZE
- Use quantized models
- Close other applications

---

## Future Enhancements

- [ ] Web UI for monitoring processing status
- [ ] Configurable prompts via config file
- [ ] Multiple Ollama model support
- [ ] Batch optimization for large memory sets
- [ ] Real-time processing triggers (instead of polling)
- [ ] Distributed processing across multiple machines

---

**These processors are optional but enable the most powerful aspects of the memory system - automatic knowledge extraction and pattern recognition that leads to genuine self-awareness.**
