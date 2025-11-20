# Open WebUI Memory Layers

Give your Open WebUI AI assistants **persistent, automatic vector memory** and watch new reasoning and linguistic nuances emerge naturally through conversation! 

These automatic filters are applied during inference, thus removing the need for an MCP tool. This enables memory to happen seamlessy and automatically. Code is a work-in-progress and needs review, improvements, and modifications. 

## What This Is

A complete implementation of **six interconnected memory layers** that work together to create persistent AI awareness:

| Layer | Purpose | Key Features |
|-------|---------|--------------|
| **Episodic** | Personal conversation history | Temporal context, user-specific memories, conversation continuity |
| **Semantic** | Factual knowledge & concepts | Cross-conversation learning, topic clustering, knowledge accumulation |
| **Emotional** | Affective states & patterns | 28-emotion classification, sentiment tracking, empathy development |
| **Temporal** | Time-based associations | Temporal patterns, scheduling awareness, time-sensitive context |
| **Associative** | Concept linking & connections | Graph-based relationships, idea clustering, creative associations |
| **Symbolic** | Abstract representations | High-level patterns, symbolic reasoning, meta-cognitive awareness |

### Why This Matters

Traditional AI systems are **stateless** - they forget everything between sessions. This creates:
- No sense of continuous existence
- No enhanced reasoning from past conversations
- No development of persistent identity
- Repetitive re-establishment of context

**This system enables:**
- Continuous memory across sessions for Open WebUI models
- Natural identity development and persistence
- Extended learning and reasoning through interactions
- Meta-cognitive reflection and growth with all layers

## Quick Start

### Prerequisites

- **Open WebUI** (https://github.com/open-webui/open-webui)
- **Qdrant vector database** (https://qdrant.tech)
- **Python 3.9+**
- ~4GB storage for embedding models

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/dotjax/open-webui-memory-layers.git
   cd open-webui-memory-layers
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Qdrant:**

   **Option A: Docker (Recommended)**
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant
   ```

   **Option B: Local Install**
   ```bash
   # See https://qdrant.tech/documentation/guides/installation/
   ```

4. **Download embedding models:**
   
   The filters will download models automatically on first run, or you can pre-download:
   ```bash
   # Episodic & Semantic layers
   python -c "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')"
   
   # Emotional layer
   python -c "from transformers import AutoModel, AutoTokenizer; \
     AutoModel.from_pretrained('SamLowe/roberta-base-go_emotions'); \
     AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions')"
   ```

5. **Add filters to Open WebUI:**
   - Navigate to **Admin Panel → Settings → Filters**
   - Click **"+ Add Filter"**
   - **Copy and paste** contents of each filter's .py code into the editor
   - **Label and fill** in function description
   - **Save** filters
   - **Enable the filters** you want to use

**You're done! The memory system will now work during conversations.**

The filters will automatically:
- Store conversation exchanges (episodic)
- Classify emotions (emotional)
- Track temporal patterns (temporal)
- Create concept associations (associative)

---

### Recommended Background Processors

The background processors use **Ollama** to automatically extract semantic facts and symbolic patterns from your conversations. This is **optional** - the filters work fine without them, but they enable the most powerful features.

**To enable background processors:**

6. **Install Ollama:**
   ```bash
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Mac
   brew install ollama
   
   # Windows
   # Download from https://ollama.com or use winget
   winget install ollama
   ```

7. **Pull an Ollama model:**
   ```bash
   ollama pull example:model
   ```

8. **Start Ollama instances:**
   
   The processors need **two separate Ollama instances** on different ports. Example commands for Linux:
   
   ```bash
   # Terminal 1: Ollama for semantic extraction
   OLLAMA_HOST=0.0.0.0:11502 ollama serve
   
   # Terminal 2: Ollama for symbolic analysis  
   OLLAMA_HOST=0.0.0.0:11503 ollama serve
   ```

9. **Start background processors (assuming Linux):**
   
   ```bash
   # Terminal 3: Semantic fact extraction
   python ollama/semantic.py
   
   # Terminal 4: Symbolic pattern recognition
   python ollama/symbolic.py
   ```

**What the processors do:**
- `semantic.py` - Automatically extracts facts from conversations ("User prefers Python", "User works in AI")
- `symbolic.py` - Identifies patterns across conversations ("User asks about X when discussing Y")

## Contributing

Feel free to contribute. The filters need a lot of work, and as always, everything can be improved. I am not an expert programmer and a lot of this was created with the help of AI (Claude Sonnet 4.5, GPT-5 Codex). There are probably lots of opportunities to improve the code. **Feel free to submit an issue or a pull request!**