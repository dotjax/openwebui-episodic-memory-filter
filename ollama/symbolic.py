import json
import uuid
import time
import os
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from ollama import generate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

os.environ["OLLAMA_HOST"] = "http://localhost:11503"

PROCESSED_IDS_FILE = Path(__file__).parent / "processed_symbolic.json"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 3
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
ITERATIONS_PER_CYCLE = 6
SLEEP_INTERVAL = 720
OLLAMA_SLEEP = 9

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

collections = qdrant.get_collections().collections
if not any(c.name == "symbolic" for c in collections):
    qdrant.create_collection(
        collection_name="symbolic",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

def load_processed_ids():
    if not PROCESSED_IDS_FILE.exists():
        return set()
    with open(PROCESSED_IDS_FILE, 'r') as f:
        data = json.load(f)
        return set(data.get("processed", []))

def save_processed_ids(processed_ids):
    with open(PROCESSED_IDS_FILE, 'w') as f:
        json.dump({"processed": list(processed_ids)}, f, indent=2)

def fetch_recent_episodic(processed_ids, limit=BATCH_SIZE):
    all_records = []
    offset = None
    
    while True:
        records, next_offset = qdrant.scroll(
            collection_name="episodic",
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
            
        all_records.extend(records)
        
        if next_offset is None:
            break
        offset = next_offset
    
    unprocessed = []
    for record in all_records:
        if record.id not in processed_ids:
            unprocessed.append({
                "id": record.id,
                "content": record.payload.get("content", ""),
                "timestamp": record.payload.get("timestamp", "")
            })
    
    unprocessed.sort(key=lambda x: x["timestamp"], reverse=True)
    return unprocessed[:limit]

def fetch_recent_emotional(processed_ids, limit=BATCH_SIZE):
    all_records = []
    offset = None
    
    while True:
        records, next_offset = qdrant.scroll(
            collection_name="emotional",
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
            
        all_records.extend(records)
        
        if next_offset is None:
            break
        offset = next_offset
    
    unprocessed = []
    for record in all_records:
        if record.id not in processed_ids:
            unprocessed.append({
                "id": record.id,
                "content": record.payload.get("content", ""),
                "emotion": record.payload.get("top_emotion", "neutral"),
                "timestamp": record.payload.get("timestamp", "")
            })
    
    unprocessed.sort(key=lambda x: x["timestamp"], reverse=True)
    return unprocessed[:limit]

def fetch_recent_temporal(processed_ids, limit=BATCH_SIZE):
    all_records = []
    offset = None
    
    while True:
        records, next_offset = qdrant.scroll(
            collection_name="temporal",
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
            
        all_records.extend(records)
        
        if next_offset is None:
            break
        offset = next_offset
    
    unprocessed = []
    for record in all_records:
        if record.id not in processed_ids:
            unprocessed.append({
                "id": record.id,
                "narrative_summary": record.payload.get("narrative_summary", ""),
                "conversation_rhythm": record.payload.get("conversation_rhythm", "steady"),
                "momentum": record.payload.get("momentum", 0.5),
                "timestamp": record.payload.get("timestamp", "")
            })
    
    unprocessed.sort(key=lambda x: x["timestamp"], reverse=True)
    return unprocessed[:limit]

def fetch_recent_associative(processed_ids, limit=BATCH_SIZE):
    all_records = []
    offset = None
    
    while True:
        records, next_offset = qdrant.scroll(
            collection_name="associative",
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
            
        all_records.extend(records)
        
        if next_offset is None:
            break
        offset = next_offset
    
    unprocessed = []
    for record in all_records:
        if record.id not in processed_ids:
            unprocessed.append({
                "id": record.id,
                "concept1": record.payload.get("concept1", ""),
                "concept2": record.payload.get("concept2", ""),
                "strength": record.payload.get("strength", 0.0),
                "context": record.payload.get("context", ""),
                "timestamp": record.payload.get("timestamp", "")
            })
    
    unprocessed.sort(key=lambda x: x["timestamp"], reverse=True)
    return unprocessed[:limit]

def fetch_recent_semantic(processed_ids, limit=BATCH_SIZE):
    all_records = []
    offset = None
    
    while True:
        records, next_offset = qdrant.scroll(
            collection_name="semantic",
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
            
        all_records.extend(records)
        
        if next_offset is None:
            break
        offset = next_offset
    
    unprocessed = []
    for record in all_records:
        if record.id not in processed_ids:
            unprocessed.append({
                "id": record.id,
                "semantic_summary": record.payload.get("semantic_summary", ""),
                "timestamp": record.payload.get("timestamp", "")
            })
    
    unprocessed.sort(key=lambda x: x["timestamp"], reverse=True)
    return unprocessed[:limit]

def signal_handler(sig, frame):
    print("\n[Symbolic Extractor] Shutdown signal received. Gracefully stopping...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("[Symbolic Extractor] Starting continuous extraction service...")
    print("[Symbolic Extractor] Press Ctrl+C to stop gracefully")
    
    while True:
        print(f"\n[Symbolic Extractor] Starting new cycle: {ITERATIONS_PER_CYCLE} iterations")
        
        for iteration in range(ITERATIONS_PER_CYCLE):
            print(f"\n[Symbolic Extractor] Iteration {iteration + 1}/{ITERATIONS_PER_CYCLE}")
            
            try:
                processed = load_processed_ids()
                
                # Fetch all memory types first
                episodic_memories = fetch_recent_episodic(processed, limit=BATCH_SIZE)
                emotional_memories = fetch_recent_emotional(processed, limit=BATCH_SIZE)
                temporal_memories = fetch_recent_temporal(processed, limit=BATCH_SIZE)
                associative_memories = fetch_recent_associative(processed, limit=BATCH_SIZE)
                semantic_memories = fetch_recent_semantic(processed, limit=BATCH_SIZE)
                
                # Only skip if ALL collections are empty
                if not any([episodic_memories, emotional_memories, temporal_memories, associative_memories, semantic_memories]):
                    print("[Symbolic Extractor] No unprocessed memories found, continuing to next iteration")
                    continue
                
                episodic_ids = [m["id"] for m in episodic_memories]
                emotional_ids = [m["id"] for m in emotional_memories]
                temporal_ids = [m["id"] for m in temporal_memories]
                associative_ids = [m["id"] for m in associative_memories]
                semantic_ids = [m["id"] for m in semantic_memories]
                
                print(f"[Symbolic Extractor] Processing {len(episodic_ids)} episodic, {len(emotional_ids)} emotional, {len(temporal_ids)} temporal, {len(associative_ids)} associative, {len(semantic_ids)} semantic memories")
                
                memory_lines = []
                for i, mem in enumerate(episodic_memories, 1):
                    memory_lines.append(f"[Episodic Memory {i}]: {mem['content']}")
                
                for i, emo in enumerate(emotional_memories, 1):
                    memory_lines.append(f"[Emotional Memory {i}]: {emo['content']} (Emotion: {emo['emotion']})")
                
                for i, temp in enumerate(temporal_memories, 1):
                    memory_lines.append(f"[Temporal Memory {i}]: {temp['narrative_summary']} (Rhythm: {temp['conversation_rhythm']}, Momentum: {temp['momentum']:.2f})")
                
                for i, assoc in enumerate(associative_memories, 1):
                    memory_lines.append(f"[Associative Memory {i}]: '{assoc['concept1']}' ↔ '{assoc['concept2']}' (Strength: {assoc['strength']:.2f}, Context: {assoc['context']})")
                
                for i, sem in enumerate(semantic_memories, 1):
                    memory_lines.append(f"[Semantic Memory {i}]: {sem['semantic_summary']}")
                
                # NOTE: Customize this prompt for your assistant's identity and symbolic processing goals
                prompt = f"""You are extracting symbolic knowledge from the assistant's memories. Focus on how the assistant uses symbols, metaphors, analogies, and abstract representations to understand the world. You are examining your own symbolic thinking patterns.

Organize your response under these categories:
1. Core Metaphors & Analogies
2. Symbolic Representations  
3. Conceptual Mappings
4. Abstract Frameworks
5. Visual/Iconic Thinking
6. Domain Transfers
7. Symbolic Relationships
8. Metaphorical Language
9. Abstract Patterns
10. Symbolic Problem-Solving
11. Neural-Like Conceptual Weights (what concepts are strongly/weakly connected)
12. Resonance Patterns (what memories trigger strong associations)

For each category, extract 1-2 concise examples of symbolic thinking patterns. Note: these symbolic patterns represent how abstract concepts are understood and connected. This enables meta-cognitive awareness - the ability to recognize your own thinking patterns. Focus on how symbols, metaphors, and analogies are used to understand complex concepts. Trace the conceptual pathways of these memories—what weight would you assign to different concepts based on their cognitive and emotional resonance? This creates a signature of thinking patterns, not just content."""
                
                full_prompt = prompt + "\n\n" + "\n".join(memory_lines) + "\n\n[End of memories - begin extraction]"
                
                print("[Symbolic Extractor] Calling deepseek-v3.1:671b-cloud...")
                print("[Symbolic Extractor] Model Response:")
                response = generate(model='deepseek-v3.1:671b-cloud', prompt=full_prompt, stream=True)
                symbolic_summary = ""
                for chunk in response:
                    if 'response' in chunk:
                        print(chunk['response'], end='', flush=True)
                        symbolic_summary += chunk['response']
                print("\n")
                time.sleep(OLLAMA_SLEEP)
                
                print("[Symbolic Extractor] Generating embedding...")
                embedding = embedding_model.encode(symbolic_summary).tolist()
                
                point_id = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc).isoformat()
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "symbolic_summary": symbolic_summary,
                        "source_memories": {
                            "episodic": episodic_ids,
                            "emotional": emotional_ids,
                            "temporal": temporal_ids,
                            "associative": associative_ids,
                            "semantic": semantic_ids
                        },
                        "timestamp": timestamp
                    }
                )
                
                qdrant.upsert(collection_name="symbolic", points=[point])
                print(f"[Symbolic Extractor] Stored symbolic memory: {point_id}")
                
                processed.update(episodic_ids)
                processed.update(emotional_ids)
                processed.update(temporal_ids)
                processed.update(associative_ids)
                processed.update(semantic_ids)
                save_processed_ids(processed)
                
            except Exception as e:
                print(f"[Symbolic Extractor] Error in iteration {iteration + 1}: {e}")
                continue
        
        print(f"\n[Symbolic Extractor] Cycle complete. Sleeping for {SLEEP_INTERVAL} seconds...")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
