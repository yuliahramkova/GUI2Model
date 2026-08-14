import json
import asyncio
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import os
import pickle

class SimpleRAG:
    def __init__(self, working_dir="./rag_storage_crawls"):
        self.working_dir = working_dir
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def initialize_storages(self):
        storage_dir = Path(self.working_dir)
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
            print("Cleared old storage directory")
        os.makedirs(self.working_dir, exist_ok=True)
        
        data_file = Path(self.working_dir) / "rag_data.pkl"
        if data_file.exists():
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = data['embeddings']
                print(f"Loaded {len(self.documents)} documents from disk")
            except:
                print("Could not load saved data")
        print("Storage initialized")
        
    async def ainsert(self, docs):
        if isinstance(docs, str):
            docs = [docs]
        
        for doc in docs:
            self.documents.append(doc)
            embedding = self.model.encode([doc], convert_to_numpy=True)[0]
            self.embeddings.append(embedding)

        data_file = Path(self.working_dir) / "rag_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Added {len(docs)} documents. Total: {len(self.documents)}")
        
    async def aquery(self, query):
        if not self.documents:
            return "No documents in database"
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.15:
                results.append({
                    'document': self.documents[idx],
                    'score': float(similarities[idx])
                })
        
        if not results:
            return "No relevant documents found"
        
        answer = "Found relevant documents:\n\n"
        for i, res in enumerate(results, 1):
            answer += f"{i}. {res['document'][:500]}... (score: {res['score']:.3f})\n\n"
        
        return answer

async def main():
    storage_dir = Path("./rag_storage_crawls")
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
        print("Cleared old storage directory")
    
    rag = SimpleRAG(working_dir="./rag_storage_crawls")
    await rag.initialize_storages()
    print("SimpleRAG initialized")

    docs = []
    
    kb_path = Path("data/a11y_explore/knowledge_base.json")
    if kb_path.exists():
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        for screen in kb.get("screens", []):
            docs.append(
                f"Screen '{screen['id']}': {screen.get('description', '')} "
                f"URL: {screen.get('url', '')} Title: {screen.get('title', '')}"
            )
        for el in kb.get("elements", []):
            docs.append(
                f"Element '{el.get('instruction', '')}' on screen '{el.get('screen_id', '')}' "
                f"at {el.get('bbox_px', '')} role={el.get('role', '')} name={el.get('name', '')}"
            )
        print(f"Loaded {len(kb.get('elements', []))} elements from a11y")

    traces_dir = Path("data/cua_explore/screenshot_plus_som/traces")
    if traces_dir.exists():
        for trace_file in traces_dir.glob("*.jsonl"):
            task_id = trace_file.stem.replace("_screenshot_plus_som", "")
            with open(trace_file, 'r', encoding='utf-8') as f:
                for line in f:
                    step = json.loads(line)
                    docs.append(
                        f"Task '{task_id}' step {step.get('step', '')}: "
                        f"from state {step.get('state_id_before', '')} "
                        f"action {step.get('action', {}).get('action', '')} "
                        f"to state {step.get('state_id_after', '')} "
                        f"success={step.get('execution', {}).get('success', False)}"
                    )
            print(f"Loaded traces from {trace_file.name}")

    print(f"Total documents: {len(docs)}")

    if len(docs) == 0:
        print("No documents to index!")
        Path("data/target_app/rag_from_crawls_ready.txt").touch()
        return

    batch_size = 50
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    print(f"Indexing {len(docs)} documents in {total_batches} batches...")
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)")
        await rag.ainsert(batch)
        print(f"Batch {batch_num} completed")
    
    print("Indexing complete")

    print("\nTest query: 'how to add product to cart'")
    result = await rag.aquery("how to add product to cart")
    print(f"Result: {result}")

    Path("data/target_app/rag_from_crawls_ready.txt").touch()
    print("Data saved")

if __name__ == "__main__":
    asyncio.run(main())