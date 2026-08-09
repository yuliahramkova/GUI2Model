import json
import asyncio
from pathlib import Path
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

async def embedding_func(texts):
    if isinstance(texts, str):
        texts = [texts]
    return embed_model.encode(texts).tolist()

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return "LLM placeholder for indexing"

async def main():
    rag = LightRAG(
        working_dir="./rag_storage_crawls",
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    print("LightRAG initialized")

    docs = []

    kb_path = Path("data/a11y_explore/knowledge_base.json")
    if kb_path.exists():
        kb = json.load(open(kb_path))
        for screen in kb.get("screens", []):
            docs.append(
                f"Screen '{screen['id']}': {screen.get('description', '')} "
                f"URL: {screen.get('url', '')} Title: {screen.get('title', '')}"
            )
        for el in kb.get("elements", []):
            docs.append(
                f"Element '{el['instruction']}' on screen '{el['screen_id']}' "
                f"at {el['bbox_px']} role={el.get('role')} name={el.get('name')}"
            )
        print(f"Loaded {len(kb.get('elements', []))} elements from a11y")

    traces_dir = Path("data/cua_explore/screenshot_plus_som/traces")
    if traces_dir.exists():
        for trace_file in traces_dir.glob("*.jsonl"):
            task_id = trace_file.stem.replace("_screenshot_plus_som", "")
            with open(trace_file) as f:
                for line in f:
                    step = json.loads(line)
                    docs.append(
                        f"Task '{task_id}' step {step['step']}: "
                        f"from state {step.get('state_id_before', '')} "
                        f"action {step.get('action', {}).get('action', '')} "
                        f"to state {step.get('state_id_after', '')} "
                        f"success={step.get('execution', {}).get('success', False)}"
                    )
            print(f"Loaded traces from {trace_file.name}")

    print(f"Total documents: {len(docs)}")

    print("Indexing documents...")
    await rag.ainsert(docs)
    print("Indexing complete")

    print("\nTest query: 'how to add product to cart'")
    result = await rag.aquery("how to add product to cart", param={"mode": "hybrid"})
    print(f"Result: {result[:500]}...")

    Path("data/target_app/rag_from_crawls_ready.txt").touch()
    print("Graph saved to ./rag_storage_crawls")

if __name__ == "__main__":
    asyncio.run(main())