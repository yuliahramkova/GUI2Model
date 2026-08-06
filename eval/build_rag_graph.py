import os
import asyncio
import json
import numpy as np
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.hf import hf_model_complete, hf_embed

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def embedding_func(texts: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return await hf_embed(
        texts,
        tokenizer=tokenizer,
        embed_model=model
    )

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return "Local LLM response placeholder"

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag

async def main():
    rag = await initialize_rag()
    print("LightRAG initialized")

    json_path = Path("data/target_app/ground_truth/bboxes.json")
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    docs = []
    for item in data:
        image_name = Path(item["image"]).name
        instruction = item["instruction"]
        bbox = item["bbox"]
        docs.append(
            f"On the screen '{image_name}', there is a UI element '{instruction}' located at {bbox}."
        )

    print(f"Loaded {len(docs)} documents")

    print("Indexing documents...")
    try:
        await rag.ainsert(docs)
        print("Indexing complete")
    except Exception as e:
        print(f"Indexing error: {e}")
        return

    print("Test query: 'click on Add to Cart'")
    try:
        result = await rag.aquery("click on Add to Cart", param=QueryParam(mode="hybrid"))
        print(f"Result: {result[:300]}...")
    except Exception as e:
        print(f"Query error: {e}")

    print("Graph saved to ./rag_storage")
    Path("data/target_app/rag_graph_ready.txt").touch()

if __name__ == "__main__":
    asyncio.run(main())