import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/target_app/ground_truth/bboxes.json", "r") as f:
    data = json.load(f)

instructions = [item["instruction"] for item in data]
embeddings = model.encode(instructions)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "data/target_app/faiss_index.bin")
np.save("data/target_app/instructions.npy", np.array(instructions))
np.save("data/target_app/embeddings.npy", embeddings)

with open("data/target_app/faiss_metadata.json", "w") as f:
    json.dump(data, f, indent=2)

print("FAISS индекс создан")
