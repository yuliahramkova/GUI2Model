import json
import random
from pathlib import Path

def load_a11y_data():
    kb_path = Path("data/a11y_explore/knowledge_base.json")
    if not kb_path.exists():
        return []
    
    kb = json.load(open(kb_path))
    samples = []
    for el in kb.get("elements", []):
        instruction = el.get("instruction", "")
        bbox = el.get("bbox_px", [0,0,0,0])
        screen_id = el.get("screen_id", "")
        samples.append({
            "instruction": instruction,
            "bbox": bbox,
            "screen_id": screen_id,
            "type": "grounding"
        })
    return samples

def load_cua_traces():
    traces_dir = Path("data/cua_explore/screenshot_plus_som/traces")
    if not traces_dir.exists():
        return []
    
    samples = []
    for trace_file in traces_dir.glob("*.jsonl"):
        task_id = trace_file.stem.replace("_screenshot_plus_som", "")
        with open(trace_file) as f:
            steps = [json.loads(line) for line in f]
            if not steps:
                continue
            
            # Формируем диалог: задача -> последовательность действий
            goal = steps[0].get("goal", "")
            action_sequence = []
            for step in steps:
                action = step.get("action", {}).get("action", "")
                value = step.get("action", {}).get("value", "")
                if action == "click":
                    action_sequence.append(f"click on element {value}")
                elif action == "type":
                    action_sequence.append(f"type '{value}'")
                elif action == "press":
                    action_sequence.append(f"press {value}")
                elif action == "done":
                    action_sequence.append("task complete")
            
            if action_sequence:
                samples.append({
                    "instruction": f"Task: {goal}",
                    "response": " -> ".join(action_sequence),
                    "task_id": task_id,
                    "type": "procedure"
                })
    return samples

def prepare_sft_dataset():
    a11y_samples = load_a11y_data()
    cua_samples = load_cua_traces()
    
    # Формат для QLoRA (conversational)
    sft_data = []
    
    for sample in a11y_samples[:100]:  # Ограничим для демо
        sft_data.append({
            "conversations": [
                {"from": "human", "value": f"Where is the element '{sample['instruction']}'?"},
                {"from": "gpt", "value": f"It is at {sample['bbox']}"}
            ]
        })
    
    for sample in cua_samples:
        sft_data.append({
            "conversations": [
                {"from": "human", "value": sample["instruction"]},
                {"from": "gpt", "value": sample["response"]}
            ]
        })
    
    output_path = Path("data/target_app/sft_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(sft_data)} samples to {output_path}")
    return sft_data

if __name__ == "__main__":
    prepare_sft_dataset()