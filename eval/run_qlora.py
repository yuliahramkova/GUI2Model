import os
import json
from pathlib import Path

train_data = "data/target_app/sft_data.jsonl"
output_dir = "data/target_app/qlora_adapter"

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_rank = 16
lora_alpha = 32
learning_rate = 2e-4
num_epochs = 3

cmd = f"""
swift sft \
    --model {model_name} \
    --dataset {train_data} \
    --output_dir {output_dir} \
    --num_train_epochs {num_epochs} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate {learning_rate} \
    --lora_rank {lora_rank} \
    --lora_alpha {lora_alpha} \
    --lora_dropout 0.05 \
    --use_flash_attn True \
    --optim adamw_torch \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 2 \
    --seed 42
"""

print(f"Running: {cmd}")
os.system(cmd)
