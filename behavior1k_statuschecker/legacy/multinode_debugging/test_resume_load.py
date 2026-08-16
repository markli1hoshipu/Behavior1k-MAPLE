"""Isolated diagnostic: does PeftModel.from_pretrained(base, lora_adapter_v3_epoch0)
hang or just take a while? Prints timestamps around each step."""
import time

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = f"{ROOT}/lora_adapter_v3_epoch0"


def ts():
    return time.strftime("%H:%M:%S")


print(f"[{ts()}] loading base model...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
print(f"[{ts()}] base model loaded. loading adapter from {ADAPTER_PATH} ...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
print(f"[{ts()}] adapter loaded successfully.", flush=True)
model.print_trainable_parameters()
print(f"[{ts()}] done.", flush=True)
