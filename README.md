# 🧠 Vexion-GPT: Custom Dense LLM Architecture

**Vexion-GPT** is a classic dense language model engine built from scratch on PyTorch. The project was created to deeply understand the architecture of Transformers, optimize memory, and pretraining processes without the use of heavy third-party frameworks.

Currently, the repository contains the **architecture source code** and **base training graphs**. Model weights will be published after completing extensive pretraining runs on large datasets.

## ⚙️ Key Features of the Engine (Under the Hood)

Unlike many "training" models, Vexion-GPT is designed for real-world big data and maximum GPU utilization:

* **Pure Dense Architecture:** Classic, mathematically pure GPT architecture without additives (no MoE, no RoPE). Only the proven Causal Attention and GELU/SiLU activation functions. * **Flash Attention Integrated:** Full support for Fused Kernels for on-the-fly attention computation. VRAM consumption has been dramatically reduced (the model can be easily trained on consumer GPUs with batch sizes that previously caused OOMs).
* **Ultra-fast Custom DataLoader:** The dataloader has been rewritten to stream binary data (`.bin`), bypassing Python's garbage collector and Windows system caching. The token feed rate is static and does not degrade over long distances.
* **HF-Compatible Config:** The architecture is completely decoupled from hardcoded code. Model configuration is implemented via `config.json` according to Hugging Face standards (full support for `hidden_size`, `num_hidden_layers`, etc.).
* **Memory-Efficient Adafactor Optimizer:** Replaced the standard AdamW with Google's Adafactor. By completely dropping the momentum matrix and factorizing the variance, the optimizer's memory footprint is drastically minimized. This saves gigabytes of VRAM, allowing for larger batch sizes on consumer hardware without hitting OOM.
* **Total Layer-wise Checkpointing:** Implemented an aggressive gradient checkpointing strategy that wraps the entire TransformerBlock (both Attention and MLP components). This forces the GPU to discard intermediate activation "drafts" during the forward pass, successfully dropping active memory consumption on the Large model from 11.3GB down to a flat 6.0GB during training.

## 📉 System Requirements and VRAM Consumption (Pre-training)
Because Vexion-gpt is a classic, dense transformer without overloaded modern add-ons, it is incredibly resource-efficient. Integration Flash Attention and proper memory management allow model training from scratch even on budget home graphics cards.

The main secret to its low power consumption lies in its support for gradient accumulation. Training with batch_size = 4 and accumulate_steps = 24 results in a huge effective batch size (96), but the physical memory consumption remains at the level of a single pass. Gradient accumulation consumes computational time but does not inflate VRAM.

Memory Measurements (with a 1024-token context)
Below are actual video memory consumption measurements for training from scratch (including weights, gradients, optimizer states, and PyTorch cache):

**Nano / 117M parameters**
(Hidden size: 768, Layers: 12, Heads: 12)

Consumption: 2.2 GB of VRAM

Suitable for: RTX 3050 / RTX 4050 Laptop / Any NVIDIA RTX graphics card with 4 GB or more.

**Medium / 345M parameters**
(Hidden size: 1024, Layers: 24, Heads: 16)

Consumption: 3.5 GB of VRAM

Suitable for: RTX 3050 / RTX 4050 / Any NVIDIA RTX graphics card with 4-6 GB or more.

**Large / 646M parameters**
(Hidden size: 1280, Layers: 30, Heads: 20)

Consumption: 4.7 GB of VRAM

Suitable for: RTX 3060 / Any NVIDIA RTX graphics card with 6 GB or more.

**XL / 1B parameters**
(Hidden size: 1536, Layers: 36, Heads: 12)

Consumption: 6.4 GB VRAM

Suitable for: RTX 3060 Ti / RTX 3050 / RTX 4060 / Any NVIDIA RTX graphics card with 8 GB or more.

**XXL / 2.5B parameters**
(Hidden size: 2048, Layers: 48, Heads: 16)

Consumption: 12.9 GB VRAM

Suitable for: RTX 4060 Ti / RTX 5060 Ti / Any NVIDIA RTX graphics card with 14-16 GB or more.

**XXXL / 3B parameters**
(Hidden size: 2176, Layers: 52, Heads: 17)

Consumption: 17 GB VRAM

Suitable for: RTX 3090 / RTX 4090 / Any NVIDIA RTX graphics card with 20+ GB or more.

Technically, 3 billion parameters can be trained on a 16GB GPU. **BUT be prepared for the following**: 
1) In terms of memory, only a very small portion of the gradients will fit in VRAM — the rest will go into RAM.
2) There will be no speed. If you take into account that only batch_size 4 will fit at most, be prepared for the fact that as the accumulate_steps values increase, the speed will drop to **MINUTES**.

Technically, in the model.py file, you can set the chunk size to 1024 instead of 2048 in the slicing process — but the speed will also drop, even though the consumption will also drop closer to 15.5 GB **APPROXIMATELY**. For example, the model’s speed on my computer is 48.64 s/it. So, do this **ONLY WITH FULL AWARENESS** that you will need a very long time to train the model.

**💡 Note: Power consumption is given for mixed precision (bfloat16 / float16).

**Now let’s run tests with the models, but only by increasing the context. We’ll start with version 1024 and move forward. How much video memory will we need to, say, train it on a broader context than the original one?**

`Model: Vexion-gpt-NANO`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 2.2GB | 3GB |
| **2048** | 3.5GB | 4GB |
| **3072** | 4.6GB | 6GB | 
| **4096** | 5.8GB | 6GB |
| **5120** | 6.9GB | 8GB |
| **6144** | 8GB | 10GB |
| **7168** | 8.9GB | 10GB |
| **8192** | 10GB | 11GB |
| **9216** | 11.4GB | 12GB |
| **10240** | 12.5GB | 14GB |
| **11264** | 13.7GB | 14GB |
| **12288** | 14.8GB | 16GB |

`Model: Vexion-gpt-Medium`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 3.5GB | 4GB |
| **2048** | 4.6GB | 6GB |
| **3072** | 5.6GB | 6GB | 
| **4096** | 6.9GB | 8GB |
| **5120** | 8.2GB | 10GB |
| **6144** | 9.8GB | 11GB |
| **7168** | 11.5GB | 12GB |
| **8192** | 12.6GB | 14GB |
| **9216** | 13.9GB | 16GB |

`Model: Vexion-gpt-Large`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 4.7GB | 6GB |
| **2048** | 6.1GB | 8GB |
| **3072** | 7.5GB | 8GB |
| **4096** | 8.7GB | 10GB |
| **5120** | 10.5GB | 11GB |
| **6144** | 12.3GB | 14GB |
| **7168** | 14GB | 16GB |
| **8192** | 15.6GB | 16GB |

`Model: Vexion-gpt-XL`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 6.4GB | 8GB |
| **2048** | 8.1GB | 10GB |
| **3072** | 9.5GB | 11GB |
| **4096** | 11.4GB | 12GB |
| **5120** | 13.1GB | 14GB |
| **6144** | 15.2GB | 16GB |

`Model: Vexion-gpt-XXL`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 12.9GB | 14GB |

`Model: Vexion-gpt-XXXL`

| Context | VRAM consumption | better on VRAM |
| :--- | :--- | :--- | 
| **1024** | 17GB | 20+GB |

## 📊 Current training status (Phase 1: Wikipedia)

The attached graphs show the first stage of pre-training the base model.

* **Current build parameters:** 117M parameters (Hidden size: 768, Layers: 12, Heads: 12, ctx: 256).
* **Dataset:** Clean corpus of the Russian-language Wikipedia (~331 MILLION tokens).
* **Goal of this stage:** To develop basic syntax, ideal grammar, and punctuation for the Russian language before moving to "dirty" datasets (CulturaX).

![Снимок экрана (427)](https://cdn-uploads.huggingface.co/production/uploads/683cc817f3e4d66ed5c658db/Y87iT5WcI_KWyeArjVsQf.png)

*
> The graph shows a steady drop in loss without spikes, confirming the absolute mathematical stability of the custom transformer and gradient mechanism. At control checkpoints, the model successfully generates grammatically correct encyclopedic text.

## 🚀 Code Usage

The architecture is ready for experimentation. You can easily scale the model from Nano to 1B+ parameters by simply changing `config.json`.

Example of initializing a pure model graph:

```python
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

vocab_size = config["vocab_size"]
hidden_size = config["hidden_size"]
num_layers = config["num_hidden_layers"]
num_heads = config["num_attention_heads"]
intermediate_size = config["intermediate_size"]
context = config["max_position_embeddings"]

token_embeddings = vocab_size * hidden_size
position_embeddings = context * hidden_size

attention = (
    (hidden_size * 3 * hidden_size) + (3 * hidden_size)
    
    + (hidden_size * hidden_size) + hidden_size
)

mlp = (
    (hidden_size * intermediate_size) + intermediate_size
    + (intermediate_size * hidden_size) + hidden_size
)

layer_norms = 4 * hidden_size

block = attention + mlp + layer_norms

transformer = block * num_layers

final_layer_norm = 2 * hidden_size

lm_head = vocab_size * hidden_size

total_parameters = (
    token_embeddings
    + position_embeddings
    + transformer
    + final_layer_norm
    + lm_head
)

print("Vexion-GPT Parameter Calculator")
print("=" * 35)

print(f"Vocabulary:        {vocab_size:,}")
print(f"Hidden size:       {hidden_size:,}")
print(f"Layers:            {num_layers}")
print(f"Attention heads:   {num_heads}")
print(f"Intermediate size: {intermediate_size:,}")
print(f"Context:           {context:,}")
print()

print(f"Parameters: {total_parameters:,}")
print(f"Parameters: {total_parameters / 1_000_000:.2f} M")
print(f"Parameters: {total_parameters / 1_000_000_000:.3f} B")
print(f"Parameters: {total_parameters / 1_000_000_000_000:.3f} T")
```

Running model training/retraining:

* Before training the model, launch the command prompt (CMD) as administrator and enter the following command: `cd C:\Users\Username\Desktop\model folder`
```
Training: python train.py --data_path train.bin --val_path val.bin --total_steps 30000 --save_every 1000 --batch_size 1 --accumulate_steps 256 --lr 1e-4 --dropout 0.0 --warmup_steps 500

Retraining: python train.py --data_path train.bin --val_path val.bin --total_steps 30000 --save_every 1000 --batch_size 1 --accumulate_steps 256 --lr 1e-4 --dropout 0.0 --warmup_steps 500 --resume checkpoints/gpt_step_1000.safetensors
```

* Instructions: In the generate_base.py file, start changing parameters starting at line 73. In the `CHECKPOINT_PATH =` line, specify the path to the checkpoint. From lines 78-82, change the model settings. **Don't touch anything ABOVE**.

To disable the scheduler for custom LR speed tuning, add the command `--scheduler_type constant` to the other settings in the CMD. To re-enable the scheduler, remove the command.

* **Use the chunk_size = setting in model.py to adjust the speed.** The base number is 4096, but you should set it depending on your memory. If you have a couple of gigabytes of VRAM free, set the number higher. This will increase the speed, albeit not by much. And it may save hours of work.

* Use the layer data storage setting in model.py: should_ckpt = (i % 2 == 0) to increase the speed in seconds. It’s better to use the setting for 16GB VRAM. If you want the minimum consumption, do: should_ckpt = (i % 1 == 0). If you want maximum speed, set: should_ckpt = (i % 2 == 0) and higher.
