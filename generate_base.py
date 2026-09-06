# Copyright 2026 Dmitry
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT, GPTConfig
from safetensors.torch import load_file  
import os
import re

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.2, device='cuda'):
    model.eval()
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding.ids, dtype=torch.long, device=device).unsqueeze(0)

    prompt_length = input_ids.size(1)

    with torch.no_grad():
        autocast_device = 'cuda' if 'cuda' in device else 'cpu'
        with torch.amp.autocast(autocast_device, enabled=(autocast_device == 'cuda')):
            for _ in range(max_new_tokens):
                
                current_input = input_ids if input_ids.size(1) <= model.config.max_position_embeddings else input_ids[:, -model.config.max_position_embeddings:]
                
                logits, _ = model(current_input)
                logits = logits[:, -1, :].float() 
                
                if repetition_penalty != 1.0:
                    seen_tokens = torch.unique(input_ids)
                    score = logits[0, seen_tokens]
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[0, seen_tokens] = score

                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                generated_tail_ids = input_ids[0, prompt_length:].tolist()
                tail_text = tokenizer.decode(generated_tail_ids)
                
                if "<|endoftext|>" in tail_text:
                    break

    final_tail_ids = input_ids[0, prompt_length:].tolist()
    final_text = tokenizer.decode(final_tail_ids)
    
    if "<|endoftext|>" in final_text:
        final_text = final_text.split("<|endoftext|>")[0]

    final_text = re.sub(r'\s+([.,:;!?])', r'\1', final_text)
    final_text = final_text.replace(' )', ')').replace(' ]', ']').replace(' "', '"')

    return final_text


if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoints/gpt_step_7000.safetensors"
    TOKENIZER_PATH = "tokenizer.json"
    CONFIG_PATH = "config.json"
    DEVICE = "cuda" 

    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_K = 40
    TOP_P = 0.9
    REP_PENALTY = 1.2

    print(" Загрузка токенизатора...")
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    else:
        raise FileNotFoundError(f"Токенизатор {TOKENIZER_PATH} не найден!")

    print(" Чтение архитектуры из конфига..")
    config = GPTConfig.from_json(CONFIG_PATH)
    
    with torch.device(DEVICE):
        model = GPT(config)
    
    print(f" Загрузка весов из {CHECKPOINT_PATH}..")
    state_dict = load_file(CHECKPOINT_PATH, device=DEVICE)

    keys_to_delete = [k for k in state_dict.keys() if k.endswith('.attn.bias')]
    for k in keys_to_delete:
        del state_dict[k]

    if "transformer.wte.weight" not in state_dict and "lm_head.weight" in state_dict:
        state_dict["transformer.wte.weight"] = state_dict["lm_head.weight"]
    elif "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    print(" Модель успешно загружена\n")

    while True:
        user_prompt = input("\nЯ: ")
        
        if user_prompt.lower() in ['exit', 'выход', 'quit']:
            print("Завершение работы...")
            break
            
        if not user_prompt.strip():
            continue

        print("\nVexion:", end=" ", flush=True)
        
        output = generate(
            model, tokenizer, user_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K, top_p=TOP_P,
            repetition_penalty=REP_PENALTY,
            device=DEVICE
        )
        
        print(output.strip())
