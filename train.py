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


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import glob
import pickle
import bitsandbytes as bnb
from tqdm import tqdm
from safetensors.torch import save_model, load_model
from model import GPT, GPTConfig
from tokenizer import train_tokenizer, load_tokenizer
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.optimization import Adafactor

class FastDataloader:
    def __init__(self, bin_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"✅ Базовый датасет загружен. Всего токенов: {len(self.data):,}")

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.max_seq_len - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.max_seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.max_seq_len].astype(np.int64)) for i in ix])
        return x, y

@torch.no_grad()
def validate(model, val_loader, batch_size, eval_iters=50):
    print("DEBUG: starting fast validation...")
    model.eval()
    
    device = next(model.parameters()).device
    
    losses = torch.zeros(eval_iters)
    
    with torch.no_grad():
        for k in range(eval_iters):
            x, y = val_loader.get_batch(batch_size)
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits, loss = model(x, y)
                
            losses[k] = loss.item()
            
    model.train()
    avg_loss = losses.mean().item()
    print(f"DEBUG: validation finished, avg_loss = {avg_loss:.4f}")
    return avg_loss
                
def line_generator(file_path, max_lines=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            yield line

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

def train(args):
    print(f"DEBUG: args.val_path = {args.val_path}")
    print(f"DEBUG: file exists? {os.path.exists(args.val_path) if args.val_path else False}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = GPTConfig.from_json("config.json")
    max_seq_len = config.max_position_embeddings

    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = load_tokenizer(args.tokenizer_path)
    else:
        raise FileNotFoundError(f"Токенизатор не найден по пути: {args.tokenizer_path}. Для чтения бинарников нужен строго оригинальный токенизатор!")

    print(f" Загрузка .bin: {args.data_path}")
    train_loader = FastDataloader(args.data_path, max_seq_len)
    
    def get_train_batch(): 
        return train_loader.get_batch(args.batch_size)
        
    get_val_batch = None
    if args.val_path and os.path.exists(args.val_path):
        val_loader = FastDataloader(args.val_path, max_seq_len)
        def get_val_batch(): 
            return val_loader.get_batch(args.batch_size)

    global_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.device(device):
        model = GPT(config)
        
    model = model.to(global_dtype)

    total_params_count = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "═"*65)
    print(" АРХИТЕКТУРА МОДЕЛИ")
    print("═"*65)
    print(f" Размер словаря (vocab):           {config.vocab_size}")
    print(f" Размерность (hidden_size):        {config.hidden_size}")
    print(f" Количество слоев (layers):        {config.num_hidden_layers}")
    print(f" Количество голов (heads):         {config.num_attention_heads}")
    print(f" Контекст:                         {max_seq_len} токенов")
    print(f" Всего параметров:                 {total_params_count:,}")
    print(f" Обучаемых параметров:             {trainable_params_count:,}")
    print("\n" + "═"*65)
    print("  Параметры запуска тренировки")
    print("═"*65)
    print(f" Микро-батч: {args.batch_size} | Накопление: {args.accumulate_steps}")
    print(f" Итоговый логический батч: {args.batch_size * args.accumulate_steps}")
    print(f" Learning Rate: {args.lr} | Всего шагов: {args.total_steps}")
    print("═"*65 + "\n")
    print("  Режим базового обучения: тренируем все параметры с нуля.")

    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    optimizer = Adafactor(
            model.parameters(),
            lr=1e-4, 
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None, 
            weight_decay=0.01, 
            relative_step=False, 
            scale_parameter=False, 
            warmup_init=False
        )
    
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps, min_lr_ratio=0.1)
        print("📈 Используется косинусный планировщик LR")
    else:
        scheduler = None
        print(f"➖ Используется ПОСТОЯННЫЙ LR: {args.lr}")
    
    if args.resume:
        try:
            resume_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
            if resume_step < args.total_steps:
                if scheduler is not None:
                    scheduler.last_epoch = resume_step
                    scheduler._step_count = resume_step + 1
        except:
            pass
    
    use_scaler = not torch.cuda.is_bf16_supported()
    if use_scaler:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading model weights from {args.resume}")
        
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        
        if 'lm_head.weight' in sd and 'transformer.wte.weight' not in sd:
            sd['transformer.wte.weight'] = sd['lm_head.weight']
        elif 'transformer.wte.weight' in sd and 'lm_head.weight' not in sd:
            sd['lm_head.weight'] = sd['transformer.wte.weight']
        
        keys_to_delete = [k for k in sd.keys() if k.endswith('.attn.bias')]
        for k in keys_to_delete:
            del sd[k]

        model.load_state_dict(sd, strict=True) 
        
        print("✅ Weights successfully loaded!")

        import gc
        del sd
        if 'new_sd' in locals():
            del new_sd
        gc.collect()
        torch.cuda.empty_cache()

        opt_path = args.resume.replace('.safetensors', '.pt')
        if os.path.exists(opt_path) and not getattr(args, 'use_lora', False):
            print(f"Loading optimizer state from {opt_path}")
            try:
                with open(opt_path, 'rb') as f:
                    opt_state = pickle.load(f)
                optimizer.load_state_dict(opt_state['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                scheduler.load_state_dict(opt_state['scheduler'])
                scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
                start_step = opt_state['step']
                
                print(f"✅ Optimizer loaded! Starting from step {start_step}")

            except Exception as e:
                print(f"⚠️ Optimizer file corrupted ({e}). Starting optimizer from scratch.")
                try:
                    start_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
                except:
                    start_step = 0

            if 'opt_state' in locals():
                del opt_state
                
            gc.collect()
            torch.cuda.empty_cache()
                
        else:
            print("⚠️ Optimizer state not found, starting optimizer from scratch.")
            try:
                start_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
            except:
                start_step = 0
            print(f"✅ Starting from step {start_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    step = start_step
    best_loss = float('inf')
    best_val_loss = float('inf')

    model.train()
    progress_bar = tqdm(total=args.total_steps, initial=step, desc="Training")

    optimizer.zero_grad(set_to_none=True)
    
    micro_step = 0 
    accum_loss = 0.0 

    print("Начинаем обучение...")
    try:
        while step < args.total_steps:
            x, y = train_loader.get_batch(args.batch_size)
            
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                with torch.amp.autocast('cuda', dtype=global_dtype):
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / args.accumulate_steps
            
            accum_loss += loss.item() 
            
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1 

            if micro_step % args.accumulate_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
    
                if scheduler is not None:
                    scheduler.step()
                        
                optimizer.zero_grad(set_to_none=True)

                step += 1
                progress_bar.update(1)

                current_train_loss = accum_loss
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                progress_bar.set_postfix(
                    loss=current_train_loss, 
                    lr=optimizer.param_groups[0]['lr'],
                    vram=f"{allocated:.1f}G/{reserved:.1f}G"
                )
                accum_loss = 0.0

                if step % args.save_every == 0:
                    ckpt_path = os.path.join(args.save_dir, f"gpt_step_{step}.safetensors")
                    save_model(model, ckpt_path)
                    
                    opt_path = ckpt_path.replace('.safetensors', '.pt')
                    with open(opt_path, 'wb') as f:
                        pickle.dump({
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler is not None else None,
                            'step': step
                        }, f)
                    print(f"\nSaved checkpoint to {ckpt_path} and optimizer state")

                    if args.val_path and os.path.exists(args.val_path):
                        print(f"Running validation at step {step}...")
                        val_loss = validate(
                            model,
                            val_loader,
                            args.batch_size,
                            eval_iters=50
                        )
                        print(f"Step {step}: val loss = {val_loss:.4f}")

                        with open("training_log.csv", "a", encoding="utf-8") as f:
                            f.write(f"{step},{current_train_loss:.4f},{val_loss:.4f}\n")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_path = os.path.join(args.save_dir, "gpt_best.safetensors")
                            save_model(model, best_path)
                            print(f"New best model saved with val loss {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано вручную (Ctrl+C)! Переходим к сохранению...")

    print(" Сохраняем финальную модель...")
    final_path = os.path.join(args.save_dir, "gpt_final.safetensors")
    state_dict = model.state_dict()

    if 'lm_head.weight' in state_dict and 'transformer.wte.weight' in state_dict:
        if (state_dict['lm_head.weight'].data_ptr() == state_dict['transformer.wte.weight'].data_ptr()):
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()

    save_model(model, final_path)
    print(f"Model saved to {final_path}")

    opt_path = final_path.replace('.safetensors', '.pt')
    with open(opt_path, 'wb') as f:
        pickle.dump({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step
        }, f)
    print(f"Optimizer state saved to {opt_path}")
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json')
    #parser.add_argument('--vocab_size', type=int, default=32000)
    #parser.add_argument('--embed_dim', type=int, default=256)
    #parser.add_argument('--n_layers', type=int, default=6)
    #parser.add_argument('--n_heads', type=int, default=8)  
    #parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--accumulate_steps', type=int, default=8, help="Шагов накопления для виртуального батча")
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'constant'])
    args = parser.parse_args()
    train(args)
