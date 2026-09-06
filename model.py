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


import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field  
import json
from typing import List, Optional
from torch.utils.checkpoint import checkpoint

@dataclass
class GPTConfig:
    vocab_size: int = 40960
    hidden_size: int = 768             
    num_hidden_layers: int = 12        
    num_attention_heads: int = 12      
    max_position_embeddings: int = 1024 
    
    intermediate_size: int = 3072      
    hidden_act: str = "gelu"           
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05      
    dropout: float = 0.1
    tie_word_embeddings: bool = True
    
    pad_token_id: int = 0
    bos_token_id: int = 8
    eos_token_id: int = 8
    
    model_type: str = "vexion_gpt"
    architectures: List[str] = field(default_factory=lambda: ["VexionGPTForCausalLM"])
    transformers_version: Optional[str] = None

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        
        if config.hidden_act == "gelu":
            self.act = nn.GELU(approximate="tanh") 
        elif config.hidden_act == "silu":
            self.act = nn.SiLU()
        elif config.hidden_act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Неизвестная активация: {config.hidden_act}")
            
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        self.n_head = config.num_attention_heads
        self.embed_dim = config.hidden_size
        
        self.dropout_p = config.dropout 
        
    def forward(self, x):
        B, T, C = x.size() 

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x, use_ckpt=True): 
        def _forward_block(hidden_states):
            h = hidden_states + self.attn(self.ln_1(hidden_states))
            h = h + self.mlp(self.ln_2(h))
            return h
            
        if self.training and use_ckpt:
            x = checkpoint(_forward_block, x, use_reentrant=False)
        else:
            x = _forward_block(x)
            
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)]),
            ln_f = nn.LayerNorm(config.hidden_size),
        ))

        self.register_buffer(
            "position_ids",
            torch.arange(0, config.max_position_embeddings, dtype=torch.long),
            persistent=False
        )
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_hidden_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.max_position_embeddings, f"Cannot forward sequence of length {t}, block size is only {self.config.max_position_embeddings}"
        
        pos = self.position_ids[:t]
        
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        
        embeddings = tok_emb + pos_emb
            
        x = self.transformer.drop(embeddings)
        
        for i, block in enumerate(self.transformer.h):
            # True for even layers, False for odd ones
            should_ckpt = (i % 1 == 0) 
            x = block(x, use_ckpt=should_ckpt)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            B, T, C = x.size()
            x_flat = x.view(B * T, C)
            targets_flat = targets.view(B * T)
            
            chunk_size = 2048 
            loss = 0.0
            
            for i in range(0, B * T, chunk_size):
                x_chunk = x_flat[i:i+chunk_size]
                targets_chunk = targets_flat[i:i+chunk_size]
                
                logits_chunk = self.lm_head(x_chunk)
                
                loss_chunk = F.cross_entropy(logits_chunk, targets_chunk, label_smoothing=0.1, reduction='sum')
                loss = loss + loss_chunk
                
            loss = loss / (B * T)
            
            return None, loss
            
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            return logits, loss
