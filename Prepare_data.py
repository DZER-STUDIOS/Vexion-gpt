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


import numpy as np
from tokenizers import Tokenizer

TXT_FILE = "dataset.txt"       
BIN_FILE = "train.bin"    
TOKENIZER_PATH = "tokenizer.json"

print("Загрузка токенизатора...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tokenizer.token_to_id("<|endoftext|>")

print(f"Конвертируем {TXT_FILE} в {BIN_FILE}...")

with open(BIN_FILE, 'wb') as f_out:
    batch_ids = []
    
    with open(TXT_FILE, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            ids = tokenizer.encode(line).ids + [eos_id]
            batch_ids.extend(ids)
            
            if len(batch_ids) >= 5_000_000:
                arr = np.array(batch_ids, dtype=np.uint16)
                f_out.write(arr.tobytes())
                batch_ids = []
                
    if batch_ids:
        arr = np.array(batch_ids, dtype=np.uint16)
        f_out.write(arr.tobytes())

print("Готово!")
