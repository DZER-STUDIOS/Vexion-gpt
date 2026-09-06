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


from tokenizer import train_tokenizer

def line_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line

train_tokenizer(
    texts=line_generator("dataset.txt"), 
    vocab_size=40960, 
    save_path="tokenizer.json"
)
