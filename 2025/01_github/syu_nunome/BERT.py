import torch
from transformers import BertTokenizer, BertModel

# モデルの読み込み
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 文
sentence1 = "これはこの文の埋め込みベクトルです"

# トークナイズ
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=128)

# モデルを使って文を埋め込みベクトルに変換
with torch.no_grad():
    outputs1 = model(**inputs1)

# 最終層の全tokenのベクトルを取得
hidden_states1 = outputs1.last_hidden_state

# ベクトルを出力
print(f"Last hidden states for sentence1 (shape={hidden_states1.shape}):\n{hidden_states1.numpy()}\n")