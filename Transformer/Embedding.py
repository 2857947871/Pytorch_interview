import torch
import torch.nn as nn

# 词汇表大小为7, 每个单词的嵌入向量为3维
vocab_size = 7
embedding_dim = 3

# 定义输入的单词索引序列
# Vocabulary: ["I", "love", "hate", "cats", "dogs", "you", "me"]
input_sentence1 = torch.tensor([0, 1, 3])  # 对应句子 "I love cats"
input_sentence2 = torch.tensor([0, 2, 4])  # 对应句子 "I hate dogs"

# 创建一个 Embedding 层
# embedding.shape: [7, 3]
# 一共 7 个词, 每个词是长度为 3 的向量
embedding = nn.Embedding(vocab_size, embedding_dim)
print(embedding.weight)

# 将单词索引转换为嵌入向量
embedded_sentence1 = embedding(input_sentence1)
embedded_sentence2 = embedding(input_sentence2)

# 打印嵌入向量
print("嵌入向量 (I love cats):")
print(embedded_sentence1)
print(f"I   : {embedded_sentence1[0]}")
print(f"love: {embedded_sentence1[1]}")
print(f"cats: {embedded_sentence1[2]}")
print("\n")

print("嵌入向量 (I hate dogs):")
print(f"I   : {embedded_sentence2[0]}")
print(f"hate: {embedded_sentence2[1]}")
print(f"dogs: {embedded_sentence2[2]}")