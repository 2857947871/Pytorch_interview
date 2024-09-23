import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
================================================================================
================================= datasets =====================================
================================================================================
"""
# word embedding, 以序列建模为例
# 考虑source sentence 和 target sentence
# 构建序列, 序列的字符以其在词表中的索引的形式表示
batch   = 2
src_len = torch.Tensor([2, 5]).to(torch.int32)
tgt_len = torch.Tensor([3, 5]).to(torch.int32)

# 单词表大小
"""
输入单词表和输出单词表都有8个词
每次输入和输出最多输入5个词
每个词用长度为8的词向量表示
"""
model_dim = 8
max_src_seq_len = 5
max_tgt_seq_len = 5
max_num_src_words = 8
max_num_tgt_words = 8

src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len - L)), 0) \
    for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_tgt_seq_len - L)), 0) \
    for L in tgt_len])

# 假设 batch 为 2, 
# src: 第一个句子长度为 2, 第二个句子长度为 5
# tgt: 第一个句子长度为 3, 第二个句子长度为 5
print(f"src_len: {src_len}\n")  # tensor([2, 5])
print(f"tgt_len: {tgt_len}\n")  # tensor([3, 5])

"""
tensor([[5, 5, 0, 0, 0],
        [2, 7, 5, 2, 0]])
tensor([[7, 4, 3, 6, 0],
        [6, 4, 5, 0, 0]])
"""
# 单词索引的序列, 单词索引构成的句子 -> 构成源句子和目标句子并padding
print(f"src_seq:\n {src_seq}\n")
print(f"tgt_seq:\n {tgt_seq}\n")


"""
================================================================================
============================== Word Embedding ==================================
================================================================================
"""
# 构建 Embedding
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim) # +1: 因为进行了padding
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)
print(f"src_embedding_table.weight:\n {src_embedding_table.weight}\n")

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
print(f"src_seq:\n {src_seq}\n")
print(f"src_embedding:\n {src_embedding}\n")
print(f"tgt_seq:\n {tgt_seq}\n")
print(f"tgt_embedding:\n {tgt_embedding}\n")


"""
================================================================================
============================ Position Embedding ================================
================================================================================
"""
# 论文公式: PE(pos, 2i) = sin(pos / 10000^(2i / dmodel))   PE(pos, 2i + 1) = cos(pos / 10000^(2i / dmodel))
max_position_len = 5 # 训练中所有样本最大长度为 5
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat   = (torch.arange(0, 8, 2) / model_dim).reshape((1, -1)) 
i_mat   = torch.pow(10000, i_mat)
pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat) # 偶数列
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat) # 奇数列
print(f"pe_embedding_table: \n{pe_embedding_table}\n")

pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
print(f"pe_embedding.weight: \n{pe_embedding.weight}\n")

# 注: 传入位置索引而不是单词在单词表中的索引
#   eg: src_seq: [2, 2, 7, 6, 0] 这是在单词表中的索引
#       但是我们要传入的是位置索引[0, 1, 2, 3] 0: padding无需传入
#       tgt_seq: [4, 4, 3, 7, 4]
#       位置索引: [0, 1, 2, 3, 4]
"""
torch.arange(max(src_len)): 生成位置索引[0, 1, 2, 3]
.unsqueeze(0): 添加一个维度, 一维张量[4] -> 二维张量[1, 4] [[0, 1, 2, 3]]
.repeat(src_len.size(0), 1): 在第0维度重复2次(2个句子), 第1维度重复一次 [[0, 1, 2, 3], [0, 1, 2, 3]]

注: 不推荐做法
    不使用 .unsqueeze(0) 直接 .repeat(src_len.size(0), 1): 没有添加维度, repeat 两个维度, 会自行“广播” -> 结果好像没有问题
    不使用 .unsqueeze(0) 直接 .repeat(src_len.size(0)): 没有添加维度, repeat 一个维度, 仅会 repeat 唯一的那个维度
        [0, 1, 2, 3] -> [0, 1, 2, 3, 0, 1, 2, 3] -> 结果错误
"""
src_pos = torch.arange(max(src_len)).repeat(src_len.size(0), 1).to(torch.int32)
tgt_pos = torch.arange(max(tgt_len)).unsqueeze(0).repeat(tgt_len.size(0), 1).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
print(f"src_seq:\n {src_seq}\n")
print(f"tgt_seq:\n {tgt_seq}\n")
print(f"src_pos:\n {src_pos}\n")
print(f"tgt_pos:\n {tgt_pos}\n")
print(f"src_pe_embedding: \n{src_pe_embedding}\n")
print(f"tgt_pe_embedding: \n{tgt_pe_embedding}\n")


"""
================================================================================
============================ self-attention masked =============================
================================================================================
"""
# mask.shape: [batch, max_src_len, max_src_len], val: 1 or -inf
# src_len: batch_size
# src_len = torch.Tensor([2, 4]).to(torch.int32)
# [tensor([1., 1., 0., 0., 0.]), tensor([1., 1., 1., 1., 0.])]
#                |
#                V
# tensor([[1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0.]])
valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_src_seq_len - L)), 0) \
                        for L in src_len])

# 每一行都是一个句子, 句子与句子之间是无关的
# valid_encoder_pos.shape: [2, 5, 1]
# 2: batch_size   5: 句子的最大长度   1: 为了矩阵运算, 扩维
# valid_encoder_pos_matrix.shape: [2, 5, 5]
# valid_encoder_pos_matrix: 邻接矩阵, 每个样本(句子)中的单词与其他单词的关系
valid_encoder_pos = torch.unsqueeze(valid_encoder_pos, 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
mask_encoder_self_attention = (1 - valid_encoder_pos_matrix).to(torch.bool)
print(mask_encoder_self_attention) # true: 需要被mask(padding 出来的0)

score = torch.randn(batch, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, -1)
print(f"score:\n{score}")
print(f"masked_score:\n{masked_score}")
print(f"prob:\n{prob}")
"""
第一个矩阵: 
    [0.2684, 0.7316, 0.0000, 0.0000, 0.0000]: 因为后3个位置没有词
    [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]: 这5个都被masked了, 归一化后为均匀概率
    [0.4504, 0.1930, 0.0653, 0.2070, 0.0843]: 没有被masked, 那就正常
prob:
tensor([[[0.2684, 0.7316, 0.0000, 0.0000, 0.0000],
        [0.7198, 0.2802, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],

        [[0.4504, 0.1930, 0.0653, 0.2070, 0.0843],
        [0.1755, 0.2840, 0.3276, 0.1692, 0.0437],
        [0.3410, 0.3916, 0.1130, 0.0881, 0.0663],
        [0.1918, 0.0959, 0.2819, 0.0205, 0.4100],
        [0.0351, 0.6241, 0.0649, 0.0537, 0.2221]]])
"""


"""
================================================================================
============================ intra-attention masked ============================
================================================================================
"""
# (Q * K.T).shape: [batch, tgt_seq_len, src_seq_len]
valid_decoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_tgt_seq_len - L)), 0) \
                        for L in tgt_len])
valid_decoder_pos = torch.unsqueeze(valid_decoder_pos, 2)

# Q: decode   K, V: encoder
# [2, 4, 1] * [2, 1, 4] -> [2, 4, 4]
# valid_cross_pos: 邻接矩阵
"""
tensor([[[1., 1., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]],
"""
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
mask_corss_attention = (1 - valid_cross_pos_matrix).to(torch.bool)
print(mask_corss_attention) 