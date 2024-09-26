import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from my_Datasets import *


"""
================================================================================
=================================== Params =====================================
================================================================================
"""
d_model     = 512   # 每个词用多少维的向量表示
d_ff        = 2048  # FeedForwrad层输入的维度
d_k         = 64
d_v         = 64
n_layers    = 6     # 6个Encoder/Decodeer Layer
n_heads     = 8     # mutil-head的head的数量


"""
================================================================================
=================================== utils ======================================
================================================================================
"""
# 生成遮盖padding部分的mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    # [batch_size, seq_len] -> [batch_size, 1, len_k]
    # 0为true, 非0为false
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    
    # [batch_size, 1, len_k] -> [batch_size, len_q, len_k]
    # pad_attn_mask = pad_attn_mask.repeat(1, len_q, 1) # 也可以使用repeat
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)

    return pad_attn_mask

# 生成上三角矩阵, 保持因果性
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


"""
================================================================================
============================== Mutil Attention =================================
================================================================================
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        
        # attn: prob   context: result
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc  = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln  = nn.LayerNorm(d_model)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v, d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        
        # [batch_size, len_q, d_model] -> Linear -> [batch_size, len_q, d_k * n_heads]
        # -> [batch_size, len_q, n_heads, d_k] -> [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_Q(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_Q(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        
        # 感觉没必要.repeat, 完全可以自行广播
        # attn_mask: [batch_size, len_v, d_model] -> [batch_size, 1, len_v, d_model]
        # -> [batch_size, n_heads, len_v, d_model]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: result   attn: prob
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        output += residual
        output = self.ln(output)
        
        return output, attn
    
    
"""
================================================================================
================================== Encoder =====================================
================================================================================
"""
# 论文公式: PE(pos, 2i) = sin(pos / 10000^(2i / dmodel))   PE(pos, 2i + 1) = cos(pos / 10000^(2i / dmodel))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 每个句子最长为50个单词, position: [0, 1, ..., 49]
        # .unsqueeze(1), 在第一维增加一个维度
        # [50,] -> [50, 1], 转为列向量(方便后续的广播)
        # pe: Positional Embedding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # sin(pos / 10000^(2i / dmodel))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 不知道为什么非要用exp
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model) # 这是我写的简易版本
        
        # pe: [50, 8] 最长50个danci, 每个单词是长度为d_model的词向量且不更新
        # input: [batch_size, src_size, d_model]
        # 因此, unsqueeze(0), 添加 batch_size 维度
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 转置: 与输入有关
        #   eg:(sequence_length, batch_size, d_model), 需要转置以对齐维度
        #      (batch_size, sequence_length, d_model), 则不需要转置, 只需要 unsqueeze(0)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        # x: [seq_len, batch_size, d_model]
        # 为什么使用相加提供位置信息?
        #   不改变词向量的语义空间, 如果使用乘除或拼接这种复杂的操作, 会破坏语义空间
        # [:x.size(0), :]: 
        #   切片: 从 max_len 截取 seq_len
        # x = x + self.pe[:x.size(0), :] # 虽然是三维, 但是使用两次切片操作已经够了
        x = x + self.pe[:x.size(0), :, :] # 更具体可读性, 切片, 除了第一维需要切, 剩下两个维度全部保留
        return self.dropout(x)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.ln(output + residual)
        
        return output # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        attn: [batch_size, n_heads, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]

        return enc_outputs, attn
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers  = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
    def forward(self, enc_inputs):
        
        # [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        
        # [batch_size, src_len, d_model]
        # 是不是脱裤子放屁？
        # 转置 -> [src_len, batch_size, d_model] -> 处理 -> 转置 -> [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        
        # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model] 
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns
    
    
"""
================================================================================
================================== Decoder =====================================
================================================================================
"""
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn  = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        
        return dec_outputs, dec_self_attn, dec_enc_attn
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers  = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)
        
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs)
        
        # dec_self_attn_pad_mask: ture or false  padding
        # dec_self_attn_subsequence_mask: 0 or 1 因果性
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        
        # 大于0 -> true, 小于0 -> false
        # 相加: 相当于"或", 避免关注不符合因果顺序和padding的区域
        dec_self_attn_mask = (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask > 0)

        # decoder不关注encoder中的padding部分
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return dec_outputs, dec_self_attns, dec_enc_attns
        
        
"""
================================================================================
================================ Transformer ===================================
================================================================================
"""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # 输入: 词向量(用d_model维的i向量表示)
        # 输出: 英语字库(tgt_vocab_size)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
    def forward(self, enc_inputs, dec_inputs):
        """
        enc_input: [batch_size, src_len]
        dec_input: [batch_size, tgt_len]
        """
        
        enc_output, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_output)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, dec_self_attns, dec_enc_attns]
        
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns




model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(100):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:         
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)            
    return dec_input

# Test
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
