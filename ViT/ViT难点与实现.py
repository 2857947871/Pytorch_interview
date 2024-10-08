import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
================================================================================
================================== image2emb ===================================
================================================================================
"""
def image2emb_naive(image, patch_size, weight):
    
    # image_size: B, C, H, W
    # 对图片分块(不重叠 -> stride = patch_size)
    # unfold: 取出每次卷积的区域
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    # print(patch.shape)
    """
    torch.Size([16, 48, 4])
    16: batch_size
    48: 每个patch为4*4, channel为3 -> 4*4*3 -> 48个pixel
    4:  8*8大小的图片, 每个patch为4*4 -> 每个图片有4个patch
    """
    patch_embedding = numpy.matmul(patch, weight)
    
    return patch_embedding

def image2emb_conv(image, weight, stride):
    
    conv_output = F.conv2d(image, weight, stride=stride)
    batch_size, channel, image_h, image_w = conv_output.shape
    patch_embedding = conv_output.reshape((batch_size, channel, image_h*image_w)).transpose(-1, -2)
    
    return patch_embedding

# test code for image2emb
batch_size, channel, image_h, image_w = 16, 3, 8, 8
patch_size      = 4
model_dim       = 8
max_num_token   = 16
num_classes     = 10
label           = torch.randint(10, (batch_size,))
patch_depth     = patch_size * patch_size * channel
image = torch.randn(batch_size, channel, image_h, image_w)
weight = torch.randn(patch_depth, model_dim) # model_dim: 输出channel   patch_depth: kernel的面积 * 输入channel
kernel = weight.reshape((-1, channel, patch_size, patch_size))

# print(f"weight.shape: {weight.shape}")
# print(image2emb_naive(image, patch_size, weight).shape)
# print(image2emb_conv(image, kernel, patch_size).shape)
patch_embedding = image2emb_naive(image, patch_size, weight)


"""
================================================================================
===================================== CLS ======================================
================================================================================
"""
# 随机初始化的值
cls_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
token_embedding = torch.cat([cls_token_embedding, patch_embedding], dim=1)


"""
================================================================================
============================== position embedding ==============================
================================================================================
"""
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1] # 一般图像大小都是固定的

# torch.tile: 复制与拓展
# 在第0维拓展batch_size次, 在第1与2维保持不变
position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
token_embedding += position_embedding


"""
================================================================================
=================================== Encoder ====================================
================================================================================
"""
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
encoder_output = transformer_encoder(token_embedding)


"""
================================================================================
================================== CLS_output ==================================
================================================================================
"""
cls_token_output = encoder_output[:, 0, :] # 仅取出CLS
linear_layer = nn.Linear(model_dim, num_classes)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
print(loss)