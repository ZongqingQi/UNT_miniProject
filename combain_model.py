import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.pos_decoder = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, input_dim)
        
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt, src, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.fc_out(output)
        
        return output
    
def creat_empty_model(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
    model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    return model


# Parameters for the model
input_dim = 100  # Size of the vocabulary
d_model = 32  # Embedding dimension
nhead = 4  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
num_decoder_layers = 2  # Number of decoder layers
dim_feedforward = 2048  # Dimension of the feedforward network
dropout = 0.1  # Dropout rate


# 加载模型参数
model1_statement = torch.load('small_model1.pth')
model2_statement = torch.load('small_model2.pth')

model1 = creat_empty_model(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model1.load_state_dict(model1_statement)

model2 = creat_empty_model(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model2.load_state_dict(model2_statement)

# model1.transformer.encoder

# for one_key in model1_statement.keys():
#     print(one_key)
# print('-----------------------------------+')

# encoder_layers_model1 = list(model1.transformer.encoder.layers)
# encoder_layers_model2 = list(model2.transformer.encoder.layers)

# decoder_layers_model1 = list(model1.transformer.decoder.layers)
# decoder_layers_model2 = list(model2.transformer.decoder.layers)
# # print(len(encoder_layers_model1))
# print(encoder_layers_model1[0])

# print('------------------------------------')
# new_encoder_layers = nn.ModuleList()
# new_encoder_layers.append(encoder_layers_model1[0])
# new_encoder_layers.append(encoder_layers_model2[0])
# new_encoder_layers.append(encoder_layers_model1[1])
# new_encoder_layers.append(encoder_layers_model2[1])
# print(new_encoder_layers)
# print('-----------------------------------=')
# new_decoder_layers = nn.ModuleList()
# new_decoder_layers.append(decoder_layers_model1[0])
# new_decoder_layers.append(decoder_layers_model2[0])
# new_decoder_layers.append(decoder_layers_model1[1])
# new_decoder_layers.append(decoder_layers_model2[1])
# print(new_decoder_layers)

# print('-----------------------------------+')
# encoder_layer = nn.TransformerEncoderLayer(new_encoder_layers, nhead=4)
# decoder_layer = nn.TransformerDecoderLayer(new_decoder_layers, nhead=4)

# model_encoder_part = nn.TransformerEncoder(encoder_layer, num_layers=4)
# model_decoder_part = nn.TransformerDecoder(decoder_layer, num_layers=4)

# print(model_encoder_part)


class CombinedTransformerModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedTransformerModel, self).__init__()
        
        self.src_embedding = model1.src_embedding
        self.tgt_embedding = model1.tgt_embedding
        
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        # Interleave encoder layers
        for enc1, enc2 in zip(model1.transformer.encoder.layers, model2.transformer.encoder.layers):
            self.encoder_layers.append(enc1)
            self.encoder_layers.append(enc2)
        
        # Interleave decoder layers
        for dec1, dec2 in zip(model1.transformer.decoder.layers, model2.transformer.decoder.layers):
            self.decoder_layers.append(dec1)
            self.decoder_layers.append(dec2)
        
        self.encoder_norm = nn.LayerNorm(model1.transformer.encoder.norm.normalized_shape)
        self.decoder_norm = nn.LayerNorm(model1.transformer.decoder.norm.normalized_shape)
        
        self.fc_out = model1.fc_out
        
        self.d_model = model1.d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Encoder pass
        for layer in self.encoder_layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = self.encoder_norm(src)
        
        # Decoder pass
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.decoder_norm(output)
        
        output = self.fc_out(output)
        
        return output


# Load model states
model1_state_dict = torch.load('small_model1.pth')
model2_state_dict = torch.load('small_model2.pth')

# Create and load models
model1 = creat_empty_model(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model1.load_state_dict(model1_state_dict)

model2 = creat_empty_model(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model2.load_state_dict(model2_state_dict)

# Create combined model
combined_model = CombinedTransformerModel(model1, model2)

# Example usage
src = torch.randint(0, input_dim, (1, 10))  # (sequence_length, batch_size)
tgt = torch.randint(0, input_dim, (1, 10))  # (sequence_length, batch_size)


# 测试数据
test_src_list = [
    [1, 2, 3, 4, 5]  # Example source sentence 1
    # Add more sentences...
]

test_tgt_list = [
    [1, 2, 3, 4, 5],  # Example target sentence 1
    # Add more sentences...
]

test_src = torch.tensor(test_src_list)
test_tgt = torch.tensor(test_tgt_list)



print(test_src)
output = combined_model(test_src, test_tgt)
output = output.argmax(dim=-1).transpose(0, 1)
print(output.shape)
print(output)
