import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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

# Parameters for the model
input_dim = 100  # Size of the vocabulary
d_model = 32  # Embedding dimension
nhead = 4  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
num_decoder_layers = 2  # Number of decoder layers
dim_feedforward = 2048  # Dimension of the feedforward network
dropout = 0.1  # Dropout rate



# 将句子拆分成单词列表
def tokenize(sentence):
    return sentence.split(' ')

# 将句子转换为词汇表中的索引序列
# 假设我们有一个简单的词汇表
vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'Hello': 3, 'world': 4, 'Bonjour': 5, 'le': 6, 'monde': 7}
def encode(sentence, vocab):
    return [vocab[word] for word in tokenize(sentence)]

def decode(sequence, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    return [inv_vocab[idx] for idx in sequence]




class TranslationDataset(Dataset):
    def __init__(self, src_list, tgt_list, src_vocab_size, tgt_vocab_size, max_len=10):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        # src = encode(self.src_list[idx], self.vocab)  # 将句子转换成数字序列
        src = self.src_list[idx]
        tgt = self.tgt_list[idx]
        
        # Pad sequences to max_len
        src = src + [0] * (self.max_len - len(src))
        tgt = tgt + [0] * (self.max_len - len(tgt))
        
        return torch.tensor(src), torch.tensor(tgt)

# 假设 src_list 和 tgt_list 是准备好的数据
src_list = [
    [1, 2, 3, 0, 5],  # Example source sentence 1
    [6, 0, 8, 9]      # Example source sentence 2
    # Add more sentences...
]

tgt_list = [
    [1, 2, 3, 4, 5],  # Example target sentence 1
    [6, 7, 8, 9]      # Example target sentence 2
    # Add more sentences...
]

src_vocab_size = 1000  # Adjust this based on your actual vocabulary size
tgt_vocab_size = 1000  # Adjust this based on your actual vocabulary size

dataset = TranslationDataset(src_list, tgt_list, src_vocab_size, tgt_vocab_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# Create the model
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 训练循环
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for src, tgt in dataloader:
        src = src.transpose(0, 1)  # 转换为[seq_len, batch_size]
        tgt_input = tgt[:, :-1].transpose(0, 1)  # 删除最后一个token，作为decoder的输入
        tgt_output = tgt[:, 1:].transpose(0, 1)  # 删除第一个token，作为预测目标

        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # 将输出reshape为[batch_size * seq_len, vocab_size]
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')



class TextDataset(Dataset):
    def __init__(self, src_list, tgt_list, vocab, max_len=10):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        # src = encode(self.src_list[idx], self.vocab)  # 将句子转换成数字序列
        # tgt = encode(self.tgt_list[idx], self.vocab)
        src = self.src_list[idx]
        tgt = self.tgt_list[idx]
        
        # Pad sequences to max_len
        src = src + [0] * (self.max_len - len(src))
        tgt = tgt + [0] * (self.max_len - len(tgt))
        
        return torch.tensor(src), torch.tensor(tgt)
    


# 测试数据
test_src_list = [
    [1, 2, 3, 4, 5],  # Example source sentence 1
    [6, 7, 8, 9, 10]      # Example source sentence 2
    # Add more sentences...
]

test_tgt_list = [
    [1, 2, 3, 4, 5],  # Example target sentence 1
    [6, 7, 8, 9, 10]      # Example target sentence 2
    # Add more sentences...
]

# 创建数据集和数据加载器
dataset = TextDataset(test_src_list, test_tgt_list, vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model.eval()

# 测试模型
with torch.no_grad():
    for src, tgt in dataloader:

        print('===')
        print(src)
        print(tgt)
        print('===')

        src = src.transpose(0, 1)
        tgt_input = tgt[:, :-1].transpose(0, 1)
        
        output = model(src, tgt_input)
        output = output.argmax(dim=-1).transpose(0, 1)
        
        # print(src)
        print('-----')
        print(output)
        print('-----+')

torch.save(model.state_dict(), 'small_model1.pth')
