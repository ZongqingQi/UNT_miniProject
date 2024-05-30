import math
import json

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


# turn sentence to index sequence
def tokenize(sentence):
    sentence_list = []
    for one_word in sentence.split(' '):
        if one_word == '':
            continue
        sentence_list.append(one_word)
    return sentence_list


def encode(sentence, vocab):
    return [vocab[word] for word in tokenize(sentence)]

def decode(sequence, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    return [inv_vocab[idx] for idx in sequence]




# Parameters for the model
input_dim = 1200  # Size of the vocabulary
d_model = 32  # Embedding dimension
nhead = 4  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
num_decoder_layers = 2  # Number of decoder layers
dim_feedforward = 1024  # Dimension of the feedforward network
dropout = 0.1  # Dropout rate


# Turn setence to index
# example dictionary
# vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'Hello': 3, 'world': 4, 'Bonjour': 5, 'le': 6, 'monde': 7}
with open('./index_dictionary/word_idx_dict.json', 'r', encoding='utf-8') as json_in:
    vocab = json.load(json_in)
vocab.update({'<pad>': 0})




class TranslationDataset(Dataset):
    def __init__(self, src_list, tgt_list, src_vocab_size, tgt_vocab_size, vocab, max_len=16):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        src = encode(self.src_list[idx], self.vocab)  # 将句子转换成数字序列
        tgt = encode(self.tgt_list[idx], self.vocab) 
        
        # Pad or Cut sequences to max_len
        if len(src) < self.max_len:
            src = src + [0] * (self.max_len - len(src))  # [vocab['<pad>']] * (self.max_len - len(src))
        else:
            src = src[: self.max_len]

        if len(tgt) < self.max_len:
            tgt = tgt + [0] * (self.max_len - len(tgt))
        else:
            tgt = tgt[: self.max_len]

        # for i in range(input_dim): 

        return torch.tensor(src), torch.tensor(tgt)





# src_list & tgt_list is prepared lv0 lv1 data
src_list = []
with open('./fine_data/fine_data_lv1', 'r', encoding='utf-8') as data_in:
    for one_line in data_in:
        one_line = one_line.strip('\n')
        src_list.append(one_line)

tgt_list = src_list



src_vocab_size = 1200  # Adjust this based on your actual vocabulary size
tgt_vocab_size = 1200  # Adjust this based on your actual vocabulary size

dataset = TranslationDataset(src_list, tgt_list, src_vocab_size, tgt_vocab_size, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# Create the model
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.01)



# Training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for src, tgt in dataloader:
        src = src.transpose(0, 1)  # change to [seq_len, batch_size] format
        tgt_input = tgt[:, :-1].transpose(0, 1)  # delete the last token, use as the input of decoder
        tgt_output = tgt[:, 1:].transpose(0, 1)  # delete the first token, use as the test target

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



# ------------------------------------------------------------------------



class TextDataset(Dataset):
    def __init__(self, src_list, tgt_list, vocab, max_len=16):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self, idx):
        src = encode(self.src_list[idx], self.vocab)  # sentence to index sequence
        tgt = encode(self.tgt_list[idx], self.vocab)
        
        # Pad or Cut sequences to max_len
        if len(src) < self.max_len:
            src = src + [0] * (self.max_len - len(src))  # [vocab['<pad>']] * (self.max_len - len(src))
        else:
            src = src[: self.max_len]

        if len(tgt) < self.max_len:
            tgt = tgt + [0] * (self.max_len - len(tgt))
        else:
            tgt = tgt[: self.max_len]
        
        return torch.tensor(src), torch.tensor(tgt)


# Test Data
test_src_list = src_list[:1000]

test_tgt_list = test_src_list


# create test data dataloader
dataset = TextDataset(test_src_list, test_tgt_list, vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model.eval()


# Test Medel Porformance
with torch.no_grad():
    for src, tgt in dataloader:

        print('===')
        print(src.tolist()[0])
        print(tgt.tolist()[0])
        print(decode(src.tolist()[0], vocab))
        print(decode(tgt.tolist()[0], vocab))
        print('===')

        src = src.transpose(0, 1)
        tgt_input = tgt[:, :-1].transpose(0, 1)
        
        output = model(src, tgt_input)
        output = output.argmax(dim=-1).transpose(0, 1)
        
        print('-----')
        print(output.tolist()[0])
        print(decode(output.tolist()[0], vocab))
        print('-----+')
        

torch.save(model.state_dict(), 'small_model1.pth')



