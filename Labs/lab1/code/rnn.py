import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 输入文本
def tokenize_and_encode(texts):
    # 使用分词器对文本进行分词
    tokenized_texts = tokenizer(texts, padding='max_length', max_length=31, 
                                        truncation=True, return_tensors="pt")
    tokenized_texts = tokenized_texts['input_ids']
    return tokenized_texts

# 定义文本数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = tokenizer(line.strip()).view(-1)
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 参数设置
file_path = './data/corpus.txt'
vocab_size = 30000
embedding_dim = 64
hidden_dim = 256
num_layers = 2
batch_size = 16
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h):
        embedded = self.embedding(x)
        # print(embedded.shape)
        out, h = self.lstm(embedded, h)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        return out, h

# 实例化数据集和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="./cache", max_vocab_size=vocab_size)
dataset = TextDataset(file_path, tokenize_and_encode)
print(type(dataset))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0
    for i, batch_inputs in enumerate(train_loader):
        optimizer.zero_grad()
        # print(type(batch_inputs))
        # print(batch_inputs.shape)
        if batch_size != batch_inputs.shape[0]:
            continue
        h = (torch.zeros(num_layers, batch_size, hidden_dim).to(device),
             torch.zeros(num_layers, batch_size, hidden_dim).to(device))  # 初始化LSTM隐藏状态
        outputs, h = model(batch_inputs.to(device), h)
        loss = criterion(outputs.view(-1, vocab_size)[:-1, :].to(device), 
                         batch_inputs.view(-1)[1:].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Iteration {i}, Loss: {loss.item()}')

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
