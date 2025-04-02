import torch
import torch.nn as nn
import torch.optim as optim

class BaselineModel(nn.Module):
    def __init__(self, d_vocab, d_hidden, d_embed, lr):
        super().__init__()
        self.embed1 = nn.Linear(d_vocab, d_hidden)
        self.relu1 = nn.ReLU()
        self.embed2 = nn.Linear(d_hidden, d_embed)
        self.relu2 = nn.ReLU()

        self.out1 = nn.Linear(d_embed, d_hidden)
        self.relu3 = nn.ReLU()
        self.out2 = nn.Linear(d_hidden, d_vocab)
        self.softmax = nn.Softmax()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

    def forward(self, x):
        # Embed input into dense vector
        x = self.embed1(x)
        x = self.relu1(x)
        x = self.embed2(x)
        x = self.relu2(x)

        # Predict next word
        x = self.out1(x)
        x = self.relu3(x)
        x = self.out2(x)
        x = self.softmax(x)

        return x

    
