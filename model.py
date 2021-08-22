import torch
import torch.nn as nn

class RengaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        # initialization
        super(RengaModel, self).__init__()

        # parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dense = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

        # gpu / cpu setting
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self = self.to(self.device)

    def forward(self, x, states):
        x = self.embedding(x)
        x, states = self.lstm(x, states)
        x = self.dense(x)
        return x, states

    def initHidden(self, batch_size):
        # batch_first does not apply to hidden or cell states
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))