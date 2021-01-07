import torch.nn as nn

# LSTM model
class FastTextLSTM(nn.Module):
    def __init__(self, device, num_hidden_fc, num_layers, max_sequence_length, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super(FastTextLSTM, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)

        input_size = embedding_dim
        self.hidden_size = num_hidden_fc
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size * max_sequence_length, int(self.hidden_size/2))
        self.bn1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.dropout = nn.Dropout(0.5)

        # self.fc2 = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        # self.bn2 = nn.BatchNorm1d(int(self.hidden_size/2))

        self.fc3 = nn.Linear(int(self.hidden_size / 2), num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, h, c):
        embeded = self.embeddings(x)
        lstm_out, (h, c) = self.lstm(embeded, (h, c))
        batch_size = lstm_out.shape[0]
        lstm_out_flattened = lstm_out.reshape(batch_size, -1)

        out = self.bn1(self.relu(self.dropout(self.fc1(lstm_out_flattened))))
        # out = self.bn2(self.relu(self.dropout(self.fc2(out))))

        out = self.fc3(out)
        return out, h.detach(), c.detach()