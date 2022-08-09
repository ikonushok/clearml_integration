import torch
import torch.nn as nn
import torch.nn.functional as F

#файл с классом модели

class Model(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, num_classes, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim, out_channels=256,
            kernel_size=7,
            padding=(3 * seq_len - 1 + 7) // 2, stride=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=256, out_channels=128,
            kernel_size=7, padding=(seq_len - 2 + 7) // 2, stride=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=64,
            kernel_size=5, padding=(seq_len - 1 + 5) // 2, stride=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=32,
            kernel_size=5, padding=(seq_len - 1 + 5) // 2, stride=1
        )

        self.bn12 = nn.BatchNorm1d(num_features=256)
        self.bn23 = nn.BatchNorm1d(num_features=128)
        self.bn34 = nn.BatchNorm1d(num_features=64)

        self.dropout = nn.Dropout(0.3)
        self.pooling = nn.MaxPool1d(32)

        self.fc1 = nn.Linear(2303, num_classes)

    def forward(self, text):
        embedded_text = self.embedding(text)
        embedded_text = embedded_text.permute(0, 2, 1)

        out = self.dropout(self.bn12(F.relu(self.conv1(embedded_text))))
        out = self.dropout(self.bn23(F.relu(self.conv2(out))))
        out = self.dropout(self.bn34(F.relu(self.conv3(out))))

        out = F.relu(self.conv4(out))
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.shape[2]).squeeze(2)

        return self.fc1(out)
