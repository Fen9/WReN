import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 8*4*4)

class lstm_module(nn.Module):
    def __init__(self):
        super(lstm_module, self).__init__()
        self.lstm = nn.LSTM(input_size=8*4*4+9, hidden_size=96, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96, 8)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        # hidden = torch.zeros(1, x.size()[1], 96)
        # cell = torch.zeros(1, x.size()[1], 96)
        # print(hidden.size())
        # print(cell.size())
        hidden, _ = self.lstm(x)
        score = self.fc(hidden[-1, :, :])
        return score

class CNN_LSTM(BasicModel):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__(args)
        self.conv = conv_module()
        self.lstm = lstm_module()
        self.register_buffer("tags", torch.tensor(self.build_tags(), dtype=torch.float))
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def build_tags(self):
        tags = np.zeros((16, 9))
        tags[:8, :8] = np.eye(8)
        tags[8:, 8] = 1
        return tags

    def compute_loss(self, output, target, _):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
        batch = x.shape[0]
        features = self.conv(x.view(-1, 1, 80, 80))
        features = torch.cat([features, self.tags.unsqueeze(0).expand(batch, -1, -1)], dim=-1)
        score = self.lstm(features)
        return score, None

    