import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        # self.fc = nn.Linear(32*4*4, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 32*4*4)

class relation_module(nn.Module):
    def __init__(self):
        super(relation_module, self).__init__()
        self.fc1 = nn.Linear(256*2, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return x

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 13)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 8, 13)

class panels_to_embeddings(nn.Module):
    def __init__(self, tag):
        super(panels_to_embeddings, self).__init__()
        self.in_dim = 512
        if tag:
            self.in_dim += 9
        self.fc = nn.Linear(self.in_dim, 256)

    def forward(self, x):
        return self.fc(x.view(-1, self.in_dim))

class WReN(BasicModel):
    def __init__(self, args):
        super(WReN, self).__init__(args)
        self.conv = conv_module()
        self.rn = relation_module()
        self.mlp = mlp_module()
        self.proj = panels_to_embeddings(args.tag)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_beta = args.meta_beta 
        self.use_tag = args.tag
        self.use_cuda = args.cuda
        self.tags = self.tag_panels(args.batch_size)

    def tag_panels(self, batch_size):
        tags = []
        for idx in range(0, 16):
            tag = np.zeros([1, 9], dtype=float)
            if idx < 8:
                tag[:, idx] = 1.0
            else:
                tag[:, 8] = 1.0
            tag = torch.tensor(tag, dtype=torch.float).expand(batch_size, -1).unsqueeze(1)
            if self.use_cuda:
                tag = tag.cuda()
            tags.append(tag)
        tags = torch.cat(tags, dim=1)
        return tags

    def group_panel_embeddings(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        embeddings_seq = torch.chunk(embeddings, 16, dim=1)
        context_pairs = []
        for context_idx1 in range(0, 8):
            for context_idx2 in range(0, 8):
                if not context_idx1 == context_idx2:
                    context_pairs.append(torch.cat((embeddings_seq[context_idx1], embeddings_seq[context_idx2]), dim=2))
        context_pairs = torch.cat(context_pairs, dim=1)
        panel_embeddings_pairs = []
        for answer_idx in range(8, len(embeddings_seq)):
            embeddings_pairs = context_pairs
            for context_idx in range(0, 8):
                # In order
                order = torch.cat((embeddings_seq[answer_idx], embeddings_seq[context_idx]), dim=2)
                reverse = torch.cat((embeddings_seq[context_idx], embeddings_seq[answer_idx]), dim=2)
                choice_pairs = torch.cat((order, reverse), dim=1)
                embeddings_pairs = torch.cat((embeddings_pairs, choice_pairs), dim=1)
            panel_embeddings_pairs.append(embeddings_pairs.unsqueeze(1))
        panel_embeddings_pairs = torch.cat(panel_embeddings_pairs, dim=1)
        return panel_embeddings_pairs.view(-1, 8, 72, 512)

    def group_panel_embeddings_batch(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        context_embeddings = embeddings[:,:8,:]
        choice_embeddings = embeddings[:,8:,:]
        context_embeddings_pairs = torch.cat((context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1), context_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)), dim=3).view(-1, 64, 512)
        
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1)
        choice_embeddings = choice_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)
        choice_context_order = torch.cat((context_embeddings, choice_embeddings), dim=3)
        choice_context_reverse = torch.cat((choice_embeddings, context_embeddings), dim=3)
        embedding_paris = [context_embeddings_pairs.unsqueeze(1).expand(-1, 8, -1, -1), choice_context_order, choice_context_reverse]
        return torch.cat(embedding_paris, dim=2).view(-1, 8, 80, 512)


    def rn_sum_features(self, features):
        features = features.view(-1, 8, 80, 256)
        sum_features = torch.sum(features, dim=2)
        return sum_features

    def compute_loss(self, output, target, meta_target):
        pred, meta_pred = output[0], output[1]
        target_loss = F.cross_entropy(pred, target)
        meta_pred = torch.chunk(meta_pred, chunks=12, dim=1)
        meta_target = torch.chunk(meta_target, chunks=12, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 12):
            meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_pred[idx]), meta_target[idx])
        loss = target_loss + self.meta_beta*meta_target_loss / 12.
        return loss

    def forward(self, x):
        panel_features = self.conv(x.view(-1, 1, 80, 80))
        # print(panel_embeddings.size())
        if self.use_tag:
            panel_features = torch.cat((panel_features, self.tags), dim=2)
        panel_embeddings = self.proj(panel_features)
        # panel_embeddings_pairs = self.group_panel_embeddings(panel_embeddings)
        # self.group_panel_embeddings(panel_embeddings)
        panel_embeddings_pairs = self.group_panel_embeddings_batch(panel_embeddings)
        # print(panel_embeddings_pairs.size())
        panel_embedding_features = self.rn(panel_embeddings_pairs.view(-1, 512))
        # print(panel_embedding_features.size())
        sum_features = self.rn_sum_features(panel_embedding_features)
        output = self.mlp(sum_features.view(-1, 256))
        pred = output[:,:,12]
        meta_pred = torch.sum(output[:,:,0:12], dim=1)
        return pred, meta_pred