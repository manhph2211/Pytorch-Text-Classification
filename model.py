import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, recall_score



class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, json_path='./data/encode_dictionary.json'):
        super(RNN, self).__init__()
        self.embedding = self.make_embedding_layer(json_path)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, nonlinearity='tanh')
        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.classifier = nn.Softmax()

    def forward(self, X):
        out = self.embedding(X)
        out = out.permute(1, 0, 2)
        out, hidden = self.rnn(out)
        out = out[-1, :, :]
        out = self.linear1(out)
        out = self.classifier(out)

        return out


