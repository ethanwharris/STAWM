import torch
import torch.nn as nn
import torch.nn.functional as F


class BMV(nn.Module):
    def __init__(self):
        super(BMV, self).__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y.unsqueeze(2)).squeeze(2)


class OuterProduct(nn.Module):
    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, x, y, **kwargs):
        return x.unsqueeze(2) * y.unsqueeze(1)


class AffineLocator(nn.Module):
    def __init__(self, glimpse_size=22):
        super(AffineLocator, self).__init__()
        self.glimpse_size = glimpse_size

    def forward(self, theta, x):
        theta = theta.view(x.size(0), 2, 3)

        grid = F.affine_grid(theta, torch.Size((x.size(0), x.size(1), self.glimpse_size, self.glimpse_size)))
        x = F.grid_sample(x, grid)
        return x


class AffineEmitter(nn.Module):
    def __init__(self, size, output_inverse=False, dropout=0.3):
        super(AffineEmitter, self).__init__()

        self.fc1 = nn.Linear(size, int(size / 2))

        if output_inverse:
            self.fc3 = nn.Linear(int(size / 2), 12)
            self.fc3.weight.data.fill_(0)
            self.fc3.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        else:
            self.fc3 = nn.Linear(int(size / 2), 6)
            self.fc3.weight.data.fill_(0)
            self.fc3.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

        self.drop = nn.Dropout(p=dropout)
        self.nl = nn.ReLU()

    def forward(self, x):
        x = self.nl(self.drop(self.fc1(x)))
        x = self.fc3(x)
        return x


class LSTM(nn.Module):
    def __init__(self, size_in, size_out):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTMCell(size_in, size_out)

    def forward(self, x, h, c):
        x, c = self.rnn(x, (h, c))
        h = x
        return x, h, c
