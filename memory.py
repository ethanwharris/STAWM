import torch
import torch.nn as nn
import torch.nn.functional as F

import modules as m


class Memory(nn.Module):
    def __init__(self, size, vectors=True, with_delta=True):
        super(Memory, self).__init__()
        self.vectors = vectors
        self.with_delta = with_delta

        self.W = nn.Parameter(torch.zeros(1, size, size), requires_grad=False)

        # self.ln = nn.LayerNorm(size)
        # self.bn = nn.BatchNorm1d(size)

        if vectors:
            self.delta = nn.Parameter(torch.randn(size), requires_grad=True)
            self.eta = nn.Parameter(torch.randn(size), requires_grad=True)
            self.theta = nn.Parameter(torch.randn(size), requires_grad=True)
        else:
            self.delta = nn.Parameter(torch.randn(1), requires_grad=True)
            self.eta = nn.Parameter(torch.randn(1), requires_grad=True)
            self.theta = nn.Parameter(torch.randn(1), requires_grad=True)

    def init(self, x):  # Make tensors
        self.W1 = self.W.repeat(x.size(0), 1, 1)

    def forward(self, x):  # Learn
        y = torch.matmul(self.W1, x.unsqueeze(2)).squeeze(2)
        y = self.eta.sigmoid() * F.relu6((self.theta.sigmoid() * x) + y)
        outer = y.unsqueeze(2) * x.unsqueeze(1)  # self.outer_product(x, p)

        if self.with_delta:
            delta = self.delta
        else:
            delta = self.eta

        if self.vectors:
            delta = self.delta.unsqueeze(1)

        self.W1 = outer - (1 - delta.sigmoid()) * self.W1
        return y

    def query(self, x):  # Query
        x = torch.matmul(self.W1, x.unsqueeze(2)).squeeze(2)
        return F.relu6(x)


class STAWM(nn.Module):
    """The Memory module to be used in each experiment.

    :param dropout: Amount of dropout to use
    :param decay: Value of the decay rate delta
    :param learn: Value of the first learning rate eta
    :param learn2: Value of the second learning rate theta
    :param hidden_size: Hidden size of the LSTMs
    :param memory_size: Memory size
    :param output_inverse: If True output the inverse transform also
    :param glimpse_net: The network to use for the glimpse features
    :param context_net: The network to use for the context features
    :param glimpse_size: The size (S) of each glimpse
    :param g_down: The size of the glimpse_net output
    :param c_down: The size of the context_net output
    """
    def __init__(self, dropout=0.5, vectors=False, with_delta=False, hidden_size=512, memory_size=256, output_inverse=False, glimpse_net=None, context_net=None, glimpse_size=22, g_down=1024, c_down=1024):
        super(STAWM, self).__init__()
        self.output_inverse = output_inverse

        self.memory = Memory(memory_size, vectors=vectors, with_delta=with_delta)

        self.drop = nn.Dropout(dropout)

        self.glimpse_cnn = glimpse_net
        self.glimpse_down = nn.Linear(g_down, hidden_size)

        self.context_cnn = context_net
        self.context_down = nn.Linear(c_down, hidden_size)

        self.emission_rnn = m.LSTM(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)

        self.locator = m.AffineLocator(glimpse_size=glimpse_size)
        self.where = nn.Linear(6, memory_size)

        self.emission = m.AffineEmitter(hidden_size, dropout=dropout)

        if output_inverse:
            self.inv_emission = m.AffineEmitter(hidden_size, dropout=dropout)

        self.aggregator_rnn = m.LSTM(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(memory_size, hidden_size)

        self.c0_in = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.c1_in = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)

        self.what = nn.Linear(hidden_size, memory_size)

        self.soft = nn.LogSoftmax(dim=1)

        self.h = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)

    def forward(self, x):
        """Project through the memory without the theta term for inference

        :param x: The vector (B, N) to project
        :return: The projected output (with ReLU6)
        """
        return self.memory.query(x)

    def init(self, image):
        """Inititalise the memory, should be called at the beginning of a forward pass.

        :param image: The input image mini-batch (B, C, H, W)
        :return: The image context after projection to the memory size and before as a tuple
        """
        context = self.context_cnn(image)
        x = F.relu(self.drop(self.context_down(context)))

        self.h0 = x
        self.c0 = self.c0_in.repeat(x.size(0), 1)
        self.c1 = self.c1_in.repeat(x.size(0), 1)
        self.h1 = self.h.repeat(x.size(0), 1)
        self.i = 0

        self.memory.init(x)

        return x, context

    def glimpse(self, x, image):
        """Perform a glimpse pass and an update to the memory

        :param x: The input to the emission RNN (i.e. the output from the previous call to glimpse or the context)
        :param image: The input image mini-batch
        :return: The output from the aggregator RNN for this glimpse and the inverse affine matrix if output_inverse is True
        """
        x, self.h0, self.c0 = self.emission_rnn(x, self.h0, self.c0)
        # x = self.bn1(x)
        x = self.ln1(x)
        pose = self.emission(x)

        if self.output_inverse:
            inverse = self.inv_emission(x)
        else:
            inverse = None

        where = self.where(pose)
        x = self.locator(pose, image)
        x = F.relu(self.drop(self.glimpse_down(self.glimpse_cnn(x))))
        x2, self.h1, self.c1 = self.aggregator_rnn(x, self.h1, self.c1)
        x2 = self.ln2(x2)
        # x2 = self.bn2(x2)
        x = F.relu6(self.what(x) * where)

        self.memory(x)
        self.i = self.i + 1

        if self.output_inverse:
            return x2, inverse
        else:
            return x2
