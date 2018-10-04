import torch
import torch.nn as nn
import torch.nn.functional as F

import modules as m


class Memory(nn.Module):
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
    def __init__(self, dropout=0.5, decay=0.2, learn=0.4, learn2=0.5, hidden_size=512, memory_size=256, output_inverse=False, glimpse_net=None, context_net=None, glimpse_size=22, g_down=1024, c_down=1024):
        super(Memory, self).__init__()
        self.output_inverse = output_inverse

        self.drop = nn.Dropout(dropout)

        self.decay = nn.Parameter(torch.ones(1) * decay, requires_grad=True)
        self.learn = nn.Parameter(torch.ones(1) * learn, requires_grad=True)
        self.learn2 = nn.Parameter(torch.ones(1) * learn2, requires_grad=True)

        self.glimpse_cnn = glimpse_net
        self.glimpse_down = nn.Linear(g_down, hidden_size)

        self.context_cnn = context_net
        self.context_down = nn.Linear(c_down, hidden_size)

        self.emission_rnn = m.LSTM(hidden_size, hidden_size)

        self.locator = m.AffineLocator(glimpse_size=glimpse_size)
        self.where = nn.Linear(6, memory_size)

        self.emission = m.AffineEmitter(hidden_size, dropout=dropout)

        if output_inverse:
            self.inv_emission = m.AffineEmitter(hidden_size, dropout=dropout)

        self.aggregator_rnn = m.LSTM(hidden_size, hidden_size)
        self.project = nn.Linear(memory_size, hidden_size)

        self.c0_in = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.c1_in = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)

        self.what = nn.Linear(hidden_size, memory_size)

        self.A = nn.Parameter(torch.zeros(1, memory_size, memory_size), requires_grad=False)
        self.p = nn.Parameter(torch.zeros(1, memory_size), requires_grad=False)

        self.soft = nn.LogSoftmax(dim=1)

        self.h = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)

        self.bmv = m.BMV()
        self.outer_product = m.OuterProduct()

    def forward(self, x):
        """Project through the memory without the theta term for inference

        :param x: The vector (B, N) to project
        :return: The projected output (with ReLU6)
        """
        return F.relu6(self.bmv(self.A1, x))

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
        self.A1 = self.A.repeat(x.size(0), 1, 1)
        self.outer = self.A.repeat(x.size(0), 1, 1)
        self.p2 = self.p.repeat(x.size(0), 1)
        self.i = 0

        return x, context

    def glimpse(self, x, image):
        """Perform a glimpse pass and an update to the memory

        :param x: The input to the emission RNN (i.e. the output from the previous call to glimpse or the context)
        :param image: The input image mini-batch
        :return: The output from the aggregator RNN for this glimpse and the inverse affine matrix if output_inverse is True
        """
        x, self.h0, self.c0 = self.emission_rnn(x, self.h0, self.c0)
        pose = self.emission(x)

        if self.output_inverse:
            inverse = self.inv_emission(x)
        else:
            inverse = None

        where = self.where(pose)
        x = self.locator(pose, image)
        x = F.relu(self.drop(self.glimpse_down(self.glimpse_cnn(x))))
        x2, self.h1, self.c1 = self.aggregator_rnn(x, self.h1, self.c1)
        x = F.relu6(self.what(x) * where)
        p2 = self.bmv(self.A1, x, outputs=self.p2)
        p2 = self.learn * F.relu6((self.learn2 * x) + p2)
        outer = self.outer_product(x, p2, outputs=self.outer)
        self.A1 = self.A1 + outer - (self.decay * self.A1)
        self.i = self.i + 1

        if self.output_inverse:
            return x2, inverse
        else:
            return x2
