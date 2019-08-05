import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbearer
import torchvision
from torchbearer import Trial, callbacks
from torchvision import transforms

from memory import STAWM
from memnist import MemNIST


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, size, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=padding, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(size)
        self.bn = nn.BatchNorm2d(out_planes)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        out = F.relu(self.bn(self.pool(self.conv(x))))
        return out


# class ContextNet(nn.Module):
#     def __init__(self):
#         super(ContextNet, self).__init__()
#         self.conv1 = Block(1, 32, size=(16, 16))
#         self.conv2 = Block(32, 32, size=(8, 8))
#         self.conv3 = Block(32, 32, size=(4, 4))
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)
#         return x


class ContextNet(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 512, device=x.device)


class GlimpseNet(nn.Module):
    def __init__(self):
        super(GlimpseNet, self).__init__()
        self.conv1 = Block(1, 32, size=(16, 16))
        self.conv2 = Block(32, 32, size=(8, 8))
        self.conv3 = Block(32, 32, size=(4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x


class MnistClassifier(nn.Module):
    def __init__(self, count, memory_size, glimpse_size, vectors, number):
        super(MnistClassifier, self).__init__()
        self.memory = STAWM(
            vectors=vectors,
            hidden_size=memory_size,
            memory_size=memory_size,
            glimpse_size=glimpse_size,
            g_down=512,
            c_down=512,
            context_net=ContextNet(),
            glimpse_net=GlimpseNet()
        )

        self.count = count
        self.drop = nn.Dropout(0.5)
        # self.qdown = nn.Linear(512, memory_size)
        self.classifier = nn.Linear(memory_size, 10)
        self.soft = nn.LogSoftmax(dim=1)

        queries = []
        for _ in range(number):
            queries.append(nn.Parameter(torch.rand(memory_size) * 6, requires_grad=True))
        self.queries = nn.ParameterList(queries)

    def forward(self, images, state):
        x, context = self.memory.init(images[:, 0].unsqueeze(1))

        for n in range(images.size(1)):
            image = images[:, n].unsqueeze(1)

            for _ in range(self.count):
                x = self.memory.glimpse(x, image)

        classes = []
        for n in range(images.size(1)):
            myp = self.memory(self.queries[n].unsqueeze(0))
            classes.append(self.soft(self.classifier(myp)))

        state[torchbearer.TARGET] = state[torchbearer.TARGET].t().contiguous().view(-1)

        return torch.cat(classes, dim=0)


def run(count, memory_size, glimpse_size, vectors, rep, number=6, root='memnist', device='cuda'):
    code = '.'.join([str(count), str(glimpse_size), str(memory_size), str(rep)])
    traintransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MemNIST(root='./data/mnist', number=number, train=True, download=True, transform=traintransform)
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=128,
                                              shuffle=True, num_workers=10)

    testtransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = MemNIST(root='./data/mnist', number=number, train=False, download=True, transform=testtransform)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=128,
                                             shuffle=False, num_workers=10)

    base_dir = root

    model = MnistClassifier(count, memory_size, glimpse_size, vectors, number)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

    # crit = nn.NLLLoss(reduction='none')
    #
    # def loss(state):
    #     l = crit(state[torchbearer.PREDICTION], state[torchbearer.TARGET]).view(number, -1)
    #     # print(l.size())
    #     return l.sum(0).mean(0)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss'], callbacks=[
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.' + code + '.pt')),
        callbacks.GradientNormClipping(5),
        callbacks.MultiStepLR(milestones=[40]),
        callbacks.ExponentialLR(0.99),
        # callbacks.CSVLogger(os.path.join('mnist', code + '.csv'))
        # callbacks.TensorBoard(write_graph=False, comment=base_dir)
        # callbacks.imaging.MakeGrid().on_train().to_file('test.png')
    ]).with_train_generator(trainloader).to(device)

    trial.run(20, verbose=2)

    history = trial.with_test_generator(testloader).evaluate(data_key=torchbearer.TEST_DATA)
    acc = history['test_acc']

    with open(os.path.join(base_dir, '.'.join([str(count), str(glimpse_size), str(memory_size), 'txt'])), 'a+') as f:
        f.write(str(acc) + '\n')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--glimpse_sizes", default=[16], nargs='+', type=int, help="glimpse size")
    parser.add_argument("--mem_size", default=256, type=int, help="memory size")
    parser.add_argument("--glimpses", default=[1], nargs='+', type=int, help="number of glimpses")
    parser.add_argument("--vectors", default=False, type=bool, help="use vector rates?")
    parser.add_argument("--reps", default=5, type=int, help="number of repeats")
    parser.add_argument("--number", default=10, type=int, help="number of images to remember")
    parser.add_argument("--root", default='memnist', type=str, help="base directory")

    args = parser.parse_args()

    for num_glimpses in args.glimpses:
        for glimpse_size in args.glimpse_sizes:
            for i in range(args.reps):
                run(num_glimpses, args.mem_size, glimpse_size, args.vectors, i, number=args.number, root=args.root)
