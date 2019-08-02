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


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Block(1, 32, size=(16, 16))
        self.conv2 = Block(32, 32, size=(8, 8))
        self.conv3 = Block(32, 32, size=(4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x


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
    def __init__(self, count, memory_size, glimpse_size, vectors):
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
        self.qdown = nn.Linear(512, memory_size)
        self.classifier = nn.Linear(memory_size, 10)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x, state=None):
        image = x
        x, context = self.memory.init(image)

        query = F.relu6(self.drop(self.qdown(context.detach())))

        for i in range(self.count):
            x = self.memory.glimpse(x, image)

        myp = self.memory(query)
        return self.soft(self.classifier(myp))


def run(count, memory_size, glimpse_size, vectors, rep, device='cuda'):
    code = '.'.join([str(count), str(glimpse_size), str(memory_size), str(rep)])
    traintransform = transforms.Compose([transforms.RandomRotation(20), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=True, transform=traintransform)
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=128,
                                              shuffle=True, num_workers=10)

    testtransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False,
                                         download=True, transform=testtransform)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=128,
                                             shuffle=False, num_workers=10)

    base_dir = 'mnist'

    model = MnistClassifier(count, memory_size, glimpse_size, vectors)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss'], callbacks=[
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.' + code + '.pt')),
        callbacks.GradientNormClipping(5),
        callbacks.MultiStepLR(milestones=[40]),
        callbacks.ExponentialLR(0.99),
        callbacks.CSVLogger(os.path.join('mnist', code + '.csv'))
        # callbacks.TensorBoard(write_graph=False, comment=base_dir)
    ]).with_train_generator(trainloader).to(device)

    trial.run(50, verbose=1)

    trial.with_test_generator(testloader).evaluate(data_key=torchbearer.TEST_DATA)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--glimpse_size", default=16, type=int, help="glimpse size")
    parser.add_argument("--mem_size", default=256, type=int, help="memory size")
    parser.add_argument("--glimpses", default=8, type=int, help="number of glimpses")
    parser.add_argument("--vectors", default=False, type=bool, help="use vector rates?")
    parser.add_argument("--reps", default=5, type=int, help="number of repeats")

    args = parser.parse_args()

    for i in range(args.reps):
        run(args.glimpses, args.mem_size, args.glimpse_size, args.vectors, i)
