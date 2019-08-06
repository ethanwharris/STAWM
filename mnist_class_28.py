import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    def __init__(self, in_planes, out_planes, stride=1, padding=0):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Block(1, 64, stride=2)
        self.conv2 = Block(64, 128, stride=2)
        self.conv3 = Block(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x


class GlimpseNet(nn.Module):
    def __init__(self):
        super(GlimpseNet, self).__init__()
        self.conv1 = Block(1, 64, stride=2)
        self.conv2 = Block(64, 128, stride=2)
        self.conv3 = Block(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x


class MnistClassifier(nn.Module):
    def __init__(self, count, memory_size):
        super(MnistClassifier, self).__init__()
        self.memory = STAWM(
            hidden_size=memory_size * 2,
            memory_size=memory_size,
            glimpse_size=28,
            g_down=1024,
            c_down=1024,
            context_net=ContextNet(),
            glimpse_net=GlimpseNet(),
            vectors=True
        )

        self.count = count
        self.drop = nn.Dropout(0.5)
        self.qdown = nn.Linear(1024, memory_size)
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


def run(count, memory_size, device='cuda'):
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

    base_dir = os.path.join('mnist_' + str(memory_size), str(count))

    model = MnistClassifier(count, memory_size)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=5e-4)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss'], callbacks=[
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.pt')),
        callbacks.GradientClipping(5),
        callbacks.MultiStepLR(milestones=[50, 100, 150, 190, 195]),
        # callbacks.ExponentialLR(0.99),
        callbacks.TensorBoard(write_graph=False, comment=base_dir)
    ]).with_train_generator(trainloader).with_val_generator(testloader).to(device)

    trial.run(200)

    trial.with_test_generator(testloader).evaluate(data_key=torchbearer.TEST_DATA)


if __name__ == "__main__":
    run(8, 512)
