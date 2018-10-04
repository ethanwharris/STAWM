import os

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory

import torchbearer
from torchbearer import Trial, callbacks

from mobilenetv2 import MobileNetV2


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
        self.conv1 = Block(3, 64, stride=2)
        self.conv2 = Block(64, 128, stride=2)
        self.conv3 = Block(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x


class CifarClassifier(nn.Module):
    def __init__(self, count, memory_size):
        super(CifarClassifier, self).__init__()
        self.memory = Memory(
            hidden_size=memory_size * 2,
            memory_size=memory_size,
            glimpse_size=32,
            g_down=1280,
            c_down=2304,
            context_net=ContextNet(),
            glimpse_net=MobileNetV2()
        )

        self.count = count
        self.drop = nn.Dropout(0.5)
        self.qdown = nn.Linear(2304, memory_size)
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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = os.path.join('cifar_' + str(memory_size), str(count))

    model = nn.DataParallel(CifarClassifier(count, memory_size))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=5e-4)

    trial = Trial(model, optimizer, nn.NLLLoss(), [torchbearer.metrics.CategoricalAccuracy(), 'loss'], callbacks=[
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.pt')),
        callbacks.GradientClipping(5),
        callbacks.MultiStepLR(milestones=[150, 250]),
        callbacks.TensorBoard(write_graph=False, comment=base_dir)
    ]).with_train_generator(trainloader).to(device)

    trial.run(350)

    trial.with_test_generator(testloader).evaluate(data_key=torchbearer.TEST_DATA)


if __name__ == "__main__":
    run(4, 1024)
