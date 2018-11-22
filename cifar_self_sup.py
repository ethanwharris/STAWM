import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbearer
import torchvision
from torchbearer import Trial, callbacks
from torchvision import transforms

from cifar_draw_16 import CifarDraw


class SelfTaught(nn.Module):
    def __init__(self, count, q_down, memory_size, memory):
        super(SelfTaught, self).__init__()

        self.memory = memory

        self.count = count

        self.qdown = nn.Linear(q_down, memory_size)

        self.classifier = nn.Sequential(
            nn.Linear(memory_size, 10)
        )

    def forward(self, x, state=None):
        image = x
        x, context = self.memory.init(image)

        query = F.relu6(self.qdown(context))

        for i in range(self.count):
            x, _ = self.memory.glimpse(x, image)

        return F.log_softmax(self.classifier(self.memory(query)))


def evaluate(count, memory_size, file, device='cuda'):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = os.path.join('cifarss_' + str(memory_size), "16")

    model = CifarDraw(count, memory_size)
    model = SelfTaught(count, 512, memory_size, model.memory)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss']).load_state_dict(torch.load(os.path.join(base_dir, file)), resume=False).with_generators(val_generator=testloader).to(device)

    return trial.evaluate()


def run(count, memory_size, file, device='cuda'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = os.path.join('cifarss_' + str(memory_size), "16")

    model = CifarDraw(count, memory_size)
    model.load_state_dict(torch.load(file)[torchbearer.MODEL])
    model = SelfTaught(count, 512, memory_size, model.memory)
    for param in model.memory.parameters():
        param.requires_grad = False

    model.memory.decay.requires_grad = True
    model.memory.learn.requires_grad = True
    model.memory.learn2.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss'], pass_state=True, callbacks=[
        callbacks.MultiStepLR([25, 40, 45]),
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.pt')),
        callbacks.GradientClipping(5)
    ]).with_generators(train_generator=trainloader, val_generator=testloader).for_val_steps(5).to(device)

    trial.run(50)


if __name__ == "__main__":
    run(8, 256, 'cifar_256/16/iter_0.99.pt')
    print(evaluate(8, 256, '49.pt'))
