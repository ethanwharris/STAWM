import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbearer
import torchvision
from torchbearer import Trial, callbacks
from torchvision import transforms

from cifar_baseline_vae import CifarVAE


class SelfTaught(nn.Module):
    def __init__(self, encoder, mu, size):
        super(SelfTaught, self).__init__()

        self.encoder = encoder

        self.mu = mu

        self.classifier = nn.Sequential(
            nn.Linear(size, 10)
        )

    def forward(self, x, state=None):
        features = self.encoder(x)
        return F.log_softmax(self.classifier(self.mu(features)))


def evaluate(file, device='cuda'):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = 'cifarss_base'

    model = CifarVAE()
    model = SelfTaught(model.encoder, model.mu, 32)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss']).load_state_dict(torch.load(os.path.join(base_dir, file)), resume=False).with_generators(val_generator=testloader).to(device)

    return trial.evaluate()


def run(file, device='cuda'):
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

    base_dir = 'cifarss_base'

    model = CifarVAE()
    model.load_state_dict(torch.load(file)[torchbearer.MODEL])
    model = SelfTaught(model.encoder, model.mu, 32)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.mu.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    trial = Trial(model, optimizer, nn.NLLLoss(), ['acc', 'loss'], pass_state=True, callbacks=[
        callbacks.MultiStepLR([25, 40, 45]),
        callbacks.MostRecent(os.path.join(base_dir, '{epoch:02d}.pt')),
        callbacks.GradientClipping(5)
    ]).with_generators(train_generator=trainloader, val_generator=testloader).for_val_steps(5).to(device)

    trial.run(50)


if __name__ == "__main__":
    run('cifar_vae/iter_0.99.pt')
    print(evaluate('49.pt'))
