import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchbearer
import torchvision
from torchbearer import Trial, callbacks
from torchvision import transforms

import tb_modules as tm

MU = torchbearer.state_key('mu')
LOGVAR = torchbearer.state_key('logvar')


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class InverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(InverseBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CifarVAE(nn.Module):
    def __init__(self):
        super(CifarVAE, self).__init__()

        self.encoder = nn.Sequential(
            Block(3, 32, 4, 1, 2),  # B,  32, 32, 32
            nn.ReLU(True),
            Block(32, 64, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            Block(64, 128, 4, 2, 1),  # B,  64,  8, 8
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),  # B,  64, 4, 4
            nn.ReLU(True),
            View((-1, 128 * 4 * 4))
        )

        self.decoder = nn.Sequential(
            View((-1, 128, 4, 4)),
            InverseBlock(128, 128, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            InverseBlock(128, 64, 4, 2, 1),  # B,  32, 8, 8
            nn.ReLU(True),
            InverseBlock(64, 32, 4, 2, 1, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            InverseBlock(32, 3, 4, 1, 2),  # B, nc, 16, 16
        )

        self.mu = nn.Linear(2048, 32)
        self.var = nn.Linear(2048, 32)

        self.sup = nn.Linear(32, 2048)

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.div(2).exp_()
            eps = std.data.new(std.size()).normal_()
            return mu + std * eps
        else:
            return mu

    def forward(self, x, state=None):
        image = x

        features = self.encoder(x)
        mu = self.mu(features)
        logvar = self.var(features)

        sample = self.sample(mu, logvar)
        sample = self.sup(sample).relu()
        out = self.decoder(sample).sigmoid()

        if state is not None:
            state[torchbearer.Y_TRUE] = image
            state[MU] = mu
            state[LOGVAR] = logvar

        return out


def draw(file, device='cuda'):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = 'cifar_vae'

    model = CifarVAE()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['acc', 'loss'], pass_state=True, callbacks=[
        callbacks.TensorBoardImages(comment=current_time, name='Prediction', write_each_epoch=True,
                                    key=torchbearer.Y_PRED, pad_value=1, nrow=16),
        callbacks.TensorBoardImages(comment=current_time + '_cifar_vae', name='Target', write_each_epoch=False,
                                    key=torchbearer.Y_TRUE, pad_value=1, nrow=16)
    ]).load_state_dict(torch.load(os.path.join(base_dir, file)), resume=False).with_generators(train_generator=testloader, val_generator=testloader).for_train_steps(1).to(device)

    trial.run()  # Evaluate doesn't work with tensorboard in torchbearer, seems to have been fixed in most recent version


def run(iteration, device='cuda:1'):
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

    base_dir = 'cifar_vae'

    model = CifarVAE()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['acc', 'loss'], pass_state=True, callbacks=[
        tm.kl_divergence(MU, LOGVAR, beta=2),
        callbacks.MultiStepLR([50, 90]),
        callbacks.MostRecent(os.path.join(base_dir, 'iter_' + str(iteration) + '.{epoch:02d}.pt')),
        callbacks.GradientClipping(5),
        callbacks.TensorBoardImages(comment=current_time, name='Prediction', write_each_epoch=True,
                                    key=torchbearer.Y_PRED),
        callbacks.TensorBoardImages(comment=current_time + '_cifar_vae', name='Target', write_each_epoch=False,
                                    key=torchbearer.Y_TRUE),
    ]).with_generators(train_generator=trainloader, val_generator=testloader).for_val_steps(5).to(device)

    trial.run(100)


if __name__ == "__main__":
    run(0)
    draw('iter_0.99.pt')
